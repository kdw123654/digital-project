"""Portia-inspired sparse spiking classifier with local, label-modulated STDP.

This is a new implementation of the design proposal, not a biological connectome.
Pulse time is internal computation over one static row, not patient time. Sparse
LIF dynamics, signed E/I recurrent wiring, timing plasticity and homeostasis are
retained. Labels modulate local synaptic eligibility during fit only; neither
backpropagation nor a frozen logistic anchor is used. Four supervised logistic
readouts remain explicit parts of the classifier.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


class ImprovedPortia:
    """Compact spiking reservoir with supervised three-factor plasticity.

    ``learning=False`` disables synaptic plasticity while retaining rate
    homeostasis. ``mechanism_scale`` controls the strength of new features in the
    regularized readout; it never rescales the original input basis. Fit receives
    training rows only. Any hyperparameter selection belongs in a separate
    validation split, outside this class.
    """

    def __init__(self, plasticity=.003, C=.1, epochs=4, seed=42,
                 neurons=32, steps=24, learning=True, mechanism_scale=.5,
                 feature_rank=24, supervised_strength=1.):
        if not np.isfinite(plasticity) or plasticity < 0:
            raise ValueError("plasticity must be finite and nonnegative")
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be finite and positive")
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not isinstance(neurons, int) or neurons < 8:
            raise ValueError("neurons must be an integer >= 8")
        if not isinstance(steps, int) or steps < 4:
            raise ValueError("steps must be an integer >= 4")
        if not isinstance(feature_rank, int) or feature_rank < 1:
            raise ValueError("feature_rank must be a positive integer")
        if not np.isfinite(mechanism_scale) or mechanism_scale < 0:
            raise ValueError("mechanism_scale must be finite and nonnegative")
        if not np.isfinite(supervised_strength) or supervised_strength < 0:
            raise ValueError("supervised_strength must be finite and nonnegative")
        self.plasticity = float(plasticity)
        self.C = float(C)
        self.epochs = epochs
        self.seed = seed
        self.neurons = neurons
        self.steps = steps
        self.learning = bool(learning)
        self.mechanism_scale = float(mechanism_scale)
        self.feature_rank = feature_rank
        self.supervised_strength = float(supervised_strength)

    @staticmethod
    def _matrix(x):
        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 2 or not a.shape[0] or not a.shape[1]:
            raise ValueError("x must be a nonempty two-dimensional matrix")
        if not np.isfinite(a).all():
            raise ValueError("x must be finite; impute within the training fold")
        return a

    def _inference_input(self, x):
        if not getattr(self, "fitted_", False):
            raise ValueError("fit must be called before prediction")
        a = self._matrix(x)
        if a.shape[1] != self.n_features_in_:
            raise ValueError("input feature count differs from fit")
        return a

    def _rates(self, x):
        z = np.clip((x - self.input_mean) / self.input_scale, -5., 5.)
        return (.08 + .84 * expit(z)).astype(np.float32)

    def _simulate(self, x, *, modulation=None, weights=None):
        """Fresh per-row states; accumulated updates apply after the episode.

        Eligibility is pre-before-post LTP minus post-before-pre LTD. The third
        factor is a bounded, prevalence-centered label signal assigned to each
        neuron. Positive and negative label-selective neurons coexist. No labels
        are injected into the membrane drive, avoiding train/inference mismatch.
        """
        rate = self._rates(x)
        n = len(x)
        voltage = np.zeros((n, self.neurons), dtype=np.float32)
        spike = np.zeros_like(voltage)
        trace = np.zeros_like(voltage)
        pre_trace = np.zeros_like(rate)
        count = np.zeros_like(voltage)
        early = np.zeros_like(voltage)
        moment = np.zeros_like(voltage)
        delta_recurrent = np.zeros_like(self.recurrent)
        delta_input = np.zeros_like(self.input_weights)
        for t in range(self.steps):
            pulse = (np.floor((t + 1) * rate + self.pulse_phase)
                     - np.floor(t * rate + self.pulse_phase)).astype(np.float32)
            signal = (2. * pulse - 1.) @ self.input_weights
            drive = .08 + .48 * expit(1.4 * signal)
            voltage = .85 * voltage + drive + .35 * (spike @ self.recurrent.T)
            new = (voltage >= self.threshold).astype(np.float32)
            voltage = np.where(new > 0., voltage - self.threshold, voltage)
            if modulation is not None:
                # Old traces exclude simultaneous co-spikes. No online change
                # to circuit weights occurs between people or simulation steps.
                reward_post = modulation * weights[:, None]
                post = new * reward_post
                post_trace = trace * reward_post
                delta_recurrent += post.T @ trace - 1.05 * post_trace.T @ new
                delta_input += pre_trace.T @ post - 1.05 * pulse.T @ post_trace
            trace = .8 * trace + .2 * new
            pre_trace = .8 * pre_trace + .2 * pulse
            spike = new
            count += new
            if t < self.steps // 2:
                early += new
            moment += new * ((t + 1.) / self.steps)
        firing = count / self.steps
        temporal_contrast = (2. * early - count) / self.steps
        timing = np.divide(moment, count, out=np.zeros_like(moment), where=count > 0)
        features = np.column_stack([firing, temporal_contrast, timing, voltage])
        return features, delta_recurrent, delta_input

    def _fit_feature_map(self, x, raw, w):
        # Standardization and rank reduction apply ONLY to the new dynamical
        # features. The original basis is concatenated bit-for-bit unchanged.
        self.mechanism_mean = np.average(raw, axis=0, weights=w)
        var = np.average((raw - self.mechanism_mean) ** 2, axis=0, weights=w)
        self.mechanism_std = np.sqrt(np.maximum(var, 1e-8))
        a = (raw - self.mechanism_mean) / self.mechanism_std
        # Remove linear copies of the already-present basis, with ridge solely
        # for numerical stability. This is not a target-trained LR anchor.
        self.residual_center = np.average(x, axis=0, weights=w)
        centered = x - self.residual_center
        weighted_x = centered * np.sqrt(w[:, None])
        gram = weighted_x.T @ weighted_x
        ridge = 1e-3 * max(float(np.trace(gram)) / x.shape[1], 1.)
        self.mechanism_linear = np.linalg.solve(
            gram + ridge * np.eye(x.shape[1]), centered.T @ (w[:, None] * a))
        residual = a - centered @ self.mechanism_linear
        covariance = (residual * w[:, None]).T @ residual
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        cutoff = max(float(eigenvalues[order[0]]) * 1e-5, 1e-8)
        keep = order[eigenvalues[order] > cutoff][:self.feature_rank]
        self.mechanism_components = eigenvectors[:, keep].T
        # Cap whitening to avoid amplifying near-constant rate coordinates.
        self.mechanism_pca_scale = np.sqrt(np.maximum(eigenvalues[keep], .25))
        return self._feature_map(x, raw)

    def _feature_map(self, x, raw):
        a = (raw - self.mechanism_mean) / self.mechanism_std
        residual = a - (x - self.residual_center) @ self.mechanism_linear
        extra = (residual @ self.mechanism_components.T) / self.mechanism_pca_scale
        return np.column_stack([x, self.mechanism_scale * extra])

    def fit(self, x, y, w):
        self.fitted_ = False
        x = self._matrix(x)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        if y.shape != (len(x), 4) or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("y must have shape (n, 4) and binary values")
        if w.shape != (len(x),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("w must be finite, nonnegative, and have positive sum")
        self.n_features_in_ = x.shape[1]
        normalized = w / w.sum()
        self.input_mean = np.average(x, axis=0, weights=w)
        variance = np.average((x - self.input_mean) ** 2, axis=0, weights=w)
        self.input_scale = np.sqrt(np.maximum(variance, 1e-8))
        self.input_scale[variance < 1e-8] = 1.
        rng = np.random.default_rng(self.seed)
        self.input_mask = rng.random((x.shape[1], self.neurons)) < .35
        self.input_weights = (rng.normal(0., 1. / np.sqrt(max(x.shape[1] * .35, 1.)),
                                       self.input_mask.shape) * self.input_mask).astype(np.float32)
        self.mask = ((rng.random((self.neurons, self.neurons)) < .2)
                     & ~np.eye(self.neurons, dtype=bool))
        self.sign = np.where(np.arange(self.neurons) < int(.8 * self.neurons), 1., -1.)
        self.recurrent = (rng.uniform(.02, .12, self.mask.shape) * self.mask
                          * self.sign[None, :]).astype(np.float32)
        self.threshold = np.ones(self.neurons, dtype=np.float32)
        self.pulse_phase = rng.uniform(0., 1., x.shape[1]).astype(np.float32)
        self.initial_recurrent = self.recurrent.copy()
        self.initial_input_weights = self.input_weights.copy()
        target = np.arange(self.neurons) % 4
        orientation = np.where((np.arange(self.neurons) // 4) % 2 == 0, 1., -1.)
        prevalence = np.average(y, axis=0, weights=w)
        teacher = (y - prevalence) / np.sqrt(np.maximum(prevalence * (1. - prevalence), .04))
        modulation = (.05 + self.supervised_strength * np.clip(teacher[:, target], -3., 3.)
                      * orientation).astype(np.float32)
        history = []
        for epoch in range(self.epochs):
            enabled = self.learning and self.plasticity > 0
            raw, dr, di = self._simulate(x, modulation=modulation if enabled else None,
                                          weights=normalized.astype(np.float32))
            rates = np.average(raw[:, :self.neurons], axis=0, weights=w)
            if enabled:
                excitatory = self.mask & (self.sign[None, :] > 0)
                # Normalized eligibility prevents small event covariances from
                # silently switching off plasticity as the dataset grows. A
                # bounded update leaves ``plasticity`` as the maximum per-epoch
                # contact change; normalization uses this training batch only.
                dr_scale = max(float(np.sqrt(np.mean(dr[excitatory] ** 2))), 1e-5)
                di_scale = max(float(np.sqrt(np.mean(di[self.input_mask] ** 2))), 1e-5)
                dr = np.tanh(dr / dr_scale)
                di = np.tanh(di / di_scale)
                self.recurrent[excitatory] = np.clip(
                    self.recurrent[excitatory] + self.plasticity * dr[excitatory], 0., .3)
                self.input_weights = np.clip(self.input_weights + self.plasticity * di, -1., 1.)
                self.input_weights *= self.input_mask
            self.threshold = np.clip(self.threshold + .2 * (rates - .18), .75, 1.4).astype(np.float32)
            history.append({"epoch": epoch + 1, "mean_firing": float(rates.mean()),
                            "recurrent_change_norm": float(np.linalg.norm(self.recurrent - self.initial_recurrent)),
                            "input_change_norm": float(np.linalg.norm(self.input_weights - self.initial_input_weights))})
        raw, _, _ = self._simulate(x)
        features = self._fit_feature_map(x, raw.astype(float), normalized)
        self.readout = []
        for j in range(4):
            if not 0. < prevalence[j] < 1.:
                self.readout.append(float(prevalence[j]))
            else:
                self.readout.append(LogisticRegression(C=self.C, max_iter=3000,
                                                       random_state=self.seed).fit(features, y[:, j], sample_weight=w))
        initial_recurrent = self.recurrent.copy()
        initial_input = self.input_weights.copy()
        self.recurrent = self.initial_recurrent.copy()
        self.input_weights = self.initial_input_weights.copy()
        no_plastic_raw, _, _ = self._simulate(x)
        self.recurrent = initial_recurrent
        self.input_weights = initial_input
        self.plasticity_change_norm = float(np.linalg.norm(self.recurrent - self.initial_recurrent))
        self.mean_firing = float(np.average(raw[:, :self.neurons].mean(1), weights=w))
        self.diagnostics = {
            "mechanism": "sparse LIF/EI, finite rate pulses, label-modulated local STDP, homeostasis",
            "patient_time_series": False, "backpropagation": False, "frozen_lr_anchor": False,
            "learning_enabled": bool(self.learning and self.plasticity > 0),
            "supervised_strength": self.supervised_strength,
            "train_rows": len(x), "neurons": self.neurons, "simulation_steps": self.steps,
            "mean_firing": self.mean_firing,
            "silent_neuron_fraction": float(np.mean(raw[:, :self.neurons].sum(0) == 0)),
            "recurrent_change_norm": self.plasticity_change_norm,
            "input_change_norm": float(np.linalg.norm(self.input_weights - self.initial_input_weights)),
            "plasticity_firing_change_fraction": float(np.mean(raw[:, :self.neurons] != no_plastic_raw[:, :self.neurons])),
            "plasticity_feature_mean_absolute_change": float(np.mean(np.abs(raw - no_plastic_raw))),
            "new_feature_rank": int(len(self.mechanism_components)),
            "original_basis_unchanged": True,
            "history": history,
        }
        self.fitted_ = True
        return self

    def transform(self, x):
        x = self._inference_input(x)
        raw, _, _ = self._simulate(x)
        return self._feature_map(x, raw)

    def predict(self, x):
        f = self.transform(x)
        return np.column_stack([np.full(len(f), m) if isinstance(m, float)
                                else m.predict_proba(f)[:, 1] for m in self.readout])
