"""Control the readout variance of the active Portia spiking mechanism.

The circuit and its local learning are inherited unchanged. Only the number and
regularization of appended circuit features change: fewer directions, no
amplification of weak principal components, and a separately scaled feature
block. This is a trainable readout over raw and spiking features, not a frozen LR
anchor or a performance-based fallback. Outer validation chooses hyperparameters.
"""

from __future__ import annotations

import numpy as np

from .portia_improved import ImprovedPortia


class GeneralizingPortia(ImprovedPortia):
    """Sparse LIF/EI/STDP model with a small, strongly regularized feature block.

    ``spectral_floor_ratio=1`` uses one common scale for all retained components,
    preserving their relative variance rather than whitening weak directions.
    ``mechanism_scale`` changes only the new features' effective L2 penalty; the
    original basis and its regularization geometry remain unchanged. No labels
    beyond fit are accessed. ``feature_rank`` and the positive scale are explicit
    controls, and diagnostics expose the actual circuit contribution.
    """

    def __init__(self, plasticity=.003, C=.1, epochs=4, seed=42,
                 neurons=32, steps=24, learning=True, mechanism_scale=.15,
                 feature_rank=8, supervised_strength=1.,
                 spectral_floor_ratio=1., inference_batch=256):
        if not np.isfinite(mechanism_scale) or mechanism_scale <= 0:
            raise ValueError("mechanism_scale must be positive; circuit suppression is not a candidate")
        if not np.isfinite(spectral_floor_ratio) or not 0 < spectral_floor_ratio <= 1:
            raise ValueError("spectral_floor_ratio must lie in (0, 1]")
        if not isinstance(inference_batch, int) or inference_batch < 1:
            raise ValueError("inference_batch must be a positive integer")
        super().__init__(plasticity=plasticity, C=C, epochs=epochs, seed=seed,
                         neurons=neurons, steps=steps, learning=learning,
                         mechanism_scale=mechanism_scale, feature_rank=feature_rank,
                         supervised_strength=supervised_strength)
        self.spectral_floor_ratio = float(spectral_floor_ratio)
        self.inference_batch = inference_batch

    def _fit_feature_map(self, x, raw, w):
        super()._fit_feature_map(x, raw, w)
        if len(self.mechanism_pca_scale):
            self.mechanism_pca_scale = np.maximum(
                self.mechanism_pca_scale,
                np.max(self.mechanism_pca_scale) * self.spectral_floor_ratio)
        return self._feature_map(x, raw)

    def fit(self, x, y, w):
        super().fit(x, y, w)
        features = self.transform(x)
        contributions = self._mechanism_logits(features)
        wn = np.asarray(w, dtype=float)
        wn = wn / wn.sum()
        mean = np.average(contributions, axis=0, weights=wn)
        sd = np.sqrt(np.average((contributions - mean) ** 2, axis=0, weights=wn))
        self.diagnostics.update({
            "readout_control": "low-rank circuit features with separate L2 shrinkage and common spectral scaling",
            "requested_feature_rank": self.feature_rank,
            "mechanism_scale": self.mechanism_scale,
            "spectral_floor_ratio": self.spectral_floor_ratio,
            "readout_C": self.C,
            "inference_batch": self.inference_batch,
            "fit_mechanism_logit_sd_per_label": sd.tolist(),
            "mechanism_readout_parameter_count": int(sum(
                0 if isinstance(m, float) else m.coef_[:, self.n_features_in_:].size
                for m in self.readout)),
            "uses_internal_validation_or_outer_test": False,
        })
        return self

    def transform(self, x):
        x = self._inference_input(x)
        chunks = []
        for start in range(0, len(x), self.inference_batch):
            part = x[start:start + self.inference_batch]
            raw, _, _ = self._simulate(part)
            chunks.append(self._feature_map(part, raw))
        return np.concatenate(chunks, axis=0)

    def _mechanism_logits(self, features):
        return np.column_stack([
            np.zeros(len(features)) if isinstance(m, float)
            else features[:, self.n_features_in_:] @ m.coef_[0, self.n_features_in_:]
            for m in self.readout])

    def mechanism_logits(self, x):
        """Expose the retained circuit contribution without altering predictions."""
        return self._mechanism_logits(self.transform(x))
