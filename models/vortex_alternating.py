"""Block-coordinate readout/wave optimization, retaining Vortex mechanisms.

The direct 72 encoded inputs and learned wave/phase coordinates share a freshly
optimized logistic readout.  No logistic prediction is frozen, and no residual
logit cap is imposed.  Training-only projection and whitening prevent redundant
wave coordinates from changing the baseline regularization accidentally.  Their
coefficients are fixed within each encoder epoch but the tensor transform stays
in the gradient graph.  Integer winding is still diagnostic-only.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from .vortex_improved import ImprovedVortex


class VortexAlternating:
    """Alternating convex readout solves and one wave-encoder epoch per round.

    ``validation=(xval, yval, wval)`` accepts one matrix or a list ordered as
    clean, masked1, ... .  Selection score is half clean macro AP plus half
    mean masked macro AP when masked matrices are supplied.  No validation
    rows/labels are used to fit projection, whitening, readout or encoder.
    With no validation, the final round is retained; training AP is not used
    for early stopping.
    """
    def __init__(self, C=.1, new_feature_scale=.3, latent_dim=32, steps=4,
                 rounds=12, encoder_lr=3e-4, batch_size=256, seed=42,
                 new_feature_dim=32, projection_ridge=.01, encoder_decay=.001,
                 readout_max_iter=5000):
        if C <= 0 or new_feature_scale <= 0 or rounds < 0 or rounds > 15:
            raise ValueError("C/feature scale must be positive and rounds in [0, 15]")
        if batch_size < 1 or new_feature_dim < 1 or projection_ridge <= 0:
            raise ValueError("Positive batch size, added feature dimension and projection ridge required")
        self.C, self.new_feature_scale = float(C), float(new_feature_scale)
        self.latent_dim, self.steps, self.rounds = latent_dim, steps, rounds
        self.encoder_lr, self.batch_size, self.seed = encoder_lr, batch_size, seed
        self.new_feature_dim, self.projection_ridge = new_feature_dim, projection_ridge
        self.encoder_decay, self.readout_max_iter = encoder_decay, readout_max_iter

    @staticmethod
    def _arrays(x, y, w):
        x, y, w = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
        if x.ndim != 2 or y.shape != (len(x), 4) or w.shape != (len(x),):
            raise ValueError("Expected x[n, d], binary y[n, 4] and weights[n]")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
            raise ValueError("Inputs, labels and weights must be finite")
        if (w < 0).any() or w.sum() <= 0 or not np.isin(y, [0, 1]).all():
            raise ValueError("Weights must be nonnegative and labels binary")
        return x, y, w

    def _learned_tensor(self, x):
        views = self.model.feature_views(x)
        return torch.cat((views["wave"], views["proxy"]), -1)

    def _learned_numpy(self, x):
        self.model.eval()
        with torch.no_grad():
            rows = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
            return self._learned_tensor(rows).numpy().astype(float)

    def _fit_transform(self, x, w):
        q = self._learned_numpy(x)
        basis = np.column_stack((np.ones(len(x)), x.astype(float)))
        wn = w / w.sum()
        normal = basis.T @ (basis * wn[:, None])
        penalty = np.eye(basis.shape[1]) * self.projection_ridge
        penalty[0, 0] = 0
        self.projection = np.linalg.solve(normal + penalty, basis.T @ (q * wn[:, None]))
        residual = q - basis @ self.projection
        self.residual_mean = np.sum(residual * wn[:, None], axis=0)
        residual -= self.residual_mean
        covariance = residual.T @ (residual * wn[:, None])
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1][:min(self.new_feature_dim, len(values))]
        self.whitener = vectors[:, order] / np.sqrt(np.maximum(values[order], .05 ** 2))
        self.projection_fit_n = len(x)
        self._set_tensor_transform()
        return np.column_stack((x, residual @ self.whitener * self.new_feature_scale))

    def _set_tensor_transform(self):
        self._projection_t = torch.as_tensor(self.projection, dtype=torch.float32)
        self._residual_mean_t = torch.as_tensor(self.residual_mean, dtype=torch.float32)
        self._whitener_t = torch.as_tensor(self.whitener, dtype=torch.float32)

    def transform_tensor(self, x):
        """Fixed train-only statistics, but a fully differentiable wave path."""
        q = self._learned_tensor(x)
        basis = torch.cat((torch.ones_like(x[:, :1]), x), -1)
        residual = q - basis @ self._projection_t - self._residual_mean_t
        added = residual @ self._whitener_t * self.new_feature_scale
        return torch.cat((x, added), -1)

    def _transform_numpy(self, x):
        q = self._learned_numpy(x)
        basis = np.column_stack((np.ones(len(x)), x))
        residual = q - basis @ self.projection - self.residual_mean
        return np.column_stack((x, residual @ self.whitener * self.new_feature_scale))

    def _solve_readout(self, features, y, w):
        models, iterations, residuals, warn_count = [], [], [], 0
        for j in range(4):
            if len(np.unique(y[w > 0, j])) != 2:
                raise ValueError("Each training label needs positive-weight examples of both classes")
            readout = LogisticRegression(C=self.C, max_iter=self.readout_max_iter,
                                         tol=1e-8, solver="lbfgs")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                readout.fit(features, y[:, j], sample_weight=w)
            warn_count += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
            models.append(readout)
            iterations.append(int(readout.n_iter_[0]))
            error = (readout.predict_proba(features)[:, 1] - y[:, j]) * w
            grad = (features.T @ error + readout.coef_[0] / self.C) / w.sum()
            residuals.append(float(max(np.abs(grad).max(), abs(error.sum()) / w.sum())))
        self.coef = np.stack([m.coef_[0] for m in models])
        self.intercept = np.array([m.intercept_[0] for m in models])
        if not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all():
            raise FloatingPointError("Readout solve produced a nonfinite coefficient")
        return {"iterations": iterations, "convergence_warnings": int(warn_count),
                "max_abs_mean_objective_gradient": max(residuals)}

    @staticmethod
    def _macro_ap(y, p, w):
        return float(np.mean([average_precision_score(y[:, j], p[:, j], sample_weight=w)
                              for j in range(4)]))

    def _validation_score(self, validation):
        xs, y, w = validation
        scores = [self._macro_ap(y, self.predict(x), w) for x in xs]
        score = scores[0] if len(scores) == 1 else .5 * scores[0] + .5 * float(np.mean(scores[1:]))
        return float(score), scores

    def _snapshot(self):
        return {"model": copy.deepcopy(self.model.state_dict()),
                **{name: getattr(self, name).copy() for name in
                   ("projection", "residual_mean", "whitener", "coef", "intercept")}}

    def _restore(self, snapshot):
        self.model.load_state_dict(snapshot["model"])
        self.model.eval()
        for name, value in snapshot.items():
            if name != "model":
                setattr(self, name, value.copy())
        self._set_tensor_transform()

    def fit(self, x, y, w, validation=None):
        x, y, w = self._arrays(x, y, w)
        # Preserve caller weight magnitude: the same C then has the same
        # meaning as the caller's LogisticRegression benchmark.
        if validation is not None:
            val_x, val_y, val_w = validation
            val_x = val_x if isinstance(val_x, (list, tuple)) else [val_x]
            checked = [self._arrays(z, val_y, val_w) for z in val_x]
            if not checked or any(z.shape[1] != x.shape[1] for z, _, _ in checked):
                raise ValueError("Validation must contain compatible encoded matrices")
            validation = ([z for z, _, _ in checked], checked[0][1], checked[0][2])
        torch.manual_seed(self.seed)
        self.model = ImprovedVortex(x.shape[1], self.latent_dim, self.steps, dropout=0.)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not name.startswith(("raw_head.", "wave_head.", "topology_head.")))
        self.history = []
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=self.encoder_lr, weight_decay=self.encoder_decay)
        rng = np.random.default_rng(self.seed)
        best_score, best, best_round = -np.inf, None, None
        initial_phase = self.model.phase.weight.detach().clone()
        initial_cayley = self.model.unitary_evolution.W.detach().clone()
        for round_index in range(self.rounds + 1):
            features = self._fit_transform(x, w)
            convergence = self._solve_readout(features, y, w)
            row = {"round": round_index, "readout": convergence}
            if validation is not None:
                score, scores = self._validation_score(validation)
                row.update(validation_score=score, validation_by_condition=scores)
                if score > best_score:
                    best_score, best, best_round = score, self._snapshot(), round_index
            else:
                best, best_round = self._snapshot(), round_index
            self.history.append(row)
            if round_index == self.rounds:
                break
            coefficients = torch.as_tensor(self.coef, dtype=torch.float32)
            intercept = torch.as_tensor(self.intercept, dtype=torch.float32)
            self.model.train()
            gradient_stats = {"phase": 0., "cayley": 0., "gp": 0.}
            total_loss, total_weight = 0., 0.
            order = rng.permutation(len(x))
            for begin in range(0, len(x), self.batch_size):
                ix = order[begin:begin + self.batch_size]
                if w[ix].sum() == 0:
                    continue
                xx = torch.as_tensor(x[ix], dtype=torch.float32)
                yy = torch.as_tensor(y[ix], dtype=torch.float32)
                ww = torch.as_tensor(w[ix], dtype=torch.float32)
                logits = self.transform_tensor(xx) @ coefficients.T + intercept
                loss = (F.binary_cross_entropy_with_logits(logits, yy, reduction="none").mean(1) * ww).sum() / ww.sum()
                optimizer.zero_grad()
                loss.backward()
                for name, parameter in (("phase", self.model.phase.weight),
                                        ("cayley", self.model.unitary_evolution.W),
                                        ("gp", self.model.non_linear_gp.alpha)):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Missing/nonfinite {name} wave gradient")
                    gradient_stats[name] = max(gradient_stats[name], float(parameter.grad.norm()))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * w[ix].sum()
                total_weight += w[ix].sum()
            row.update(encoder_loss=total_loss / total_weight, encoder_gradient_norm_max=gradient_stats)
        self._restore(best)
        self.best_round = best_round
        self.diagnostics_ = {
            "best_round": best_round, "completed_encoder_epochs": self.rounds,
            "projection_fit_rows": self.projection_fit_n, "readout_features": self.coef.shape[1],
            "phase_selected_change_norm": float((self.model.phase.weight.detach() - initial_phase).norm()),
            "cayley_selected_change_norm": float((self.model.unitary_evolution.W.detach() - initial_cayley).norm()),
            "maximum_readout_mean_gradient": max(r["readout"]["max_abs_mean_objective_gradient"] for r in self.history),
            "readout_convergence_warnings": sum(r["readout"]["convergence_warnings"] for r in self.history),
            "selection": "validation_half_clean_half_mean_masked" if validation is not None else "final_round_no_validation",
        }
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.model.input_dim or not np.isfinite(x).all():
            raise ValueError("Expected finite encoded input matrix with fitted feature width")
        features = self._transform_numpy(x)
        return expit(features @ self.coef.T + self.intercept)
