"""Meta models fitted only to caller-supplied out-of-fold expert predictions.

Expert construction and data splitting belong to the caller. No model here
contains a fallback expert, accesses raw data, or selects on evaluation labels.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


EPSILON = 1e-7


def _probabilities(values, expected_experts=None):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[1] == 0 or p.shape[2] != 4:
        raise ValueError("Probabilities must have shape (n > 0, experts > 0, 4)")
    if expected_experts is not None and p.shape[1] != expected_experts:
        raise ValueError("The expert count differs from the fitted ensemble")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return p


def _fit_arrays(probabilities, y, sample_weight):
    p = _probabilities(probabilities)
    target = np.asarray(y, dtype=np.float64)
    w = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if target.shape != (len(p), 4) or not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("Targets must be finite binary values with shape (n, 4)")
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Sample weights must have shape (n,), be finite and nonnegative, with positive total")
    # Scaling all survey weights by a constant must not change regularization.
    normalized = w / np.max(w)
    normalized = normalized / np.mean(normalized[normalized > 0])
    return p, target, normalized


class _BaseEnsemble:
    def __init__(self, expert_names=None):
        self.expert_names = None if expert_names is None else tuple(expert_names)

    def _set_fit_shape(self, p):
        count = p.shape[1]
        if self.expert_names is not None:
            if len(self.expert_names) != count or len(set(self.expert_names)) != count:
                raise ValueError("Expert names must be unique and match the expert axis")
            if any(not isinstance(name, str) or not name for name in self.expert_names):
                raise ValueError("Expert names must be nonempty strings")
        self.n_experts_ = count

    def _predict_array(self, probabilities):
        if not hasattr(self, "n_experts_"):
            raise RuntimeError("The ensemble must be fitted before prediction")
        return _probabilities(probabilities, self.n_experts_)

    def to_state(self):
        if not hasattr(self, "n_experts_"):
            raise RuntimeError("The ensemble must be fitted before serialization")
        return {
            "class": type(self).__name__,
            "n_experts": self.n_experts_,
            "expert_names": list(self.expert_names) if self.expert_names is not None else None,
            "probability_epsilon": EPSILON,
            "fit_source": "caller-supplied meta-training predictions only",
        }


class UniformProbabilityEnsemble(_BaseEnsemble):
    def fit(self, probabilities, y, sample_weight=None):
        p, _, _ = _fit_arrays(probabilities, y, sample_weight)
        self._set_fit_shape(p)
        return self

    def predict(self, probabilities):
        return self._predict_array(probabilities).mean(axis=1)


class UniformLogitEnsemble(UniformProbabilityEnsemble):
    def predict(self, probabilities):
        p = self._predict_array(probabilities)
        return expit(logit(np.clip(p, EPSILON, 1 - EPSILON)).mean(axis=1))


class SingleExpertEnsemble(UniformProbabilityEnsemble):
    """An explicitly selected raw expert control, without learned calibration."""

    def __init__(self, expert_index=0, expert_names=None):
        super().__init__(expert_names)
        if isinstance(expert_index, bool) or not isinstance(expert_index, (int, np.integer)) or expert_index < 0:
            raise ValueError("expert_index must be a nonnegative integer")
        self.expert_index = int(expert_index)

    def fit(self, probabilities, y, sample_weight=None):
        p, _, _ = _fit_arrays(probabilities, y, sample_weight)
        if self.expert_index >= p.shape[1]:
            raise ValueError("expert_index is outside the expert axis")
        self._set_fit_shape(p)
        return self

    def predict(self, probabilities):
        return self._predict_array(probabilities)[:, self.expert_index].copy()

    def to_state(self):
        return {**super().to_state(), "expert_index": self.expert_index}


class SimplexProbabilityEnsemble(_BaseEnsemble):
    """Per-label convex mixing, with L2 shrinkage toward uniform expert weights.

    Objective for each label: mean weighted BCE + regularization * ||a-u||².
    The coefficients a are nonnegative and sum to one. There is no intercept,
    negative expert weight, extra input feature, or implicit LR bypass.
    """

    def __init__(self, regularization=0.1, expert_names=None, max_iter=1000):
        super().__init__(expert_names)
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.regularization = float(regularization)
        self.max_iter = max_iter

    def fit(self, probabilities, y, sample_weight=None):
        p, target, w = _fit_arrays(probabilities, y, sample_weight)
        wn = w / w.sum()
        expert_count = p.shape[1]
        uniform = np.full(expert_count, 1 / expert_count)
        mixing = np.empty((4, expert_count))
        optimization = []
        for label in range(4):
            # Clipping the expert probabilities makes the objective and its
            # analytic gradient smooth even if an expert emits exactly 0 or 1.
            q = np.clip(p[:, :, label], EPSILON, 1 - EPSILON)
            outcome = target[:, label]

            def objective(a):
                estimate = q @ a
                value = np.sum(wn * (-outcome * np.log(estimate) - (1 - outcome) * np.log1p(-estimate)))
                value += self.regularization * np.sum((a - uniform) ** 2)
                derivative = wn * (estimate - outcome) / (estimate * (1 - estimate))
                gradient = q.T @ derivative + 2 * self.regularization * (a - uniform)
                return float(value), gradient

            if expert_count == 1:
                mixing[label] = uniform
                optimization.append({"success": True, "iterations": 0, "objective": objective(uniform)[0]})
                continue
            result = minimize(
                objective, uniform.copy(), jac=True, method="SLSQP",
                bounds=[(0.0, 1.0)] * expert_count,
                constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.0, "jac": lambda a: np.ones_like(a)}],
                options={"ftol": 1e-10, "maxiter": self.max_iter},
            )
            if not result.success or not np.isfinite(result.x).all() or abs(result.x.sum() - 1) > 1e-6:
                raise RuntimeError(f"Simplex fit did not converge for label {label}: {result.message}")
            a = np.maximum(result.x, 0)
            mixing[label] = a / a.sum()
            optimization.append({"success": True, "iterations": int(result.nit), "objective": objective(mixing[label])[0]})
        self._set_fit_shape(p)
        self.weights_ = mixing
        self.optimization_ = optimization
        return self

    def predict(self, probabilities):
        p = np.clip(self._predict_array(probabilities), EPSILON, 1 - EPSILON)
        return np.clip(np.einsum("nek,ke->nk", p, self.weights_), EPSILON, 1 - EPSILON)

    def to_state(self):
        return {
            **super().to_state(), "regularization": self.regularization,
            "weights_by_label": self.weights_.tolist(), "optimization": self.optimization_,
        }


class LogisticStackingEnsemble(_BaseEnsemble):
    """Ridge logistic stacking across all four logits from each supplied expert.

    For an LR-only calibration/control, pass an expert axis of length one;
    the exact same scaler, four-logit features and fitting procedure apply.
    """

    def __init__(self, C=0.1, expert_names=None, max_iter=3000):
        super().__init__(expert_names)
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be finite and positive")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.C = float(C)
        self.max_iter = max_iter

    @staticmethod
    def _features(p):
        return logit(np.clip(p, EPSILON, 1 - EPSILON)).reshape(len(p), -1)

    def fit(self, probabilities, y, sample_weight=None):
        p, target, w = _fit_arrays(probabilities, y, sample_weight)
        if any(len(np.unique(target[w > 0, label])) != 2 for label in range(4)):
            raise ValueError("Every target needs both classes among positive-weight meta-training rows")
        x = self._features(p)
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        coefficients, intercepts, iterations = [], [], []
        for label in range(4):
            model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs", tol=1e-8, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    model.fit(standardized, target[:, label], sample_weight=w)
                except ConvergenceWarning as exc:
                    raise RuntimeError(f"Logistic stack did not converge for label {label}") from exc
            coefficients.append(model.coef_[0])
            intercepts.append(model.intercept_[0])
            iterations.append(int(model.n_iter_[0]))
        self._set_fit_shape(p)
        self.mean_ = mean
        self.scale_ = scale
        self.coef_ = np.asarray(coefficients)
        self.intercept_ = np.asarray(intercepts)
        self.iterations_ = iterations
        return self

    def predict(self, probabilities):
        p = self._predict_array(probabilities)
        x = (self._features(p) - self.mean_) / self.scale_
        return np.clip(expit(x @ self.coef_.T + self.intercept_), EPSILON, 1 - EPSILON)

    def to_state(self):
        return {
            **super().to_state(), "C": self.C, "feature_order": "expert-major, four labels per expert",
            "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
            "coefficients": self.coef_.tolist(), "intercepts": self.intercept_.tolist(),
            "iterations": self.iterations_, "weight_normalization": "mean one among positive-weight meta-training rows",
        }
