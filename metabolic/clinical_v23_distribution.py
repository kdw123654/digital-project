"""One conditional distribution for five observed clinical measurements.

The model consumes only noninvasive features. Targets are transformed on fit
rows, and this module derives all four threshold flags and their 16 joint
states from a single rank-one Gaussian distribution in log coordinates.
The Gaussian is a model for observed measurements conditional on the inputs;
its residual variance is not an identified assay-error or physiology state.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


RAW_TARGETS = ("HE_glu", "HE_sbp", "HE_dbp", "HE_TG", "HE_HDL_st2")
COORDINATES = ("log_glucose", "log_dbp", "log_pulse_pressure", "log_tg", "log_hdl")
COMPONENTS = ("elevated_glucose", "elevated_bp", "elevated_tg", "low_hdl")
_BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
_LOG_2PI = math.log(2 * math.pi)
_LOG_GLU = math.log(100.)
_LOG_DBP = math.log(85.)
_LOG_TG = math.log(150.)


def _raw_coordinates(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not len(raw):
        raise ValueError("Expected nonempty raw targets [n,5] in glucose/SBP/DBP/TG/HDL order")
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Five raw measurements must be finite and positive")
    pulse_pressure = raw[:, 1] - raw[:, 2]
    if np.any(pulse_pressure <= 0):
        raise ValueError("Every observed SBP must exceed DBP for log pulse pressure")
    return np.log(np.column_stack((raw[:, 0], raw[:, 2], pulse_pressure,
                                   raw[:, 3], raw[:, 4])))


def _weights(values: np.ndarray, n: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Expected one finite positive survey weight per fit row")
    return weights


def _parameters_numpy(mu, diag_scale, loading) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for item in (mu, diag_scale, loading):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        arrays.append(np.asarray(item, dtype=np.float64))
    center, diag, rank1 = arrays
    if (center.ndim != 2 or center.shape[1] != 5 or not len(center) or
            diag.shape != center.shape or rank1.shape != center.shape or
            not all(np.isfinite(array).all() for array in arrays) or
            np.any(diag <= 0)):
        raise ValueError("Expected finite [n,5] mu, positive diag_scale, and loading")
    return center, diag, rank1


class JointTargetTransform:
    """Weighted fit-only standardization of log G, DBP, PP, TG and HDL."""

    SCHEMA_VERSION = "clinical-v23-joint-target/1"

    def fit(self, values, weights) -> "JointTargetTransform":
        coordinates = _raw_coordinates(values)
        w = _weights(weights, len(coordinates))
        self.mean = np.average(coordinates, axis=0, weights=w)
        self.scale = np.sqrt(np.average((coordinates - self.mean) ** 2, axis=0, weights=w))
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Each log target needs positive fit-only weighted variation")
        self.fit_n = len(coordinates)
        return self

    def _check_fit(self) -> None:
        if not hasattr(self, "mean") or not hasattr(self, "scale"):
            raise RuntimeError("Fit the target transform on clean fit rows first")

    def transform(self, values) -> np.ndarray:
        self._check_fit()
        return (_raw_coordinates(values) - self.mean) / self.scale

    @staticmethod
    def cutoffs(sex) -> np.ndarray:
        value = np.asarray(sex)
        if value.ndim != 1 or not len(value) or not np.isin(value, (1, 2)).all():
            raise ValueError("Known sex code 1 or 2 required for HDL threshold")
        return np.column_stack((np.full(len(value), 100.), np.full(len(value), 130.),
                                np.full(len(value), 85.), np.full(len(value), 150.),
                                np.where(value == 1, 40., 50.)))

    def inverse_means(self, mu, diag_scale, loading) -> np.ndarray:
        """Exact physical means for the five-dimensional lognormal construction."""
        self._check_fit()
        center, diag, rank1 = _parameters_numpy(mu, diag_scale, loading)
        log_mean = self.mean[None, :] + self.scale[None, :] * center
        log_variance = self.scale[None, :] ** 2 * (diag ** 2 + rank1 ** 2)
        with np.errstate(over="raise", invalid="raise"):
            try:
                physical = np.exp(log_mean + .5 * log_variance)
            except FloatingPointError as exc:
                raise FloatingPointError("Nonfinite lognormal physical mean") from exc
        result = np.column_stack((physical[:, 0], physical[:, 1] + physical[:, 2],
                                  physical[:, 1], physical[:, 3], physical[:, 4]))
        if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result[:, 1] <= result[:, 2]):
            raise FloatingPointError("Invalid physical mean after inverse transform")
        return result

    def to_dict(self) -> dict:
        self._check_fit()
        return {"schema": self.SCHEMA_VERSION, "raw_targets": list(RAW_TARGETS),
                "coordinates": list(COORDINATES), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "fit_n": int(self.fit_n)}

    @classmethod
    def from_dict(cls, record: dict) -> "JointTargetTransform":
        if (not isinstance(record, dict) or record.get("schema") != cls.SCHEMA_VERSION or
                record.get("raw_targets") != list(RAW_TARGETS) or
                record.get("coordinates") != list(COORDINATES)):
            raise ValueError("Unknown joint target transform schema")
        result = cls()
        result.mean = np.asarray(record.get("mean"), dtype=np.float64)
        result.scale = np.asarray(record.get("scale"), dtype=np.float64)
        result.fit_n = record.get("fit_n")
        if (result.mean.shape != (5,) or result.scale.shape != (5,) or
                not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all() or
                np.any(result.scale <= 0) or not isinstance(result.fit_n, int) or result.fit_n <= 0):
            raise ValueError("Invalid saved joint target fit statistics")
        return result


def weighted_residual_covariance(target_z, predicted_mu, weights) -> np.ndarray:
    """Fit-only weighted covariance for the model's initial residual factor."""
    truth = np.asarray(target_z, dtype=np.float64)
    predicted = np.asarray(predicted_mu, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != 5 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("Expected aligned [n,5] standardized targets and predictions")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Residual inputs must be finite")
    w = _weights(weights, len(truth))
    residual = truth - predicted
    centered = residual - np.average(residual, axis=0, weights=w)
    covariance = (centered * w[:, None]).T @ centered / w.sum()
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all() or np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
        raise FloatingPointError("Invalid fit residual covariance")
    return covariance


def gaussian_nll(mu: torch.Tensor, diag_scale: torch.Tensor, loading: torch.Tensor,
                 target_z: torch.Tensor, weights: torch.Tensor | None = None,
                 reduction: Literal["mean", "none"] = "mean") -> torch.Tensor:
    """Gaussian NLL in standardized log coordinates, including the normalizer.

    Sigma = diag(diag_scale**2) + loading @ loading.T per person. Internal
    float64 arithmetic limits cancellation in the Sherman-Morrison quadratic.
    A raw-unit likelihood would additionally need the fixed log Jacobian.
    """
    if reduction not in ("mean", "none"):
        raise ValueError("Reduction must be mean or none")
    if (not all(isinstance(value, torch.Tensor) for value in
                (mu, diag_scale, loading, target_z)) or
            mu.ndim != 2 or mu.shape[1] != 5 or mu.shape[0] == 0 or
            any(value.shape != mu.shape for value in (diag_scale, loading, target_z))):
        raise ValueError("Expected aligned Torch [batch,5] distribution and target tensors")
    if (not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(diag_scale).all()) or
            not bool(torch.isfinite(loading).all()) or not bool(torch.isfinite(target_z).all()) or
            not bool((diag_scale > 0).all())):
        raise ValueError("Distribution parameters must be finite with positive diagonal scales")
    center = mu.to(torch.float64)
    diagonal = diag_scale.to(torch.float64)
    rank1 = loading.to(torch.float64)
    truth = target_z.to(torch.float64)
    residual = (truth - center) / diagonal
    factor = rank1 / diagonal
    factor_norm = factor.square().sum(1)
    projection = (residual * factor).sum(1)
    factor_mode = projection / (1 + factor_norm)
    quadratic = (residual - factor_mode[:, None] * factor).square().sum(1) + factor_mode.square()
    logdet = 2 * diagonal.log().sum(1) + torch.log1p(factor_norm)
    per_row = .5 * (5 * _LOG_2PI + logdet + quadratic)
    if not bool(torch.isfinite(per_row).all()):
        raise FloatingPointError("Nonfinite Gaussian NLL")
    if reduction == "none":
        return per_row
    if weights is None:
        return per_row.mean()
    w = torch.as_tensor(weights, device=per_row.device, dtype=torch.float64)
    if w.shape != (len(per_row),) or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
        raise ValueError("Expected one finite positive weight per NLL row")
    return (w * per_row).sum() / w.sum()


@lru_cache(maxsize=8)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(order, bool) or not isinstance(order, int) or order < 8 or order > 256:
        raise ValueError("Quadrature order must be an integer from 8 to 256")
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(order)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(order)
    return (gh_nodes * math.sqrt(2), gh_weights / math.sqrt(math.pi),
            (gl_nodes + 1) / 2, gl_weights / 2)


def _safe_bp_probability(mean_dbp: np.ndarray, scale_dbp: np.ndarray,
                         mean_pp: np.ndarray, scale_pp: np.ndarray,
                         inner_nodes: np.ndarray, inner_weights: np.ndarray) -> np.ndarray:
    """P(DBP<85 and DBP+PP<130), conditional on the common factor."""
    below_85 = ndtr((_LOG_DBP - mean_dbp) / scale_dbp)
    safe = np.zeros_like(below_85)
    active = below_85 > 0
    if not active.any():
        return safe
    uniform = below_85[active, None] * inner_nodes[None, :]
    dbp = np.exp(mean_dbp[active, None] + scale_dbp[active, None] * ndtri(uniform))
    remaining = 130. - dbp
    if not np.isfinite(dbp).all() or np.any(remaining <= 0):
        raise FloatingPointError("Invalid DBP quadrature node for BP OR")
    pp_cdf = ndtr((np.log(remaining) - mean_pp[active, None]) / scale_pp[active, None])
    safe[active] = below_85[active] * (pp_cdf @ inner_weights)
    return safe


def _conditional_flags(log_means: np.ndarray, log_scales: np.ndarray,
                       hdl_cutoffs: np.ndarray, inner_nodes: np.ndarray,
                       inner_weights: np.ndarray) -> np.ndarray:
    safe_bp = _safe_bp_probability(log_means[:, 1], log_scales[:, 1],
                                    log_means[:, 2], log_scales[:, 2],
                                    inner_nodes, inner_weights)
    probability = np.column_stack((
        ndtr((log_means[:, 0] - _LOG_GLU) / log_scales[:, 0]),
        1 - safe_bp,
        ndtr((log_means[:, 3] - _LOG_TG) / log_scales[:, 3]),
        ndtr((np.log(hdl_cutoffs) - log_means[:, 4]) / log_scales[:, 4]),
    ))
    if (not np.isfinite(probability).all() or np.any(probability < -1e-12) or
            np.any(probability > 1 + 1e-12)):
        raise FloatingPointError("Conditional event probabilities left [0,1]")
    return probability


def _states_from_conditional(probability: np.ndarray) -> np.ndarray:
    return np.prod(np.where(_BITS[None, :, :], probability[:, None, :],
                            1 - probability[:, None, :]), axis=2)


def _integrate_gh(center: np.ndarray, diag: np.ndarray, rank1: np.ndarray,
                  hdl_cutoffs: np.ndarray, transform: JointTargetTransform,
                  outer_order: int, inner_order: int) -> np.ndarray:
    factors, factor_weights, _, _ = _quadrature(outer_order)
    _, _, inner_nodes, inner_weights = _quadrature(inner_order)
    log_scales = transform.scale[None, :] * diag
    q16 = np.zeros((len(center), 16), dtype=np.float64)
    for factor, factor_weight in zip(factors, factor_weights, strict=True):
        log_means = transform.mean[None, :] + transform.scale[None, :] * (
            center + factor * rank1)
        conditional = _conditional_flags(log_means, log_scales, hdl_cutoffs,
                                         inner_nodes, inner_weights)
        q16 += factor_weight * _states_from_conditional(conditional)
    return q16


def _analytic_three_marginals(center: np.ndarray, diag: np.ndarray,
                              rank1: np.ndarray, hdl_cutoffs: np.ndarray,
                              transform: JointTargetTransform) -> np.ndarray:
    means = transform.mean[None, :] + transform.scale[None, :] * center
    scales = transform.scale[None, :] * np.sqrt(diag**2 + rank1**2)
    return np.column_stack((
        ndtr((means[:, 0] - _LOG_GLU) / scales[:, 0]),
        ndtr((means[:, 3] - _LOG_TG) / scales[:, 3]),
        ndtr((np.log(hdl_cutoffs) - means[:, 4]) / scales[:, 4]),
    ))


def _bp_reference(center: np.ndarray, diag: np.ndarray, rank1: np.ndarray,
                  transform: JointTargetTransform) -> np.ndarray:
    """Independent 1D reference from the unconditional bivariate log D/PP law."""
    mean = transform.mean[None, :] + transform.scale[None, :] * center
    scale = transform.scale[None, :]
    sd_d = scale[:, 1] * np.sqrt(diag[:, 1]**2 + rank1[:, 1]**2)
    covariance = scale[:, 1] * scale[:, 2] * rank1[:, 1] * rank1[:, 2]
    sd_p_given_d = scale[:, 2] * np.sqrt(
        diag[:, 2]**2 + rank1[:, 2]**2 * diag[:, 1]**2 /
        (diag[:, 1]**2 + rank1[:, 1]**2))
    if not np.isfinite(sd_p_given_d).all() or np.any(sd_p_given_d <= 0):
        raise FloatingPointError("Invalid conditional BP reference covariance")
    below_85 = ndtr((_LOG_DBP - mean[:, 1]) / sd_d)
    result = np.ones(len(center), dtype=np.float64)
    for start in range(0, len(center), 256):
        stop = min(start + 256, len(center))
        active = np.flatnonzero(below_85[start:stop] > 0) + start
        if not len(active):
            continue
        a = below_85[active]
        m_d, m_p = mean[active, 1], mean[active, 2]
        s_d, s_pc = sd_d[active], sd_p_given_d[active]
        adjustment = covariance[active] / s_d

        def integrand(unit: float) -> np.ndarray:
            z = ndtri(np.maximum(a * unit, np.nextafter(0., 1.)))
            dbp = np.exp(m_d + s_d * z)
            remaining = 130. - dbp
            if np.any(remaining <= 0):
                raise FloatingPointError("Invalid unconditional BP reference node")
            conditional_mean = m_p + adjustment * z
            return a * ndtr((np.log(remaining) - conditional_mean) / s_pc)

        safe, error = quad_vec(integrand, 0., 1., epsabs=1e-9, epsrel=1e-9,
                               norm="max", limit=512)
        if not np.isfinite(safe).all() or not np.isfinite(error) or error > 1e-7:
            raise FloatingPointError("Unconditional BP reference integration failed")
        result[active] = 1 - safe
    return result


def _safe_bp_scalar_quad(mean_d: float, scale_d: float,
                         mean_p: float, scale_p: float) -> float:
    a = float(ndtr((_LOG_DBP - mean_d) / scale_d))
    if a == 0:
        return 0.

    def integrand(unit: float) -> float:
        dbp = math.exp(mean_d + scale_d * float(ndtri(a * unit)))
        return a * float(ndtr((math.log(130. - dbp) - mean_p) / scale_p))

    safe, error = quad(integrand, 0., 1., epsabs=1e-10, epsrel=1e-10, limit=200)
    if not math.isfinite(safe) or error > 1e-7:
        raise FloatingPointError("Conditional BP quadrature fallback failed")
    return safe


def _factor_breakpoints(center: np.ndarray, rank1: np.ndarray,
                        hdl_cutoff: float, transform: JointTargetTransform) -> list[float]:
    mean = transform.mean + transform.scale * center
    coefficient = transform.scale * rank1
    points: list[float] = []
    for index, cutoff in ((0, _LOG_GLU), (1, _LOG_DBP), (3, _LOG_TG),
                          (4, math.log(hdl_cutoff))):
        if abs(coefficient[index]) > 1e-12:
            point = (cutoff - mean[index]) / coefficient[index]
            if -10 < point < 10:
                points.append(float(point))

    def mean_sbp(factor: float) -> float:
        return (math.exp(mean[1] + coefficient[1] * factor) +
                math.exp(mean[2] + coefficient[2] * factor) - 130.)

    try:
        if mean_sbp(-10.) * mean_sbp(10.) < 0:
            points.append(float(brentq(mean_sbp, -10., 10.)))
    except OverflowError:
        pass
    return sorted(set(points))


def _integrate_quad_row(center: np.ndarray, diag: np.ndarray,
                        rank1: np.ndarray, hdl_cutoff: float,
                        transform: JointTargetTransform,
                        inner_order: int) -> tuple[np.ndarray, int]:
    _, _, inner64_nodes, inner64_weights = _quadrature(inner_order)
    _, _, inner128_nodes, inner128_weights = _quadrature(2 * inner_order)
    log_scale = transform.scale * diag
    inner_fallback_calls = 0

    def integrand(factor: float) -> np.ndarray:
        nonlocal inner_fallback_calls
        log_mean = transform.mean + transform.scale * (center + factor * rank1)
        safe64 = _safe_bp_probability(log_mean[1:2], log_scale[1:2],
                                       log_mean[2:3], log_scale[2:3],
                                       inner64_nodes, inner64_weights)[0]
        safe128 = _safe_bp_probability(log_mean[1:2], log_scale[1:2],
                                        log_mean[2:3], log_scale[2:3],
                                        inner128_nodes, inner128_weights)[0]
        if abs(safe64 - safe128) > 1e-7:
            safe = _safe_bp_scalar_quad(log_mean[1], log_scale[1],
                                        log_mean[2], log_scale[2])
            inner_fallback_calls += 1
        else:
            safe = safe128
        conditional = np.array([[
            ndtr((log_mean[0] - _LOG_GLU) / log_scale[0]),
            1 - safe,
            ndtr((log_mean[3] - _LOG_TG) / log_scale[3]),
            ndtr((math.log(hdl_cutoff) - log_mean[4]) / log_scale[4]),
        ]])
        return math.exp(-.5 * factor * factor) / math.sqrt(2 * math.pi) * (
            _states_from_conditional(conditional)[0])

    points = _factor_breakpoints(center, rank1, hdl_cutoff, transform)
    value, error = quad_vec(integrand, -10., 10., points=points,
                            epsabs=1e-9, epsrel=1e-9, norm="max", limit=512)
    if not np.isfinite(value).all() or not np.isfinite(error) or error > 1e-7:
        raise FloatingPointError("Adaptive outer factor integration failed")
    return value, inner_fallback_calls


def _normalize_and_derive(q16: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = q16.sum(axis=1)
    if (not np.isfinite(q16).all() or np.any(q16 < -1e-12) or
            not np.allclose(mass, 1, atol=1e-9, rtol=0)):
        raise FloatingPointError("Joint probability integration lost simplex mass")
    q16 = q16 / mass[:, None]
    marginals = q16 @ _BITS.astype(np.float64)
    any_probability = 1 - q16[:, 0]
    if (np.any(any_probability + 1e-12 < marginals.max(axis=1)) or
            np.any(any_probability - 1e-12 > np.minimum(1, marginals.sum(axis=1)))):
        raise FloatingPointError("Integrated any-event risk violates Fréchet bounds")
    return q16, marginals, any_probability


def joint_probabilities(mu, diag_scale, loading, sex, transform: JointTargetTransform,
                        order: int = 64, audit: bool = True) -> dict[str, np.ndarray | dict]:
    """One deterministic joint law; audit/adapt quadrature before returning risks.

    The default checks GH order doubling, GL order doubling, exact lognormal
    glucose/TG/HDL tails and an independent bivariate-lognormal BP integral.
    Rows with a failed check or loading/diagonal ratio >=3 use adaptive outer
    integration. ``audit=False`` exposes raw quadrature for convergence tests.
    """
    transform._check_fit()
    center, diag, rank1 = _parameters_numpy(mu, diag_scale, loading)
    cutoffs = transform.cutoffs(sex)
    if len(cutoffs) != len(center):
        raise ValueError("Sex length must equal distribution batch size")
    if not isinstance(audit, bool):
        raise ValueError("audit must be a bool")
    _quadrature(order)
    if audit and 2 * order > 256:
        raise ValueError("Audited quadrature order cannot exceed 128")
    q16 = _integrate_gh(center, diag, rank1, cutoffs[:, 4], transform, order, order)
    diagnostics: dict = {"audited": audit, "requested_order": order,
                         "effective_method_counts": {f"GH{order}/GL{order}": len(center)},
                         "fallback_rows": 0}
    if audit:
        outer = _integrate_gh(center, diag, rank1, cutoffs[:, 4], transform, 2 * order, order)
        inner = _integrate_gh(center, diag, rank1, cutoffs[:, 4], transform, order, 2 * order)
        outer_error = np.max(np.abs(q16 - outer), axis=1)
        inner_error = np.max(np.abs(q16 - inner), axis=1)
        analytic = _analytic_three_marginals(center, diag, rank1, cutoffs[:, 4], transform)
        bp_reference = _bp_reference(center, diag, rank1, transform)
        _, raw_marginals, _ = _normalize_and_derive(q16)
        analytic_error = np.max(np.abs(raw_marginals[:, [0, 2, 3]] - analytic), axis=1)
        bp_error = np.abs(raw_marginals[:, 1] - bp_reference)
        ratio = np.max(np.abs(rank1) / diag, axis=1)
        tolerance = 5e-5
        fallback = ((outer_error > tolerance) | (inner_error > tolerance) |
                    (analytic_error > tolerance) | (bp_error > tolerance) |
                    (ratio >= 3.))
        inner_fallback_calls = 0
        for row in np.flatnonzero(fallback):
            q16[row], calls = _integrate_quad_row(center[row], diag[row], rank1[row],
                                                   float(cutoffs[row, 4]), transform, order)
            inner_fallback_calls += calls
        q16, final_marginals, _ = _normalize_and_derive(q16)
        final_analytic_error = np.max(np.abs(final_marginals[:, [0, 2, 3]] - analytic))
        final_bp_error = np.max(np.abs(final_marginals[:, 1] - bp_reference))
        if final_analytic_error > 1e-4 or final_bp_error > 1e-4:
            raise FloatingPointError("Audited joint marginals disagree with independent reference")
        diagnostics.update({
            "outer_check_order": 2 * order, "inner_check_order": 2 * order,
            "max_outer_order_diff": float(outer_error.max()),
            "max_inner_order_diff": float(inner_error.max()),
            "max_analytic_marginal_error_before": float(analytic_error.max()),
            "max_bp_reference_error_before": float(bp_error.max()),
            "max_analytic_marginal_error_after": float(final_analytic_error),
            "max_bp_reference_error_after": float(final_bp_error),
            "max_loading_diag_ratio": float(ratio.max()),
            "fallback_due_ratio_rows": int((ratio >= 3.).sum()),
            "fallback_rows": int(fallback.sum()),
            "inner_scalar_quad_calls": inner_fallback_calls,
            "effective_method_counts": {f"GH{order}/GL{order}": int((~fallback).sum()),
                                        "adaptive_quad_vec": int(fallback.sum())},
        })
    q16, marginals, any_probability = _normalize_and_derive(q16)
    return {"state_probabilities": q16, "marginals4": marginals,
            "any_probability": any_probability,
            "physical_means": transform.inverse_means(center, diag, rank1),
            "integration_diagnostics": diagnostics}


def apply_calibration(mu, diag_scale, loading, delta_mu=None,
                      tau: float = 1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply five location offsets and one common positive spread to the joint law."""
    center, diag, rank1 = _parameters_numpy(mu, diag_scale, loading)
    shift = np.zeros(5) if delta_mu is None else np.asarray(delta_mu, dtype=np.float64)
    if shift.shape != (5,) or not np.isfinite(shift).all():
        raise ValueError("Calibration location shift must have five finite coordinates")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Joint calibration spread tau must be finite and positive")
    return center + shift[None, :], diag * tau, rank1 * tau
