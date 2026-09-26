"""Differentiable v24 flag risks and coherent mixtures of v23 joint laws.

The training risk starts with GL32 integration for the DBP/PP event and checks
successive orders, escalating individual rows when needed. Fixed-order mode
is exposed for numerical audits. Final joint probabilities remain the audited
v23 ``joint_probabilities`` result.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from .clinical_v23_distribution import JointTargetTransform


_LOG_GLU = math.log(100.)
_LOG_DBP = math.log(85.)
_LOG_TG = math.log(150.)
_BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float64)


@lru_cache(maxsize=8)
def _unit_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(order, bool) or not isinstance(order, int) or not 8 <= order <= 1024:
        raise ValueError("Quadrature order must be an integer from 8 to 1024")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1.) / 2., weights / 2.


def _torch_parameters(mu: torch.Tensor, diag_scale: torch.Tensor,
                      loading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (not all(isinstance(v, torch.Tensor) for v in (mu, diag_scale, loading)) or
            mu.ndim != 2 or mu.shape[1] != 5 or mu.shape[0] == 0 or
            diag_scale.shape != mu.shape or loading.shape != mu.shape or
            not mu.is_floating_point() or not diag_scale.is_floating_point() or
            not loading.is_floating_point() or
            mu.device != diag_scale.device or mu.device != loading.device):
        raise ValueError("Expected aligned floating Torch [batch,5] tensors on one device")
    if (not bool(torch.isfinite(mu).all()) or
            not bool(torch.isfinite(diag_scale).all()) or
            not bool(torch.isfinite(loading).all()) or
            not bool((diag_scale > 0).all())):
        raise ValueError("Distribution parameters must be finite with positive diagonal scales")
    return mu.to(torch.float64), diag_scale.to(torch.float64), loading.to(torch.float64)


def _bp_gl(means: torch.Tensor, diag: torch.Tensor, rank1: torch.Tensor,
           fit_scale: torch.Tensor, order: int) -> torch.Tensor:
    """P(DBP>=85 or DBP+PP>=130), using conditional PP given log DBP."""
    nodes_np, weights_np = _unit_legendre(order)
    d_var = diag[:, 1].square() + rank1[:, 1].square()
    sd_d = fit_scale[1] * torch.sqrt(d_var)
    covariance = fit_scale[1] * fit_scale[2] * rank1[:, 1] * rank1[:, 2]
    sd_p_given_d = fit_scale[2] * torch.sqrt(
        diag[:, 2].square() + rank1[:, 2].square() * diag[:, 1].square() / d_var)
    a = torch.special.ndtr((_LOG_DBP - means[:, 1]) / sd_d)
    # If P(DBP<85) is below 1e-15, BP risk is one at this precision.
    # Skipping the inactive rows avoids ndtri(0) and keeps gradients finite.
    active = torch.nonzero((a > 1e-15).detach(), as_tuple=False).flatten()
    result = torch.ones_like(a)
    if active.numel() == 0:
        return result
    a_active = a[active]
    nodes = torch.as_tensor(nodes_np, dtype=torch.float64, device=means.device)
    weights = torch.as_tensor(weights_np, dtype=torch.float64, device=means.device)
    u = torch.clamp(a_active[:, None] * nodes[None, :], min=1e-15,
                    max=1. - 1e-15)
    z = torch.special.ndtri(u)
    dbp = torch.exp(means[active, 1, None] + sd_d[active, None] * z)
    remaining = 130. - dbp
    if not bool((remaining > 0).all()):
        raise FloatingPointError("Invalid DBP quadrature node for BP OR")
    conditional_pp_mean = means[active, 2, None] + (covariance[active] / sd_d[active])[:, None] * z
    pp_below = torch.special.ndtr((remaining.log() - conditional_pp_mean) /
                                   sd_p_given_d[active, None])
    safe_bp = a_active * (pp_below * weights[None, :]).sum(dim=1)
    return result.index_copy(0, active, 1. - safe_bp)


def torch_flag_probabilities(mu: torch.Tensor, diag_scale: torch.Tensor,
                             loading: torch.Tensor, sex, transform: JointTargetTransform,
                             order: int = 32, *, adaptive: bool = True,
                             tolerance: float = 1e-5) -> torch.Tensor:
    """Return differentiable [batch,4] G/BP/TG/HDL risks in float64.

    ``mu``, ``diag_scale`` and ``loading`` are standardized log-coordinate
    [batch,5] tensors, with ``Sigma = diag(diag_scale**2) + ll.T``.  Gradients
    flow back to float32 or float64 inputs.  The three non-BP risks use exact
    lognormal marginals.  BP uses the unconditional bivariate log DBP/PP law
    and Gauss-Legendre quadrature over its truncated DBP CDF. By default each
    row starts at ``order`` (32), compares to successive doubled orders, and
    retains the lower order only when their difference is at most ``tolerance``.
    It raises if GL1024 does not converge. ``adaptive=False`` exposes a fixed
    order for diagnostic comparisons and is not the training default.
    """
    _unit_legendre(order)
    if not isinstance(adaptive, bool):
        raise ValueError("adaptive must be a bool")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if adaptive and order not in (8, 16, 32, 64, 128, 256, 512):
        raise ValueError("Adaptive quadrature order must be a power of two from 8 to 512")
    transform._check_fit()
    center, diag, rank1 = _torch_parameters(mu, diag_scale, loading)
    sex_t = torch.as_tensor(sex, device=center.device)
    if (sex_t.shape != (len(center),) or
            not bool(((sex_t == 1) | (sex_t == 2)).all())):
        raise ValueError("Expected one known sex code 1 or 2 per row")
    mean_np = np.asarray(transform.mean, dtype=np.float64)
    scale_np = np.asarray(transform.scale, dtype=np.float64)
    if (mean_np.shape != (5,) or scale_np.shape != (5,) or
            not np.isfinite(mean_np).all() or not np.isfinite(scale_np).all() or
            np.any(scale_np <= 0)):
        raise ValueError("Expected finite fitted log means and positive scales")
    fit_mean = torch.as_tensor(mean_np, dtype=torch.float64, device=center.device)
    fit_scale = torch.as_tensor(scale_np, dtype=torch.float64, device=center.device)
    means = fit_mean + fit_scale * center
    marginal_sd = fit_scale * torch.sqrt(diag.square() + rank1.square())

    glucose = torch.special.ndtr((means[:, 0] - _LOG_GLU) / marginal_sd[:, 0])
    triglyceride = torch.special.ndtr((means[:, 3] - _LOG_TG) / marginal_sd[:, 3])
    hdl_cutoff = torch.where(sex_t == 1, 40., 50.).to(torch.float64).log()
    low_hdl = torch.special.ndtr((hdl_cutoff - means[:, 4]) / marginal_sd[:, 4])

    bp = _bp_gl(means, diag, rank1, fit_scale, order)
    if adaptive:
        active = torch.arange(len(center), device=center.device)
        previous = bp
        next_order = order * 2
        while active.numel():
            candidate = _bp_gl(means[active], diag[active], rank1[active], fit_scale, next_order)
            needs_more = (previous.detach() - candidate.detach()).abs() > tolerance
            if next_order == 1024 and bool(needs_more.any()):
                raise FloatingPointError("BP GL quadrature did not converge through order 1024")
            chosen = torch.where(needs_more, candidate, previous)
            bp = bp.index_copy(0, active, chosen)
            if not bool(needs_more.any()):
                break
            active = active[needs_more]
            previous = candidate[needs_more]
            next_order *= 2
    result = torch.stack((glucose, bp, triglyceride, low_hdl), dim=1)
    if (not bool(torch.isfinite(result).all()) or
            not bool(((result >= 0.) & (result <= 1.)).all())):
        raise FloatingPointError("Flag probability left [0,1] or became nonfinite")
    return result


def audit_flag_quadrature(mu: torch.Tensor, diag_scale: torch.Tensor,
                          loading: torch.Tensor, sex, transform: JointTargetTransform,
                          tolerance: float = 1e-5) -> dict:
    """Compare fixed GL32 and GL64 flag risks without changing training order.

    The difference is a diagnostic, not a rigorous accuracy bound against the
    audited v23 joint law.  A failed check should be reported explicitly.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    with torch.no_grad():
        p32 = torch_flag_probabilities(mu, diag_scale, loading, sex, transform,
                                       order=32, adaptive=False)
        p64 = torch_flag_probabilities(mu, diag_scale, loading, sex, transform,
                                       order=64, adaptive=False)
        diff = (p32 - p64).abs()
        return {"order_low": 32, "order_high": 64,
                "max_abs_difference": float(diff.max().item()),
                "count_rows_above_tolerance": int((diff.max(dim=1).values > tolerance).sum().item()),
                "tolerance": float(tolerance),
                "within_tolerance": bool((diff <= tolerance).all().item())}


def shrink_diag_scale(diag_scale, fit_mean, alpha):
    """Shrink [n,5] diagonal scales toward fit-role means; loading stays intact.

    ``fit_mean`` is a caller-supplied positive weighted fit-role mean [5].
    ``alpha`` is a scalar or five coordinate weights in [0,1].  The return
    type matches ``diag_scale`` (NumPy array or Torch tensor).
    """
    is_torch = isinstance(diag_scale, torch.Tensor)
    if is_torch:
        diag = diag_scale
        mean = torch.as_tensor(fit_mean, device=diag.device, dtype=diag.dtype)
        shrink = torch.as_tensor(alpha, device=diag.device, dtype=diag.dtype)
        if (not diag.is_floating_point() or diag.ndim != 2 or
                diag.shape[1] != 5 or diag.shape[0] == 0 or
                mean.shape != (5,) or shrink.shape not in ((), (5,)) or
                not all(bool(torch.isfinite(v).all()) for v in (diag, mean, shrink)) or
                not bool((diag > 0).all()) or not bool((mean > 0).all()) or
                not bool(((shrink >= 0) & (shrink <= 1)).all())):
            raise ValueError("Expected positive finite [n,5] scales, [5] fit mean, alpha in [0,1]")
        return (1. - shrink) * diag + shrink * mean
    diag = np.asarray(diag_scale)
    mean = np.asarray(fit_mean, dtype=np.float64)
    shrink = np.asarray(alpha, dtype=np.float64)
    if (diag.ndim != 2 or diag.shape[1] != 5 or diag.shape[0] == 0 or
            mean.shape != (5,) or shrink.shape not in ((), (5,)) or
            not all(np.isfinite(v).all() for v in (diag, mean, shrink)) or
            np.any(diag <= 0) or np.any(mean <= 0) or
            np.any(shrink < 0) or np.any(shrink > 1)):
        raise ValueError("Expected positive finite [n,5] scales, [5] fit mean, alpha in [0,1]")
    return (1. - shrink) * diag + shrink * mean


def mixture_probabilities(q16stack, weights=None) -> dict[str, np.ndarray]:
    """Mix seed joint laws [seed,n,16], deriving p4 and any from mean p16.

    ``weights`` is an optional nonnegative [seed] vector. Zero-weight seeds
    are ignored, but the total weight must be positive. Outputs use the v23
    keys ``state_probabilities``, ``marginals4`` and ``any_probability``.
    """
    q = np.asarray(q16stack, dtype=np.float64)
    if (q.ndim != 3 or q.shape[0] == 0 or q.shape[1] == 0 or q.shape[2] != 16 or
            not np.isfinite(q).all() or np.any(q < 0) or
            not np.allclose(q.sum(axis=2), 1., atol=1e-8, rtol=0)):
        raise ValueError("Expected finite normalized nonnegative [seed,n,16] joint laws")
    if weights is None:
        w = np.ones(q.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if (w.shape != (q.shape[0],) or not np.isfinite(w).all() or
            np.any(w < 0) or w.sum() <= 0):
        raise ValueError("Expected nonnegative seed weights with positive total")
    p16 = np.tensordot(w / w.sum(), q, axes=(0, 0))
    p16 = p16 / p16.sum(axis=1, keepdims=True)
    p4 = p16 @ _BITS
    any_probability = 1. - p16[:, 0]
    if (np.any(any_probability + 1e-12 < p4.max(axis=1)) or
            np.any(any_probability - 1e-12 > np.minimum(1., p4.sum(axis=1)))):
        raise FloatingPointError("Mixture any-event risk violates Frechet bounds")
    return {"state_probabilities": p16, "marginals4": p4,
            "any_probability": any_probability}
