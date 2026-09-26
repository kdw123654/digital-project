"""Original v26 optimizer grouping and objective helpers used by v28."""
from __future__ import annotations

from torch import nn
import torch

from .clinical_v23_distribution import gaussian_nll


COV_PENALTY, CLIP_NORM, LINEAR_LR_RATIO = .001, 5., .1


def _optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        linear = name.startswith(("linear_skip.", "core.linear_skip."))
        key = (linear, parameter.ndim >= 2)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([{"params": params,
                               "lr": lr * (LINEAR_LR_RATIO if linear else 1.),
                               "weight_decay": weight_decay if matrix else 0.}
                              for (linear, matrix), params in groups.items()])


def _loss(model, parts, z, w):
    nll = gaussian_nll(parts["mu"], parts["diag_scale"], parts["loading"], z,
                       reduction="none")
    total = nll + COV_PENALTY * parts["covariance_penalty_per_row"].to(nll)
    if "recon_per_row" in parts and getattr(model, "recon_weight", 0.) > 0:
        total = total + model.recon_weight * parts["recon_per_row"].to(nll)
    return (total * w).mean()
