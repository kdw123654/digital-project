"""Original v28 model definitions and training loop for prepared 114-column arrays.

The cohort, role assignment, preprocessing, frozen anchors and evaluation
harness are deliberately outside this portable source export. See README.md.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from . import _training as ud
from .clinical_v23_distribution import gaussian_nll, joint_probabilities
from .clinical_v24_distribution import torch_flag_probabilities
from .event_field_v21 import EventPlasticFieldV21, MLPControlV21
from .event_field_v23 import BODY_INDICES, V23JointModel


SEEDS = (42, 43, 44, 45, 46)


LR_GRID, WD_GRID = (3e-4, 1e-3), (0., 1e-3)


MAX_EPOCHS, PATIENCE, BATCH, CLIP_NORM, LINEAR_LR_RATIO = 300, 30, 256, 5., .1


DESIGNS = ("binary", "joint")


ARCHS = {"Linear": {},
         "MLP": {"head_width": 64, "hidden_layers": 1},
         "EPF": {"dimension": 12, "steps": 4, "head_width": 32, "plasticity": True},
         "EPF_noP": {"dimension": 12, "steps": 4, "head_width": 32, "plasticity": False},
         "EPF_step1": {"dimension": 12, "steps": 1, "head_width": 32, "plasticity": True}}


CELLS = tuple(f"{d}_{a}" for d in DESIGNS for a in ARCHS)


LOGIT_C_GRID = (.001, .003, .01, .03, .1, .3, 1., 3.)


EQUIVALENCE_MARGIN = .005


H1 = tuple((f"joint_{a}", f"binary_{a}") for a in ARCHS)


H2 = tuple(("joint_EPF", f"joint_{a}") for a in ("EPF_noP", "EPF_step1", "MLP", "Linear"))


class BinaryModel(nn.Module):
    """Four Bernoulli logits from Linear / MLP / EPF cores with an LR anchor."""

    def __init__(self, arch: str, seed: int) -> None:
        super().__init__()
        options = ARCHS[arch]
        self.arch = arch
        if arch == "Linear":
            self.core = None
            self.linear = nn.Linear(114, 4)
        elif arch == "MLP":
            self.core = MLPControlV21(114, 4, head_width=options["head_width"],
                                      hidden_layers=options["hidden_layers"],
                                      normalization="layer", dropout=.1, seed=seed,
                                      initialization="linear_anchor", use_linear_skip=True,
                                      residual_gate="trainable")
        else:
            self.core = EventPlasticFieldV21(
                114, 4, dimension=options["dimension"], steps=options["steps"],
                head_width=options["head_width"], dropout=.1, feature_norm="fit", seed=seed,
                initialization="linear_anchor", use_linear_skip=True, activation_target=.15,
                plasticity=options["plasticity"], quartic=True,
                plastic_readout="field" if options["plasticity"] else "capacity_control",
                residual_gate="trainable", event_mode="hard")
        if self.core is not None:
            with torch.no_grad():
                self.core.gate_logits.fill_(math.log(.3 / .7))

    @torch.no_grad()
    def initialize_on_fit(self, x_fit, coef, bias) -> None:
        coef = torch.as_tensor(coef, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)
        if self.core is None:
            self.linear.weight.copy_(coef)
            self.linear.bias.copy_(bias)
            return
        device = next(self.parameters()).device
        self.core.set_linear(coef.to(device), bias.to(device))
        self.core.initialize_on_fit(torch.as_tensor(x_fit, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) if self.core is None else self.core(x)


def logistic_anchor(item, labels_fit, labels_val) -> tuple[np.ndarray, np.ndarray, list]:
    """Per-flag fold-fit LR; C by validation weighted BCE (the binary Ridge analogue)."""
    coef, bias, chosen = np.zeros((4, 114)), np.zeros(4), []
    w = item.fit_weights / item.fit_weights.mean()
    for j in range(4):
        best = None
        for c in LOGIT_C_GRID:
            model = LogisticRegression(C=c, max_iter=5000, tol=1e-8)
            model.fit(item.fit_x, labels_fit[:, j], sample_weight=w)
            p = np.clip(model.predict_proba(item.validation_x)[:, 1], 1e-12, 1 - 1e-12)
            loss = np.average(-(labels_val[:, j] * np.log(p) + (1 - labels_val[:, j]) *
                                np.log1p(-p)), weights=item.validation_weights)
            if best is None or loss < best[0]:
                best = (loss, c, model.coef_[0], model.intercept_[0])
        coef[j], bias[j] = best[2], best[3]
        chosen.append(best[1])
    return coef, bias, chosen


def build(cell: str, seed: int, inputs: dict, device: torch.device) -> nn.Module:
    design, arch = cell.split("_", 1)
    torch.manual_seed(seed)
    item, base = inputs["item"], inputs["baseline"]
    if design == "binary":
        model = BinaryModel(arch, seed).to(device)
        model.initialize_on_fit(item.fit_x, *inputs["logit_anchor"][:2])
        return model
    options = dict(ARCHS[arch])
    family = "Linear" if arch == "Linear" else ("MLP" if arch == "MLP" else "EPF")
    variant = "bio_noplastic" if options.pop("plasticity", True) is False else "bio"
    model = V23JointModel(family, variant, 114, seed=seed, **options, dropout=.1,
                          gate_init=.3).to(device)
    model.initialize_on_fit(item.fit_x, base["coef"], base["bias"], base["residual_cov"],
                            BODY_INDICES)
    return model


def _bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits.double(), y.double(), reduction="none").mean(1)


def _objective(design, model, x, target, w):
    if design == "binary":
        return (_bce(model(x), target) * w).mean()
    return ud._loss(model, model.forward_parts(x), target, w)


@torch.no_grad()
def predict(design, model, x, sex, target, transform, device) -> dict:
    """p4, per-row proper score (BCE or NLL), and the model's own 16-state law."""
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    if design == "binary":
        p4 = torch.sigmoid(model(xt)).double().cpu().numpy()
        score = _bce(torch.logit(torch.as_tensor(p4).clamp(1e-12, 1 - 1e-12)),
                     torch.as_tensor(target)).numpy()
        bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
        p16 = np.prod(np.where(bits[None], p4[:, None, :], 1 - p4[:, None, :]), axis=2)
        return {"p4": p4, "score": score, "p16": p16}
    parts = model.forward_parts(xt)
    mu, diag, loading = (parts[k].double().cpu() for k in ("mu", "diag_scale", "loading"))
    return {"p4": torch_flag_probabilities(mu, diag, loading, sex, transform).numpy(),
            "score": gaussian_nll(mu, diag, loading, torch.as_tensor(target),
                                  reduction="none").numpy(),
            "p16": joint_probabilities(mu, diag, loading, sex, transform)["state_probabilities"]}


def train_unit(inputs: dict, cell: str, lr: float, wd: float, seed: int,
               device: torch.device) -> tuple[dict, dict]:
    design = cell.split("_", 1)[0]
    item, roles = inputs["item"], inputs["roles"]
    model = build(cell, seed, inputs, device)
    optimizer = ud._optimizer(model, lr, wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10)
    x = torch.as_tensor(item.fit_x, dtype=torch.float32, device=device)
    target = torch.as_tensor(inputs["labels_fit"] if design == "binary" else item.fit_z,
                             dtype=torch.float64, device=device)
    w = torch.as_tensor(item.fit_weights / item.fit_weights.mean(), dtype=torch.float64,
                        device=device)
    val = roles["validation"]
    vx = torch.as_tensor(val["x"], dtype=torch.float32, device=device)
    vt = torch.as_tensor(val["y"] if design == "binary" else val["z"], dtype=torch.float64,
                         device=device)
    vw = torch.as_tensor(val["w"] / val["w"].mean(), dtype=torch.float64, device=device)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for ids in np.array_split(rng.permutation(item.fit_n), max(1, item.fit_n // BATCH)):
            loss = _objective(design, model, x[ids], target[ids], w[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {cell} seed{seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if design == "binary":
                score = float((_bce(model(vx), vt) * vw).mean())
            else:
                parts = model.forward_parts(vx)
                score = float((gaussian_nll(parts["mu"], parts["diag_scale"], parts["loading"],
                                            vt, reduction="none") * vw).mean())
        scheduler.step(score)
        if score < best - 1e-4:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    transform = item.target_transform
    preds = {}
    for role, row in roles.items():
        out = predict(design, model, row["x"], row["sex"],
                      row["y"] if design == "binary" else row["z"], transform, device)
        preds[role] = {"p4": out["p4"], "score": out["score"]}
    meta = {"cell": cell, "lr": lr, "weight_decay": wd, "seed": seed, "best_epoch": best_epoch,
            "epochs_run": epoch, "best_validation_score": best,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return preds, {"meta": meta, "state_dict": best_state}


def configs() -> list[tuple[float, float]]:
    return [(lr, wd) for lr in LR_GRID for wd in WD_GRID]
