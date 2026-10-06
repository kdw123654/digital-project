"""The seven v30 conditions behind one fit/predict interface.

Direct six-group models (LR, RF, LightGBM, MLP) output p6 directly. Joint
models (EPF, MLP, Linear on ``V23JointModel(..., "bio", ...)``) learn one
conditional Gaussian over five transformed measurements; their p16 comes from
the audited ``joint_probabilities`` and p6 = p16 @ M16->6. Direct p6 is never
turned into four-flag or 16-state probabilities.

Optimiser, loss, early stopping and anchors follow v26/v28 (see protocol §6).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .clinical_v23_distribution import (JointTargetTransform, gaussian_nll,
                                        joint_probabilities)
from .clinical_v24_distribution import mixture_probabilities
from .event_field_v21 import MLPControlV21
from .event_field_v23 import BODY_INDICES, V23JointModel
from .v30_data import BITS16, M16_TO_6
from .v30_metrics import check_prob, six_group_metrics

CONDITIONS = ("LR", "RF", "LGBM", "MLP", "jEPF", "jMLP", "jLinear")
CONDITION_LABEL = {"LR": "direct LR", "RF": "direct RF", "LGBM": "direct LightGBM",
                   "MLP": "direct MLP", "jEPF": "joint EPF", "jMLP": "joint MLP",
                   "jLinear": "joint Linear"}
DIRECT = ("LR", "RF", "LGBM", "MLP")
JOINT = ("jEPF", "jMLP", "jLinear")
TORCH_CONDITIONS = ("MLP", "jEPF", "jMLP", "jLinear")
SELECT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
LR_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.)
RIDGE_ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
MAX_EPOCHS, PATIENCE, CLIP_NORM, LINEAR_LR_RATIO, COV_PENALTY = 300, 30, 5., .1, .001
JOINT_OPTIONS = {"jEPF": ("EPF", {"dimension": 12, "steps": 4, "head_width": 32}),
                 "jMLP": ("MLP", {"head_width": 64, "hidden_layers": 1}),
                 "jLinear": ("Linear", {})}
TREE_THREADS = 6


def configs(condition: str) -> list[dict]:
    if condition == "LR":
        return [{"C": c} for c in LR_C_GRID]
    if condition == "RF":
        return [{"max_features": mf, "min_samples_leaf": leaf, "max_depth": depth}
                for mf in ("sqrt", 0.2) for leaf in (5, 20) for depth in (None, 12)]
    if condition == "LGBM":
        return [{"num_leaves": nl, "min_child_samples": mc, "reg_lambda": rl}
                for nl in (7, 31) for mc in (20, 60) for rl in (0., 10.)]
    return [{"lr": lr, "weight_decay": wd, "batch": b}
            for lr in (3e-4, 1e-3) for wd in (0., 1e-3) for b in (128, 256)]


def select_seeds(condition: str) -> tuple[int, ...]:
    return (0,) if condition == "LR" else SELECT_SEEDS


def final_seeds(condition: str) -> tuple[int, ...]:
    return (0,) if condition == "LR" else FINAL_SEEDS


@dataclass
class Role:
    """Encoded arrays for one role. ``w`` is normalised to mean 1 on the role."""
    x: np.ndarray
    group: np.ndarray
    w: np.ndarray
    raw_w: np.ndarray
    numbers: np.ndarray
    sex: np.ndarray
    state: np.ndarray
    flags: np.ndarray
    index: np.ndarray


@dataclass
class Shared:
    """Fit-only anchors shared by all configs of one (panel, split)."""
    lr_anchor: dict | None = None
    ridge: dict | None = None
    transform: JointTargetTransform | None = None
    fit_z: np.ndarray | None = None
    val_z: np.ndarray | None = None
    records: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- anchors
def fit_lr_anchor(fit: Role, val: Role) -> dict:
    from sklearn.linear_model import LogisticRegression
    trials, best = [], None
    for c in LR_C_GRID:
        model = LogisticRegression(C=c, max_iter=3000, tol=1e-6)
        model.fit(fit.x, fit.group, sample_weight=fit.w)
        p = np.clip(model.predict_proba(val.x), 1e-12, 1)
        loss = float(np.average(-np.log(p[np.arange(len(val.group)), val.group]), weights=val.w))
        trials.append({"C": c, "validation_weighted_logloss": loss})
        if best is None or loss < best[0] - 1e-12:
            best = (loss, c, model.coef_.copy(), model.intercept_.copy())
    return {"C": best[1], "coef": best[2], "bias": best[3], "trials": trials}


def fit_joint_shared(fit: Role, val: Role) -> tuple[JointTargetTransform, dict, np.ndarray, np.ndarray]:
    from sklearn.linear_model import Ridge
    transform = JointTargetTransform().fit(fit.numbers, fit.raw_w)
    z_fit, z_val = transform.transform(fit.numbers), transform.transform(val.numbers)
    trials, models = [], []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(fit.x.astype(np.float64), z_fit, sample_weight=fit.w)
        mse = float(np.average(((model.predict(val.x.astype(np.float64)) - z_val) ** 2).mean(1), weights=val.w))
        trials.append({"alpha": alpha, "validation_mean_MSE": mse})
        models.append(model)
    chosen = min(range(len(trials)), key=lambda j: (trials[j]["validation_mean_MSE"], RIDGE_ALPHAS[j]))
    model = models[chosen]
    residual = z_fit - model.predict(fit.x.astype(np.float64))
    centered = residual - np.average(residual, axis=0, weights=fit.w)
    covariance = (centered * fit.w[:, None]).T @ centered / fit.w.sum()
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise FloatingPointError("Invalid fit residual covariance")
    ridge = {"alpha": RIDGE_ALPHAS[chosen], "coef": model.coef_.astype(np.float64),
             "bias": model.intercept_.astype(np.float64), "residual_cov": covariance, "trials": trials}
    return transform, ridge, z_fit, z_val


def build_shared(conditions, fit: Role, val: Role) -> Shared:
    shared = Shared()
    if "MLP" in conditions:
        shared.lr_anchor = fit_lr_anchor(fit, val)
        shared.records["mlp_lr_anchor"] = {"C": shared.lr_anchor["C"], "trials": shared.lr_anchor["trials"]}
    if any(c in JOINT for c in conditions):
        shared.transform, shared.ridge, shared.fit_z, shared.val_z = fit_joint_shared(fit, val)
        shared.records["joint_ridge_anchor"] = {"alpha": shared.ridge["alpha"], "trials": shared.ridge["trials"],
                                                "target_transform": shared.transform.to_dict()}
    return shared


# --------------------------------------------------------------------------- torch helpers
def _optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """v26 parameter groups: linear skip at lr*0.1; weight decay on matrices only."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        linear = name.startswith(("linear_skip.", "core.linear_skip."))
        groups.setdefault((linear, parameter.ndim >= 2), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr * (LINEAR_LR_RATIO if linear else 1.),
                               "weight_decay": weight_decay if matrix else 0.}
                              for (linear, matrix), params in groups.items()])


def _joint_loss(parts, z, w):
    nll = gaussian_nll(parts["mu"], parts["diag_scale"], parts["loading"], z, reduction="none")
    total = nll + COV_PENALTY * parts["covariance_penalty_per_row"].to(nll)
    return (total * w).mean()


def _ce(logits, target, w):
    return (nn.functional.cross_entropy(logits.double(), target, reduction="none") * w).mean()


class TorchSixGroupMLP(nn.Module):
    """MLPControlV21 with six softmax outputs and a multinomial-LR linear anchor."""

    def __init__(self, input_dim: int, seed: int):
        super().__init__()
        self.core = MLPControlV21(input_dim, 6, hidden_layers=1, head_width=64, normalization="layer",
                                  dropout=.1, seed=seed, initialization="linear_anchor",
                                  use_linear_skip=True, residual_gate="trainable")
        with torch.no_grad():
            self.core.gate_logits.fill_(math.log(.3 / .7))

    def forward(self, x):
        return self.core(x)


def _count_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------------------------------------------------------- trained model
@dataclass
class Trained:
    condition: str
    config: dict
    seed: int
    model: object
    meta: dict
    transform: JointTargetTransform | None = None

    # --- prediction
    def predict(self, x: np.ndarray, sex: np.ndarray, *, audited: bool = True, device=None) -> dict:
        if self.condition in DIRECT:
            return {"p6": self._direct_p6(x, device)}
        mu, diag, loading = self.joint_parameters(x, device)
        if audited:
            out = joint_probabilities(mu, diag, loading, np.asarray(sex, int), self.transform,
                                      order=64, audit=True)
            p16 = out["state_probabilities"]
        else:
            p16 = p16_fast(mu, diag, loading, sex, self.transform)
        return {"p16": p16, "p4": p16 @ BITS16.astype(np.float64), "p6": p16 @ M16_TO_6}

    def _direct_p6(self, x, device=None) -> np.ndarray:
        if self.condition in ("LR", "RF", "LGBM"):
            p = self.model.predict_proba(x)
            return check_prob(p)
        model = self.model
        dev = next(model.parameters()).device if device is None else device
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 65536):
                xt = torch.as_tensor(x[start:start + 65536], dtype=torch.float32, device=dev)
                out.append(torch.softmax(model(xt).double(), dim=1).cpu().numpy())
        return check_prob(np.concatenate(out))

    def joint_parameters(self, x, device=None):
        model = self.model
        dev = next(model.parameters()).device if device is None else device
        model.eval()
        mus, diags, loads = [], [], []
        with torch.no_grad():
            for start in range(0, len(x), 65536):
                parts = model.forward_parts(torch.as_tensor(x[start:start + 65536], dtype=torch.float32, device=dev))
                mus.append(parts["mu"].double())
                diags.append(parts["diag_scale"].double())
                loads.append(parts["loading"].double())
        return torch.cat(mus), torch.cat(diags), torch.cat(loads)


# --------------------------------------------------------------------------- training
def train_one(condition: str, config: dict, seed: int, fit: Role, val: Role, shared: Shared,
              device: torch.device) -> Trained:
    start = time.perf_counter()
    if condition == "LR":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=config["C"], max_iter=3000, tol=1e-6)
        model.fit(fit.x, fit.group, sample_weight=fit.w)
        meta = {"n_iter": int(np.max(model.n_iter_)), "parameters": int(model.coef_.size + model.intercept_.size)}
    elif condition == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, n_jobs=TREE_THREADS, random_state=seed, **config)
        model.fit(fit.x, fit.group, sample_weight=fit.w)
        meta = {"parameters": int(sum(t.tree_.node_count for t in model.estimators_)),
                "parameter_unit": "total tree nodes"}
    elif condition == "LGBM":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(objective="multiclass", learning_rate=0.03, n_estimators=3000,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=seed, n_jobs=TREE_THREADS, deterministic=True,
                                   force_row_wise=True, verbose=-1, **config)
        model.fit(fit.x, fit.group, sample_weight=fit.w, eval_set=[(val.x, val.group)],
                  eval_sample_weight=[val.w], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        info = model.booster_.dump_model(num_iteration=model.best_iteration_)["tree_info"]
        meta = {"best_iteration": int(model.best_iteration_),
                "parameters": int(sum(tree["num_leaves"] for tree in info)),
                "parameter_unit": "total leaves in used trees"}
    elif condition == "MLP":
        model, meta = _train_torch_direct(config, seed, fit, val, shared, device)
    else:
        model, meta = _train_torch_joint(condition, config, seed, fit, val, shared, device)
    meta["fit_seconds"] = time.perf_counter() - start
    return Trained(condition, dict(config), seed, model, meta,
                   shared.transform if condition in JOINT else None)


def _train_loop(model, config, seed, n_fit, batch_loss, val_loss):
    optimizer = _optimizer(model, config["lr"], config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, stale, epoch = math.inf, None, 0, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for ids in np.array_split(rng.permutation(n_fit), max(1, n_fit // config["batch"])):
            loss = batch_loss(ids)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(val_loss())
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation loss")
        scheduler.step(score)
        if score < best - 1e-4:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "epochs_run": epoch, "best_validation_loss": best,
            "parameters": _count_trainable(model)}


def _train_torch_direct(config, seed, fit, val, shared, device):
    torch.manual_seed(seed)
    model = TorchSixGroupMLP(fit.x.shape[1], seed).to(device)
    model.core.set_linear(torch.as_tensor(shared.lr_anchor["coef"], dtype=torch.float32),
                          torch.as_tensor(shared.lr_anchor["bias"], dtype=torch.float32))
    model.core.initialize_on_fit(torch.as_tensor(fit.x, dtype=torch.float32, device=device))
    x = torch.as_tensor(fit.x, dtype=torch.float32, device=device)
    y = torch.as_tensor(fit.group, dtype=torch.long, device=device)
    w = torch.as_tensor(fit.w, dtype=torch.float64, device=device)
    vx = torch.as_tensor(val.x, dtype=torch.float32, device=device)
    vy = torch.as_tensor(val.group, dtype=torch.long, device=device)
    vw = torch.as_tensor(val.w, dtype=torch.float64, device=device)
    meta = _train_loop(model, config, seed, len(fit.x),
                       lambda ids: _ce(model(x[ids]), y[ids], w[ids]),
                       lambda: _ce(model(vx), vy, vw))
    meta["anchor_C"] = shared.lr_anchor["C"]
    return model, meta


def _train_torch_joint(condition, config, seed, fit, val, shared, device):
    family, options = JOINT_OPTIONS[condition]
    torch.manual_seed(seed)
    model = V23JointModel(family, "bio", fit.x.shape[1], seed=seed, **options, dropout=.1,
                          gate_init=.3).to(device)
    ridge = shared.ridge
    model.initialize_on_fit(fit.x, ridge["coef"], ridge["bias"], ridge["residual_cov"], BODY_INDICES)
    x = torch.as_tensor(fit.x, dtype=torch.float32, device=device)
    z = torch.as_tensor(shared.fit_z, dtype=torch.float64, device=device)
    w = torch.as_tensor(fit.w, dtype=torch.float64, device=device)
    vx = torch.as_tensor(val.x, dtype=torch.float32, device=device)
    vz = torch.as_tensor(shared.val_z, dtype=torch.float64, device=device)
    vw = torch.as_tensor(val.w, dtype=torch.float64, device=device)

    def val_loss():
        parts = model.forward_parts(vx)
        return (gaussian_nll(parts["mu"], parts["diag_scale"], parts["loading"], vz, reduction="none") * vw).mean()

    meta = _train_loop(model, config, seed, len(fit.x),
                       lambda ids: _joint_loss(model.forward_parts(x[ids]), z[ids], w[ids]), val_loss)
    meta["ridge_alpha"] = ridge["alpha"]
    return model, meta


# --------------------------------------------------------------------------- ensembles
def ensemble(condition: str, outputs: list[dict]) -> dict:
    """Seed mixture: joint -> mean p16 then aggregate; direct -> mean p6."""
    if condition in JOINT:
        mix = mixture_probabilities(np.stack([o["p16"] for o in outputs]))
        p16 = mix["state_probabilities"]
        result = {"p16": p16, "p4": mix["marginals4"], "p6": p16 @ M16_TO_6}
        check_invariants(result)
        return result
    return {"p6": check_prob(np.mean([o["p6"] for o in outputs], axis=0))}


def check_invariants(out: dict) -> None:
    if "p16" in out:
        p16 = out["p16"]
        if not (np.isfinite(p16).all() and (p16 >= -1e-12).all() and np.allclose(p16.sum(1), 1, atol=1e-8)):
            raise FloatingPointError("p16 left the simplex")
        if not np.allclose(out["p4"], p16 @ BITS16.astype(np.float64), atol=1e-9):
            raise FloatingPointError("p4 disagrees with p16 @ BITS")
    check_prob(out["p6"])


def validation_macro_ap(trained: Trained, val: Role, device=None) -> tuple[float, dict]:
    out = trained.predict(val.x, val.sex, audited=True, device=device)
    check_invariants(out)
    metrics = six_group_metrics(val.group, out["p6"], val.raw_w)
    return metrics["macro_ap"], out


# --------------------------------------------------------------------------- fast p16 (explanations)
_GH_CACHE: dict = {}


def _nodes(order: int, device):
    key = (order, str(device))
    if key not in _GH_CACHE:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(order)
        gl_x, gl_w = np.polynomial.legendre.leggauss(order)
        _GH_CACHE[key] = tuple(torch.as_tensor(a, dtype=torch.float64, device=device) for a in
                               (gh_x * math.sqrt(2), gh_w / math.sqrt(math.pi), (gl_x + 1) / 2, gl_w / 2))
    return _GH_CACHE[key]


def p16_fast(mu, diag, loading, sex, transform: JointTargetTransform, order: int = 64,
             device: str | None = None, chunk: int = 131072) -> np.ndarray:
    """Unaudited GH(order)/GL(order) integral of the v23 joint law, on GPU if available.

    Mirrors ``clinical_v23_distribution._integrate_gh`` term by term (tested for
    agreement); used only inside explanation passes, never for reported metrics.
    """
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    factors, factor_w, inner_x, inner_w = _nodes(order, dev)
    mean = torch.as_tensor(transform.mean, dtype=torch.float64, device=dev)
    scale = torch.as_tensor(transform.scale, dtype=torch.float64, device=dev)
    bits = torch.as_tensor(BITS16.astype(bool), device=dev)
    log_glu, log_dbp, log_tg = math.log(100.), math.log(85.), math.log(150.)
    sex_np = np.asarray(sex)
    out = []
    for start in range(0, len(sex_np), chunk):
        sl = slice(start, start + chunk)
        center = torch.as_tensor(mu[sl], dtype=torch.float64, device=dev)
        dg = torch.as_tensor(diag[sl], dtype=torch.float64, device=dev)
        r1 = torch.as_tensor(loading[sl], dtype=torch.float64, device=dev)
        hdl = torch.log(torch.where(torch.as_tensor(sex_np[sl], device=dev) == 1,
                                    torch.tensor(40., dtype=torch.float64, device=dev),
                                    torch.tensor(50., dtype=torch.float64, device=dev)))
        log_scales = scale * dg
        q16 = torch.zeros((len(center), 16), dtype=torch.float64, device=dev)
        for f, fw in zip(factors, factor_w):
            lm = mean + scale * (center + f * r1)
            below = torch.special.ndtr((log_dbp - lm[:, 1]) / log_scales[:, 1])
            u = below[:, None] * inner_x[None, :]
            dbp = torch.exp(lm[:, 1, None] + log_scales[:, 1, None] * torch.special.ndtri(u))
            pp = torch.special.ndtr((torch.log(130. - dbp) - lm[:, 2, None]) / log_scales[:, 2, None])
            safe = below * (pp @ inner_w)
            probs = torch.stack((torch.special.ndtr((lm[:, 0] - log_glu) / log_scales[:, 0]), 1 - safe,
                                 torch.special.ndtr((lm[:, 3] - log_tg) / log_scales[:, 3]),
                                 torch.special.ndtr((hdl - lm[:, 4]) / log_scales[:, 4])), dim=1)
            states = torch.where(bits[None], probs[:, None, :], 1 - probs[:, None, :]).prod(2)
            q16 += fw * states
        q16 = q16 / q16.sum(1, keepdim=True)
        out.append(q16.cpu().numpy())
    return np.concatenate(out)


def load_trained(models_dir, seed: int, device) -> Trained:
    """Rebuild a saved final model (spec JSON + joblib or torch state)."""
    import json
    from pathlib import Path

    import joblib

    models_dir = Path(models_dir)
    spec = json.loads((models_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
    condition = spec["condition"]
    transform = JointTargetTransform.from_dict(spec["transform"]) if spec["transform"] else None
    if condition in ("LR", "RF", "LGBM"):
        model = joblib.load(models_dir / f"seed{seed}.joblib")
    else:
        if condition == "MLP":
            model = TorchSixGroupMLP(spec["input_dim"], seed).to(device)
        else:
            family, options = JOINT_OPTIONS[condition]
            model = V23JointModel(family, "bio", spec["input_dim"], seed=seed, **options, dropout=.1,
                                  gate_init=.3).to(device)
        state = torch.load(models_dir / f"seed{seed}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
    return Trained(condition, spec["config"], seed, model, {"loaded": True}, transform)
