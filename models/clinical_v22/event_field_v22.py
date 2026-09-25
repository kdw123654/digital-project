"""Multi-objective v22 heads over the unchanged v21 single-field/MLP cores.

The target transform and feature encoder live outside this module. In
particular, laboratory values are targets only and never enter ``forward``.
The state16 arm uses four linear skip logits expanded by BITS plus one shared
field/MLP residual, so the independent LR anchor receives one L2 penalty.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .event_field_v21 import EventPlasticFieldV21, MLPControlV21


Mode = Literal["binary4", "aux9", "union5", "full10", "regression5",
               "state16", "focal4", "labelweight4", "focal_labelweight4"]
Family = Literal["EPF", "MLP"]
HEAD_WIDTHS = {"binary": 4, "aux": 5, "union": 1}
MODE_HEADS: dict[str, tuple[str, ...]] = {
    "binary4": ("binary",),
    "aux9": ("binary", "aux"),
    "union5": ("binary", "union"),
    "full10": ("binary", "aux", "union"),
    "regression5": ("aux",),
    "state16": (),
    "focal4": ("binary",),
    "labelweight4": ("binary",),
    "focal_labelweight4": ("binary",),
}
LAMBDA_AUX = .25
LAMBDA_UNION = .25
FOCAL_GAMMA = 2.0
TASK_WEIGHT_MIN = .5
TASK_WEIGHT_MAX = 2.0
STATE_LOGIT_EPS = 1e-6


def _bits() -> torch.Tensor:
    states = torch.arange(16, dtype=torch.long)
    powers = 2 ** torch.arange(4, dtype=torch.long)
    return ((states[:, None] // powers[None, :]) % 2).float()


def _finite_fit(value, columns: int | None, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    shape_ok = tensor.ndim == (1 if columns is None else 2)
    if columns is not None:
        shape_ok = shape_ok and tensor.shape[1] == columns
    if not shape_ok or len(tensor) == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Expected finite nonempty {name} with {columns or 1} columns")
    return tensor


def _weighted_entropy(probability: torch.Tensor) -> torch.Tensor:
    if not bool(((probability > 0) & (probability < 1)).all()):
        raise ValueError("Every active fit label needs both classes")
    return -(probability * probability.log() + (1 - probability) * torch.log1p(-probability))


def _bounded_mean_one_task_weights(prevalence: torch.Tensor) -> torch.Tensor:
    """Scale then cap, solving the scale so the final capped mean is one."""
    raw = torch.rsqrt(prevalence * (1 - prevalence))
    low, high = 0., 1.
    while float(torch.clamp(raw * high, TASK_WEIGHT_MIN, TASK_WEIGHT_MAX).mean()) < 1:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if float(torch.clamp(raw * mid, TASK_WEIGHT_MIN, TASK_WEIGHT_MAX).mean()) < 1:
            low = mid
        else:
            high = mid
    value = torch.clamp(raw * ((low + high) / 2), TASK_WEIGHT_MIN, TASK_WEIGHT_MAX)
    if not bool(torch.isclose(value.mean(), torch.tensor(1., dtype=value.dtype), atol=1e-12)):
        raise RuntimeError("Final capped task weights did not average to one")
    return value


class V22MultiTask(nn.Module):
    """One v21 core and only the rows used by the selected target objective."""

    def __init__(self, family: Family, mode: Mode, input_dim: int, *,
                 seed: int = 42, dimension: int = 12, steps: int = 4,
                 head_width: int = 32, hidden_layers: Literal[1, 2] = 1,
                 normalization: Literal["none", "layer"] = "layer",
                 dropout: float = 0., feature_norm: Literal["fit", "none"] = "fit",
                 plastic_readout: Literal["none", "field", "capacity_control"] = "field",
                 residual_gate: Literal["off", "trainable"] = "trainable",
                 plasticity: bool = True, quartic: bool = True,
                 activation_target: float | None = .15) -> None:
        super().__init__()
        if family not in ("EPF", "MLP") or mode not in MODE_HEADS or input_dim < 1:
            raise ValueError("Unknown family/mode or invalid input dimension")
        self.family, self.mode, self.input_dim, self.seed = family, mode, input_dim, seed
        self.heads = MODE_HEADS[mode]
        self.head_slices: dict[str, slice] = {}
        offset = 0
        for head in self.heads:
            self.head_slices[head] = slice(offset, offset + HEAD_WIDTHS[head])
            offset += HEAD_WIDTHS[head]
        self.output_dim = 16 if mode == "state16" else offset
        state_mode = mode == "state16"
        if family == "EPF":
            self.core = EventPlasticFieldV21(
                input_dim, self.output_dim, dimension=dimension, steps=steps,
                head_width=head_width, dropout=dropout, feature_norm=feature_norm,
                seed=seed, initialization="native" if state_mode else "linear_anchor",
                use_linear_skip=not state_mode, activation_target=activation_target,
                plasticity=plasticity, quartic=quartic, plastic_readout=plastic_readout,
                residual_gate=residual_gate, event_mode="hard", numeric_embedding=False)
        else:
            self.core = MLPControlV21(
                input_dim, self.output_dim, head_width=head_width,
                hidden_layers=hidden_layers, normalization=normalization,
                dropout=dropout, seed=seed,
                initialization="native" if state_mode else "linear_anchor",
                use_linear_skip=not state_mode, residual_gate=residual_gate,
                numeric_embedding=False)
        if state_mode:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed + 2201)
                self.linear_skip4 = nn.Linear(input_dim, 4)
        else:
            self.linear_skip4 = None
        self.register_buffer("bits", _bits())
        self.register_buffer("anchor_initialized", torch.tensor(False))
        self.register_buffer("loss_initialized", torch.tensor(False))
        for name in ("B0", "R0", "U0", "S0", "fit_weight_mean"):
            self.register_buffer(name, torch.tensor(float("nan"), dtype=torch.float64))
        self.register_buffer("fit_unique_n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("task_weights", torch.ones(4, dtype=torch.float64))

    def initialize_on_fit(self, clean_fit_x, batch_size: int = 256) -> dict:
        """Re-fit field/embedding buffers once, including after backbone copy."""
        if self.family == "EPF":
            return self.core.initialize_on_fit(clean_fit_x, batch_size=batch_size)
        return self.core.initialize_on_fit(clean_fit_x)

    @torch.no_grad()
    def initialize_loss_on_fit(self, labels4, numbers_z, raw_weights) -> dict:
        """Freeze target-only scales and survey weight mean from clean fit rows."""
        if bool(self.loss_initialized):
            raise RuntimeError("Fit loss statistics already initialized")
        y = _finite_fit(labels4, 4, "four fit labels")
        z = _finite_fit(numbers_z, 5, "five standardized fit measurements")
        w = _finite_fit(raw_weights, None, "fit survey weights")
        if len(y) != len(z) or len(y) != len(w) or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("Aligned binary labels and numeric targets required")
        if not bool((w > 0).all()):
            raise ValueError("Positive fit survey weights required")
        wn = w / w.mean()
        p = (wn[:, None] * y).mean(0)
        union = y.any(1).to(y.dtype)
        pu = (wn * union).mean()
        index = (y.long() * (2 ** torch.arange(4))).sum(1)
        state_mass = torch.bincount(index, weights=wn, minlength=16) / len(y)
        state_entropy = -(state_mass[state_mass > 0] * state_mass[state_mass > 0].log()).sum() / 4
        scales = torch.stack((_weighted_entropy(p).mean(),
                              (wn[:, None] * z.square()).mean(),
                              _weighted_entropy(pu), state_entropy))
        if not bool(torch.isfinite(scales).all() and (scales > 1e-8).all()):
            raise ValueError("Fit-only objective scale is zero or nonfinite")
        for name, value in zip(("B0", "R0", "U0", "S0"), scales, strict=True):
            getattr(self, name).copy_(value.to(getattr(self, name)))
        self.fit_weight_mean.copy_(w.mean().to(self.fit_weight_mean))
        self.fit_unique_n.fill_(len(y))
        self.task_weights.copy_(_bounded_mean_one_task_weights(p).to(self.task_weights))
        self.loss_initialized.fill_(True)
        return {"rows": len(y), "B0": float(self.B0), "R0": float(self.R0),
                "U0": float(self.U0), "S0": float(self.S0),
                "fit_weight_mean": float(self.fit_weight_mean),
                "task_weights": self.task_weights.tolist(),
                "task_weight_rule": "inverse sqrt p(1-p), solve scale then cap [0.5,2] with final mean 1"}

    @torch.no_grad()
    def set_task_linear(self, anchors: dict[str, tuple]) -> None:
        """Copy separately fit LR, Ridge and union LR rows in active-head order."""
        required = {"binary"} if self.mode == "state16" else set(self.heads)
        if set(anchors) != required:
            raise ValueError(f"Expected exactly these anchor heads: {sorted(required)}")
        layer = self.linear_skip4 if self.mode == "state16" else self.core.linear_skip
        weight = torch.empty_like(layer.weight)
        bias = torch.empty_like(layer.bias)
        for head in ("binary",) if self.mode == "state16" else self.heads:
            sl = slice(0, 4) if self.mode == "state16" else self.head_slices[head]
            coef, intercept = anchors[head]
            candidate_w = torch.as_tensor(coef, dtype=weight.dtype, device=weight.device)
            candidate_b = torch.as_tensor(intercept, dtype=bias.dtype, device=bias.device)
            if (candidate_w.shape != (sl.stop - sl.start, self.input_dim) or
                    candidate_b.shape != (sl.stop - sl.start,) or
                    not bool(torch.isfinite(candidate_w).all() and torch.isfinite(candidate_b).all())):
                raise ValueError(f"Invalid finite {head} linear anchor shape")
            weight[sl].copy_(candidate_w)
            bias[sl].copy_(candidate_b)
        if self.mode == "state16":
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        else:
            self.core.set_linear(weight, bias)
        self.anchor_initialized.fill_(True)

    @staticmethod
    def state_probabilities(state_logits: torch.Tensor,
                            temperature: float = 1.) -> dict[str, torch.Tensor]:
        """A single positive posthoc temperature preserves joint coherence."""
        if state_logits.ndim != 2 or state_logits.shape[1] != 16:
            raise ValueError("Expected [batch,16] state logits")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive finite joint temperature required")
        probability = torch.softmax(state_logits / temperature, dim=1)
        bits = _bits().to(device=probability.device, dtype=probability.dtype)
        marginals = probability @ bits
        any_risk = 1 - probability[:, 0]
        return {"state_probabilities": probability, "marginals4": marginals,
                "raw_any": any_risk,
                "marginal_logits_for_component_calibration": torch.logit(
                    marginals.clamp(STATE_LOGIT_EPS, 1 - STATE_LOGIT_EPS))}

    def forward_parts(self, x: torch.Tensor,
                      return_diagnostics: bool = False) -> dict:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong v22 input shape")
        core = self.core.forward_parts(x, return_diagnostics=return_diagnostics)
        if self.mode == "state16":
            four_linear = self.linear_skip4(x)
            induced = four_linear @ self.bits.T
            logits = induced + core["residual"]
            probabilities = self.state_probabilities(logits)
            return {"logits": logits, "linear": induced, "linear4": four_linear,
                    "residual": core["residual"], "residual_raw": core["residual_raw"],
                    "gate": core["gate"], "diagnostics": core["diagnostics"],
                    "state_logits": logits, "binary_logits": None, "aux_z": None,
                    "union_logit": None, "raw_components": probabilities["marginals4"],
                    "raw_component_union": None, "raw_direct_union": None,
                    "raw_joint_any": probabilities["raw_any"], **probabilities}
        logits = core["logits"]
        binary = logits[:, self.head_slices["binary"]] if "binary" in self.heads else None
        aux = logits[:, self.head_slices["aux"]] if "aux" in self.heads else None
        union = logits[:, self.head_slices["union"]].squeeze(1) if "union" in self.heads else None
        components = torch.sigmoid(binary) if binary is not None else None
        return {**core, "binary_logits": binary, "aux_z": aux,
                "union_logit": union, "state_logits": None,
                "state_probabilities": None, "marginals4": None,
                "raw_components": components,
                "raw_component_union": (1 - torch.prod(1 - components, dim=1)
                                        if components is not None else None),
                "raw_direct_union": torch.sigmoid(union) if union is not None else None,
                "raw_joint_any": None, "raw_any": None,
                "head_slices": self.head_slices}

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        parts = self.forward_parts(x, return_diagnostics=return_diagnostics)
        return (parts["logits"], parts["diagnostics"]) if return_diagnostics else parts["logits"]

    def regression_scores(self, aux_z: torch.Tensor, target_transform, sex) -> np.ndarray:
        """Raw threshold-distance ranks; no sigmoid or probability claim."""
        if "aux" not in self.heads or aux_z.ndim != 2 or aux_z.shape[1] != 5:
            raise ValueError("Five active regression predictions required")
        return target_transform.regression_scores(aux_z.detach().cpu().numpy(), sex)

    def _runtime_targets(self, labels4, numbers_z, raw_weights, parts):
        if not bool(self.loss_initialized):
            raise RuntimeError("Call initialize_loss_on_fit using clean fit targets first")
        device, dtype = parts["logits"].device, parts["logits"].dtype
        y = torch.as_tensor(labels4, device=device, dtype=dtype)
        w = torch.as_tensor(raw_weights, device=device, dtype=dtype)
        if (y.ndim != 2 or y.shape != (len(parts["logits"]), 4) or
                w.shape != (len(y),) or not bool(torch.isfinite(y).all() and torch.isfinite(w).all()) or
                not bool(((y == 0) | (y == 1)).all() and (w > 0).all())):
            raise ValueError("Aligned finite binary targets and positive survey weights required")
        z = None
        if "aux" in self.heads:
            z = torch.as_tensor(numbers_z, device=device, dtype=dtype)
            if z.shape != (len(y), 5) or not bool(torch.isfinite(z).all()):
                raise ValueError("Five finite standardized measurements required")
        return y, z, w / self.fit_weight_mean.to(device=device, dtype=dtype)

    @staticmethod
    def _positive(value, name: str) -> float:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"Positive finite {name} required for an active head")
        return float(value)

    def compute_loss(self, parts: dict, labels4, numbers_z, raw_weights, *,
                     C_binary: float | None = None, C_union: float | None = None,
                     ridge_alpha: float | None = None,
                     residual_shrink: float = 0.) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Global-fit-weight minibatch estimator and matching within-task L2."""
        if not math.isfinite(residual_shrink) or residual_shrink < 0:
            raise ValueError("Nonnegative finite residual shrink required")
        y, z, w = self._runtime_targets(labels4, numbers_z, raw_weights, parts)
        logits = parts["logits"]
        zero = logits.new_zeros(())
        terms = {name: zero for name in (
            "binary_data_raw", "binary_linear_l2_raw", "binary_scaled",
            "aux_data_raw", "aux_linear_l2_raw", "aux_scaled",
            "union_data_raw", "union_linear_l2_raw", "union_scaled",
            "state_data_raw", "state_linear_l2_raw", "state_scaled",
            "gated_residual_shrink")}
        n = int(self.fit_unique_n)
        if self.mode == "state16":
            C = self._positive(C_binary, "binary C")
            indices = (y.long() * (2 ** torch.arange(4, device=y.device))).sum(1)
            data = (w * (F.cross_entropy(logits, indices, reduction="none") / 4)).mean()
            penalty = self.linear_skip4.weight.square().sum() / (2 * C * n * 4)
            terms["state_data_raw"] = data
            terms["state_linear_l2_raw"] = penalty
            terms["state_scaled"] = (data + penalty) / self.S0.to(logits)
            total = terms["state_scaled"]
        else:
            total = zero
            if "binary" in self.heads:
                C = self._positive(C_binary, "binary C")
                elemental = F.binary_cross_entropy_with_logits(
                    parts["binary_logits"], y, reduction="none")
                if self.mode in ("focal4", "focal_labelweight4"):
                    elemental = elemental * (1 - torch.exp(-elemental)).pow(FOCAL_GAMMA)
                if self.mode in ("labelweight4", "focal_labelweight4"):
                    elemental = elemental * self.task_weights.to(logits)
                data = (w * elemental.mean(1)).mean()
                weight = self.core.linear_skip.weight[self.head_slices["binary"]]
                penalty = weight.square().sum() / (2 * C * n * 4)
                terms["binary_data_raw"], terms["binary_linear_l2_raw"] = data, penalty
                terms["binary_scaled"] = (data + penalty) / self.B0.to(logits)
                total = total + terms["binary_scaled"]
            if "aux" in self.heads:
                alpha = self._positive(ridge_alpha, "Ridge alpha")
                data = (w * (parts["aux_z"] - z).square().mean(1)).mean()
                weight = self.core.linear_skip.weight[self.head_slices["aux"]]
                penalty = alpha * weight.square().sum() / (n * 5)
                factor = 1. if self.mode == "regression5" else LAMBDA_AUX
                terms["aux_data_raw"], terms["aux_linear_l2_raw"] = data, penalty
                terms["aux_scaled"] = factor * (data + penalty) / self.R0.to(logits)
                total = total + terms["aux_scaled"]
            if "union" in self.heads:
                C = self._positive(C_union, "union C")
                union_y = y.any(1).to(y.dtype)
                data = (w * F.binary_cross_entropy_with_logits(
                    parts["union_logit"], union_y, reduction="none")).mean()
                weight = self.core.linear_skip.weight[self.head_slices["union"]]
                penalty = weight.square().sum() / (2 * C * n)
                terms["union_data_raw"], terms["union_linear_l2_raw"] = data, penalty
                terms["union_scaled"] = LAMBDA_UNION * (data + penalty) / self.U0.to(logits)
                total = total + terms["union_scaled"]
        shrink = residual_shrink * (w * parts["residual"].square().mean(1)).mean()
        terms["gated_residual_shrink"] = shrink
        total = total + shrink
        terms["total"] = total.detach()
        return total, {key: value.detach() for key, value in terms.items()}

    def parameter_groups(self) -> dict[str, list[str]]:
        groups = {"field_matrix": [], "field_other": [], "linear_skip": [], "frozen": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                groups["frozen"].append(name)
            elif name.startswith("linear_skip4.") or name.startswith("core.linear_skip."):
                groups["linear_skip"].append(name)
            elif parameter.ndim >= 2:
                groups["field_matrix"].append(name)
            else:
                groups["field_other"].append(name)
        return groups

    def model_spec(self) -> dict:
        return {"family": self.family, "mode": self.mode, "input_dim": self.input_dim,
                "output_dim": self.output_dim, "heads": list(self.heads),
                "head_slices": {name: [sl.start, sl.stop] for name, sl in self.head_slices.items()},
                "core": self.core.model_spec(), "B0_R0_U0_S0_fit_only": bool(self.loss_initialized),
                "state_calibration": ({"joint_temperature": "one positive scalar fitted on calibration role",
                                       "marginal_logit_epsilon": STATE_LOGIT_EPS,
                                       "independent_component_Platt_is_not_joint_calibration": True}
                                      if self.mode == "state16" else None),
                "regression_risk": ("TargetTransform.regression_scores gives raw ranks, not probabilities"
                                    if self.mode == "regression5" else None),
                "raw_probability_contract": {"binary": "sigmoid(binary_logits) when present",
                                             "component_union": "1-product(1-raw_components), not the direct union head",
                                             "direct_union": "sigmoid(union_logit) when present",
                                             "state": "softmax16 marginals and raw_joint_any=1-p0 when state16",
                                             "calibrated": "separate calibration-role artifact, never returned as raw"},
                "parameters_total": sum(p.numel() for p in self.parameters()),
                "parameters_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "parameter_groups": self.parameter_groups()}

    @staticmethod
    def _backbone_names(model: "V22MultiTask") -> list[str]:
        names = []
        if model.family == "EPF":
            dynamics = {"h_real", "h_imag", "decay_raw", "quartic_raw", "threshold_raw",
                        "reset_raw", "trace_raw", "homeostasis_raw", "plastic_decay_raw",
                        "plastic_gain_raw", "dt_raw", "plastic_u", "plastic_v"}
            for name, _ in model.named_parameters():
                if (name.startswith("core.drive.") or name[5:] in dynamics or
                        name.startswith("core.readout.0.")):
                    names.append(name)
        else:
            final = f"core.readout.{len(model.core.readout) - 1}."
            names = [name for name, _ in model.named_parameters()
                     if name.startswith("core.readout.") and not name.startswith(final)]
        return sorted(names)

    @torch.no_grad()
    def copy_backbone_from(self, source: "V22MultiTask") -> dict:
        """Copy an explicit shared-trunk whitelist before target fit initialization.

        No head, gate, skip, target scale, fixed feature buffer, optimizer or RNG
        state is copied. The caller must set young anchors then fit clean young X.
        """
        if (not isinstance(source, V22MultiTask) or
                self.family != source.family or self.input_dim != source.input_dim or
                bool(self.core.fit_initialized) or bool(self.loss_initialized)):
            raise ValueError("Transfer requires same family/input and fresh target fit state")
        if not bool(source.core.fit_initialized):
            raise ValueError("Source backbone must already be fit initialized")
        same = ("dimension", "steps", "head_width", "dropout", "feature_norm",
                "plasticity", "quartic", "plastic_readout", "event_mode") if self.family == "EPF" else (
                    "head_width", "hidden_layers", "normalization")
        target_spec, source_spec = self.core.model_spec(), source.core.model_spec()
        if any(target_spec.get(name) != source_spec.get(name) for name in same):
            raise ValueError("Source and target shared-backbone settings differ")
        def dropout_signature(core):
            return tuple(layer.p for layer in core.readout if isinstance(layer, nn.Dropout))
        if dropout_signature(self.core) != dropout_signature(source.core):
            raise ValueError("Source and target shared-backbone dropout differs")
        target_names, source_names = self._backbone_names(self), self._backbone_names(source)
        if target_names != source_names:
            raise ValueError("Backbone parameter names differ")
        destination = dict(self.named_parameters())
        origin = dict(source.named_parameters())
        digest = hashlib.sha256()
        for name in target_names:
            left, right = destination[name], origin[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"Backbone parameter shape/dtype mismatch: {name}")
            value = right.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(value.numpy().tobytes())
            left.copy_(right.to(left.device))
        return {"source_family": source.family, "source_mode": source.mode,
                "target_mode": self.mode, "input_dim": self.input_dim,
                "copied_names": target_names, "copied_sha256": digest.hexdigest(),
                "excluded": "final readout, gate, linear skip, all fit-only buffers, loss scales, optimizer and RNG"}
