"""Five-measurement conditional Gaussian heads over the frozen v21 field.

The EPF uses its existing z, event trace, and antisymmetric P recurrence. Its
mean sees only the five d-dimensional state blocks; the four P projections
enter only the input-dependent covariance head. The capacity and no-plasticity
controls retain the same readout shape and replace the P projection with the
pre-existing z/trace antisymmetric projection.

All statistics and anchors are supplied by the fit role. Measured outcomes are
never part of ``forward`` inputs. This module does not interpret latent states
as measured physiological states.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import nn

from .event_field_v21 import EventPlasticFieldV21, MLPControlV21


Family = Literal["EPF", "MLP", "Linear"]
Variant = Literal["diag", "shared", "bio", "bio_capacity", "bio_noplastic"]
BIO_VARIANTS = frozenset({"bio", "bio_capacity", "bio_noplastic"})
BODY_INDICES = (0, 1, 2, 3, 8)  # age, BMI, waist, WHtR, sex==2 in the 72-column prefix
MIN_DIAG_SCALE = 1e-3


def _validate_residual_covariance(value) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (5, 5) or not np.isfinite(covariance).all():
        raise ValueError("Finite [5,5] fit residual covariance required")
    if not np.allclose(covariance, covariance.T, atol=1e-7, rtol=1e-7):
        raise ValueError("Fit residual covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    if (np.diag(covariance) <= 0).any():
        raise ValueError("Fit residual variances must be positive")
    if np.linalg.eigvalsh(covariance).min() < -1e-6 * np.diag(covariance).max():
        raise ValueError("Fit residual covariance is not positive semidefinite")
    return covariance


def _fit_diagonal_and_rank_one(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one bounded factor to off-diagonal covariance, preserving variances.

    The top eigenvector gives a deterministic direction. Its amplitude is the
    least-squares fit to off-diagonal entries for that direction. Each loading
    is bounded to leave at least 20% of its observed marginal variance in the
    diagonal. The tiny fallback keeps the adaptive loading head differentiable
    when empirical off-diagonals are exactly zero.
    """
    variance = np.maximum(np.diag(covariance).copy(), MIN_DIAG_SCALE**2)
    offdiag = covariance.copy()
    np.fill_diagonal(offdiag, 0.)
    eigenvalues, eigenvectors = np.linalg.eigh(offdiag)
    direction = eigenvectors[:, -1]
    pair = np.triu_indices(5, 1)
    basis = direction[pair[0]] * direction[pair[1]]
    denominator = float(basis @ basis)
    amplitude_squared = (max(float(offdiag[pair] @ basis) / denominator, 0.)
                         if denominator > 1e-12 and eigenvalues[-1] > 1e-12 else 0.)
    loading = math.sqrt(amplitude_squared) * direction
    if np.linalg.norm(loading) < 1e-8:
        loading = 0.01 * np.sqrt(variance) / math.sqrt(5)
    cap = np.sqrt(0.8 * variance)
    loading = np.clip(loading, -cap, cap)
    pivot = int(np.argmax(np.abs(loading)))
    if loading[pivot] < 0:
        loading = -loading
    diagonal_scale = np.sqrt(np.maximum(variance - loading**2, MIN_DIAG_SCALE**2))
    return diagonal_scale, loading


class EventPlasticFieldV23(EventPlasticFieldV21):
    """V21 recurrence with a smaller fit-only floor for its four P projections."""

    def __init__(self, *args, plastic_feature_floor: float = .005, **kwargs) -> None:
        if not math.isfinite(plastic_feature_floor) or plastic_feature_floor <= 0:
            raise ValueError("Positive finite plastic feature floor required")
        super().__init__(*args, **kwargs)
        if self.plastic_readout == "none":
            raise ValueError("V23 requires four plastic or capacity-control features")
        self.plastic_feature_floor = float(plastic_feature_floor)
        # V21's first layer has four P columns. The v23 mean must not consume
        # them; the covariance pathway receives those columns separately.
        old = self.readout[0]
        seed = kwargs.get("seed", args[7] if len(args) > 7 else 42)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(int(seed) + 2300)
            mean_first = nn.Linear(5 * self.dimension, old.out_features,
                                   device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            mean_first.weight.copy_(old.weight[:, :5 * self.dimension])
            mean_first.bias.copy_(old.bias)
        self.readout[0] = mean_first

    @torch.no_grad()
    def initialize_on_fit(self, x_fit, batch_size: int = 256) -> dict:
        info = super().initialize_on_fit(x_fit, batch_size=batch_size)
        x = torch.as_tensor(x_fit, device=self.drive.weight.device,
                            dtype=self.drive.weight.dtype)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not bool(torch.isfinite(x).all()):
            raise ValueError("Finite [n,input_dim] fit features required")
        was_training = self.training
        self.eval()
        try:
            raw = torch.cat([self._field_pass(part, "none")[0][:, 5 * self.dimension:]
                             for part in x.split(batch_size)], dim=0)
            raw_std = raw.std(0, unbiased=False)
            effective = raw_std.clamp_min(self.plastic_feature_floor)
            self.feature_scale_extra.copy_(effective.to(self.feature_scale_extra))
            info = dict(info)
            info["plastic_feature_scale_v21_pre_override"] = info["plastic_feature_scale"]
            info["plastic_feature_raw_std"] = raw_std.detach().cpu().tolist()
            info["plastic_feature_scale"] = effective.detach().cpu().tolist()
            info["plastic_feature_floor"] = self.plastic_feature_floor
            return info
        finally:
            self.train(was_training)

    def forward_parts(self, x: torch.Tensor, return_diagnostics: bool = False) -> dict:
        if not bool(self.fit_initialized):
            raise RuntimeError("Initialize EPF on clean fit features before forward")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong EPF input shape")
        features, diagnostics = self._field_pass(
            x, "full" if return_diagnostics else "none")
        boundary = 5 * self.dimension
        mean_features = ((features[:, :boundary] - self.feature_mean) /
                         self.feature_scale)
        covariance_features = ((features[:, boundary:] - self.feature_mean_extra) /
                               self.feature_scale_extra)
        residual_raw = self.readout(mean_features)
        gate = torch.sigmoid(self.gate_logits)
        residual = gate * residual_raw
        linear = self.linear_skip(x)
        return {"logits": linear + residual, "linear": linear,
                "residual": residual, "residual_raw": residual_raw,
                "gate": gate, "cov_features": covariance_features,
                "diagnostics": diagnostics}


class V23JointModel(nn.Module):
    """Five standardized transformed measurements with diagonal plus rank-1 Σ."""

    def __init__(self, family: Family, variant: Variant, input_dim: int, *,
                 seed: int = 42, dimension: int = 12, steps: int = 4,
                 head_width: int = 32, hidden_layers: Literal[1, 2] = 1,
                 dropout: float = .1, gate_init: float = .3,
                 plastic_feature_floor: float = .005) -> None:
        super().__init__()
        if family not in ("EPF", "MLP", "Linear") or variant not in (
                "diag", "shared", "bio", "bio_capacity", "bio_noplastic"):
            raise ValueError("Unknown family or covariance variant")
        if family != "EPF" and variant in ("bio_capacity", "bio_noplastic"):
            raise ValueError("Plasticity controls require EPF")
        if input_dim <= max(BODY_INDICES) or head_width < 1 or dimension < 2 or steps < 1:
            raise ValueError("Invalid model/input dimensions")
        if not 0 < gate_init < 1 or not 0 <= dropout < 1:
            raise ValueError("Gate must be in (0,1) and dropout in [0,1)")
        if not math.isfinite(plastic_feature_floor) or plastic_feature_floor <= 0:
            raise ValueError("Positive finite plastic feature floor required")
        self.family, self.variant, self.input_dim = family, variant, int(input_dim)
        self.seed, self.dimension, self.steps = int(seed), int(dimension), int(steps)
        self.head_width, self.hidden_layers = int(head_width), int(hidden_layers)
        self.dropout, self.gate_init = float(dropout), float(gate_init)
        self.plastic_feature_floor = float(plastic_feature_floor)
        adaptive = variant in BIO_VARIANTS
        self.register_buffer("fit_initialized", torch.tensor(False))
        self.register_buffer("body_indices", torch.full((5,), -1, dtype=torch.long))
        self.register_buffer("base_diag_scale", torch.ones(5))
        self.register_buffer("base_loading", torch.zeros(5))
        self.register_buffer("fit_residual_cov", torch.eye(5))

        if family == "EPF":
            readout_kind = ("capacity_control" if variant in
                            ("bio_capacity", "bio_noplastic") else "field")
            self.core = EventPlasticFieldV23(
                input_dim, 5, dimension=dimension, steps=steps,
                head_width=head_width, dropout=dropout, feature_norm="fit",
                seed=seed, initialization="linear_anchor", use_linear_skip=True,
                activation_target=.15, plasticity=variant != "bio_noplastic",
                quartic=True, plastic_readout=readout_kind,
                residual_gate="trainable", event_mode="hard",
                numeric_embedding=False, plastic_feature_floor=plastic_feature_floor)
            if not adaptive:
                self.core.plastic_u.requires_grad_(False)
                self.core.plastic_v.requires_grad_(False)
            self.mean_linear = None
        elif family == "MLP":
            self.core = MLPControlV21(
                input_dim, 5, head_width=head_width,
                hidden_layers=hidden_layers, normalization="layer",
                dropout=dropout, seed=seed, initialization="linear_anchor",
                use_linear_skip=True, residual_gate="trainable",
                numeric_embedding=False)
            self.mean_linear = None
        else:
            self.core = None
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed + 2301)
                self.mean_linear = nn.Linear(input_dim, 5)
        if self.core is not None:
            with torch.no_grad():
                self.core.gate_logits.fill_(math.log(gate_init / (1 - gate_init)))

        if adaptive:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed + 2302)
                if family == "EPF":
                    self.cov_projection = None  # the four fitted P/capacity coordinates
                else:
                    source_width = head_width if family == "MLP" else 5
                    self.cov_projection = nn.Linear(source_width, 4)
                self.covariance_scale_head = nn.Linear(5, 5)
                self.covariance_loading_head = nn.Linear(4, 5)
            with torch.no_grad():
                self.covariance_scale_head.weight.zero_()
                self.covariance_scale_head.bias.zero_()
                self.covariance_loading_head.weight.zero_()
                self.covariance_loading_head.bias.zero_()
        else:
            self.cov_projection = None
            self.covariance_scale_head = None
            self.covariance_loading_head = None

    @torch.no_grad()
    def initialize_on_fit(self, x_fit, ridge_coef, ridge_bias, residual_cov,
                          body_indices=BODY_INDICES, *, batch_size: int = 256) -> dict:
        if bool(self.fit_initialized):
            raise RuntimeError("V23 fit initialization already completed")
        device = next(self.parameters()).device
        x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
        coef = torch.as_tensor(ridge_coef, dtype=torch.float32, device=device)
        bias = torch.as_tensor(ridge_bias, dtype=torch.float32, device=device)
        if (x.ndim != 2 or x.shape[1] != self.input_dim or len(x) < 2 or
                coef.shape != (5, self.input_dim) or bias.shape != (5,) or
                not bool(torch.isfinite(x).all() and torch.isfinite(coef).all() and
                         torch.isfinite(bias).all())):
            raise ValueError("Finite fit x and five-row transformed Ridge anchor required")
        indices = tuple(int(i) for i in body_indices)
        if indices != BODY_INDICES:
            raise ValueError("V23 body context must be age,BMI,waist,WHtR,sex columns")
        covariance = _validate_residual_covariance(residual_cov)
        variance = np.maximum(np.diag(covariance), MIN_DIAG_SCALE**2)
        if self.variant == "diag":
            diagonal_scale, loading = np.sqrt(variance), np.zeros(5)
        else:
            diagonal_scale, loading = _fit_diagonal_and_rank_one(covariance)
        self.body_indices.copy_(torch.as_tensor(indices, dtype=torch.long,
                                                device=self.body_indices.device))
        self.base_diag_scale.copy_(torch.as_tensor(diagonal_scale, device=device,
                                                   dtype=self.base_diag_scale.dtype))
        self.base_loading.copy_(torch.as_tensor(loading, device=device,
                                                dtype=self.base_loading.dtype))
        self.fit_residual_cov.copy_(torch.as_tensor(covariance, device=device,
                                                    dtype=self.fit_residual_cov.dtype))
        if self.core is None:
            self.mean_linear.weight.copy_(coef)
            self.mean_linear.bias.copy_(bias)
            feature_info = None
        else:
            self.core.set_linear(coef, bias)
            feature_info = self.core.initialize_on_fit(x, batch_size=batch_size) if (
                self.family == "EPF") else self.core.initialize_on_fit(x)
        self.fit_initialized.fill_(True)
        return {"fit_n": len(x), "body_indices": list(indices),
                "base_diag_scale": diagonal_scale.tolist(),
                "base_loading": loading.tolist(),
                "fit_residual_cov": covariance.tolist(),
                "feature": feature_info,
                "optimized_trainable_parameters": self.active_parameter_count(),
                "parameter_groups": self.parameter_groups()}

    def _mean_and_cov_features(self, x: torch.Tensor,
                               return_diagnostics: bool) -> tuple[dict, torch.Tensor | None]:
        if self.family == "EPF":
            parts = self.core.forward_parts(x, return_diagnostics=return_diagnostics)
            return parts, parts["cov_features"] if self.variant in BIO_VARIANTS else None
        if self.family == "MLP":
            hidden = self.core.readout[:-1](x)
            residual_raw = self.core.readout[-1](hidden)
            gate = torch.sigmoid(self.core.gate_logits)
            residual = residual_raw * gate
            linear = self.core.linear_skip(x)
            features = (torch.tanh(self.cov_projection(hidden))
                        if self.variant in BIO_VARIANTS else None)
            return {"linear": linear, "residual": residual,
                    "residual_raw": residual_raw, "gate": gate,
                    "diagnostics": None}, features
        linear = self.mean_linear(x)
        features = (torch.tanh(self.cov_projection(x[:, self.body_indices]))
                    if self.variant in BIO_VARIANTS else None)
        return {"linear": linear, "residual": torch.zeros_like(linear),
                "residual_raw": torch.zeros_like(linear),
                "gate": torch.zeros(5, dtype=linear.dtype, device=linear.device),
                "diagnostics": None}, features

    def forward_parts(self, x: torch.Tensor,
                      return_diagnostics: bool = False) -> dict:
        if not bool(self.fit_initialized):
            raise RuntimeError("Initialize from clean fit statistics before forward")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("[batch,input_dim] inputs required")
        parts, covariance_features = self._mean_and_cov_features(x, return_diagnostics)
        mu = parts["linear"] + parts["residual"]
        base_scale = self.base_diag_scale.to(mu)
        base_loading = self.base_loading.to(mu)
        if self.variant in BIO_VARIANTS:
            context = x[:, self.body_indices]
            diag_scale = (base_scale * torch.exp(
                .5 * torch.tanh(self.covariance_scale_head(context))))
            loading = (base_loading + .5 * base_scale * torch.tanh(
                self.covariance_loading_head(covariance_features)))
            covariance_penalty_per_row = (
                torch.log(diag_scale / base_scale).square().mean(1) +
                ((loading - base_loading) / base_scale).square().mean(1))
            covariance_penalty = covariance_penalty_per_row.mean()
        else:
            diag_scale = base_scale.expand(len(x), -1)
            loading = base_loading.expand(len(x), -1)
            covariance_penalty = mu.new_zeros(())
            covariance_penalty_per_row = mu.new_zeros((len(x),))
        covariance = (torch.diag_embed(diag_scale.square()) +
                      loading.unsqueeze(2) * loading.unsqueeze(1))
        return {"mu": mu, "diag_scale": diag_scale, "loading": loading,
                "covariance": covariance, "covariance_penalty": covariance_penalty,
                "covariance_penalty_per_row": covariance_penalty_per_row,
                "linear": parts["linear"], "residual": parts["residual"],
                "residual_raw": parts["residual_raw"], "gate": parts["gate"],
                "cov_features": covariance_features,
                "diagnostics": parts["diagnostics"]}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = self.forward_parts(x)
        return parts["mu"], parts["diag_scale"], parts["loading"]

    def parameter_groups(self) -> dict[str, list[str]]:
        groups = {"field_matrix": [], "field_other": [], "covariance": [],
                  "linear_skip": [], "frozen": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                group = "frozen"
            elif name.startswith(("core.linear_skip.", "mean_linear.")):
                group = "linear_skip"
            elif name.startswith(("cov_projection.", "covariance_scale_head.",
                                  "covariance_loading_head.")):
                group = "covariance"
            elif parameter.ndim >= 2:
                group = "field_matrix"
            else:
                group = "field_other"
            groups[group].append(name)
        return groups

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def model_spec(self) -> dict:
        return {"family": self.family, "variant": self.variant,
                "input_dim": self.input_dim, "seed": self.seed,
                "dimension": self.dimension, "steps": self.steps,
                "head_width": self.head_width, "hidden_layers": self.hidden_layers,
                "dropout": self.dropout, "gate_init": self.gate_init,
                "plastic_feature_floor": self.plastic_feature_floor,
                "body_indices": self.body_indices.detach().cpu().tolist(),
                "fit_initialized": bool(self.fit_initialized),
                "covariance_kind": "diagonal_plus_rank_one",
                "optimized_trainable_parameters": self.active_parameter_count(),
                "stored_parameters": sum(p.numel() for p in self.parameters()),
                "parameter_groups": self.parameter_groups()}
