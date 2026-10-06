"""v21 mechanism probes built on the single v19 event-plastic field.

Only the 72-column linear skip can see the encoded inputs directly. Every
nonlinear EPF contribution passes through the common (z, trace, P) recurrence.
The optional six-column interpolation is shared with the MLP control and its
knots are fitted from clean fit-role inputs without targets.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .event_field_v19 import EventPlasticFieldV19, MLPControlV19, _fit_tensor
from .event_plastic_field import SurrogateEvent


NUMERIC_COLUMNS = (0, 1, 2, 3, 40, 41)
EMBEDDING_WIDTH = 4
EMBEDDING_KNOTS = 5
SMOOTH_EVENT_BETA = 5.0


class NumericPiecewiseEmbedding(nn.Module):
    """Continuous five-knot interpolation with linear edge extrapolation."""

    def __init__(self, input_dim: int = 72) -> None:
        super().__init__()
        if input_dim != 72:
            raise ValueError("The numeric embedding requires engineered72 input")
        self.input_dim = input_dim
        self.output_dim = input_dim - len(NUMERIC_COLUMNS) + len(NUMERIC_COLUMNS) * EMBEDDING_WIDTH
        self.register_buffer("columns", torch.tensor(NUMERIC_COLUMNS, dtype=torch.long))
        self.register_buffer("other_columns", torch.tensor(
            [j for j in range(input_dim) if j not in NUMERIC_COLUMNS], dtype=torch.long))
        self.register_buffer("knots", torch.full((len(NUMERIC_COLUMNS), EMBEDDING_KNOTS), float("nan")))
        self.register_buffer("fit_initialized", torch.tensor(False))
        self.values = nn.Parameter(torch.empty(len(NUMERIC_COLUMNS), EMBEDDING_KNOTS, EMBEDDING_WIDTH))
        with torch.no_grad():
            self.values.zero_()

    @torch.no_grad()
    def initialize_on_fit(self, x_fit: torch.Tensor) -> dict:
        if bool(self.fit_initialized):
            raise RuntimeError("Embedding fit initialization already completed")
        x = _fit_tensor(x_fit, self.input_dim, self.values.device)
        numeric = x.index_select(1, self.columns)
        quantiles = torch.quantile(numeric.to(torch.float64),
                                   torch.tensor((.05, .275, .5, .725, .95),
                                                dtype=torch.float64, device=x.device), dim=0).T
        fixed = quantiles.clone()
        for row in range(len(NUMERIC_COLUMNS)):
            span = float(quantiles[row, -1] - quantiles[row, 0])
            if span < 1e-6:
                fixed[row] = quantiles[row, 2] + torch.tensor((-.5, -.25, 0., .25, .5),
                                                               dtype=torch.float64, device=x.device)
            else:
                gap = max(1e-4, span * 1e-3)
                for j in range(1, EMBEDDING_KNOTS):
                    fixed[row, j] = torch.maximum(fixed[row, j], fixed[row, j - 1] + gap)
        self.knots.copy_(fixed.to(dtype=self.knots.dtype))
        if not bool((self.knots[:, 1:] > self.knots[:, :-1]).all()):
            raise RuntimeError("Numeric knots are not strictly increasing")
        # The first channel is exact identity at initialization, including
        # extrapolation. Remaining channels begin small but nonzero, so the
        # shared downstream projection and knot values both receive gradients.
        self.values[..., 0].copy_(self.knots)
        t = self.knots
        self.values[..., 1].copy_(.01 * torch.sigmoid(t))
        self.values[..., 2].copy_(.01 * torch.sigmoid(-t))
        self.values[..., 3].copy_(.01 * (.5 + t.square() / (1 + t.square())))
        self.fit_initialized.fill_(True)
        return {"rows": len(x), "numeric_columns": list(NUMERIC_COLUMNS),
                "knot_count": EMBEDDING_KNOTS, "output_dim": self.output_dim,
                "min_knot_gap": float((self.knots[:, 1:] - self.knots[:, :-1]).min())}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong numeric embedding input shape")
        if not bool(self.fit_initialized):
            raise RuntimeError("Numeric embedding needs clean fit initialization")
        transformed = []
        for j, column in enumerate(NUMERIC_COLUMNS):
            coordinate = x[:, column].contiguous()
            knots = self.knots[j].contiguous()
            segment = torch.searchsorted(knots[1:-1], coordinate).clamp(0, EMBEDDING_KNOTS - 2)
            left_knot = knots[segment]
            right_knot = knots[segment + 1]
            fraction = ((coordinate - left_knot) / (right_knot - left_knot)).unsqueeze(1)
            left = self.values[j, segment]
            right = self.values[j, segment + 1]
            transformed.append(left + fraction * (right - left))
        return torch.cat((x.index_select(1, self.other_columns),
                          torch.cat(transformed, dim=1)), dim=1)


class _V21Diagnostics:
    """Shared, detached component summaries and explicit parameter ownership."""

    def forward_parts(self, x: torch.Tensor, return_diagnostics: bool = False) -> dict:
        raise NotImplementedError

    @staticmethod
    def component_statistics(parts: dict) -> dict:
        result = {}
        for name in ("linear", "residual_raw", "residual", "logits"):
            value = parts[name].detach().to(torch.float64)
            mean = value.mean(0)
            centered = value - mean
            result[name] = {
                "mean": mean.cpu().tolist(),
                "centered_sd": centered.square().mean(0).sqrt().cpu().tolist(),
                "centered_rms": centered.square().mean(0).sqrt().cpu().tolist(),
                "rms": value.square().mean(0).sqrt().cpu().tolist(),
            }
        result["gate"] = parts["gate"].detach().cpu().tolist()
        return result

    @staticmethod
    def retain_output_gradients(parts: dict) -> None:
        for name in ("linear", "residual_raw", "residual", "logits"):
            if parts[name].requires_grad:
                parts[name].retain_grad()

    def gradient_diagnostics(self, parts: dict | None = None) -> dict:
        groups = self.parameter_groups()
        parameters = dict(self.named_parameters())
        result = {"parameter_groups": {group: names for group, names in groups.items()},
                  "parameter_grad_norm": {}}
        for group, names in groups.items():
            result["parameter_grad_norm"][group] = math.sqrt(sum(
                float(parameters[name].grad.detach().square().sum())
                for name in names if parameters[name].grad is not None))
        if parts is not None:
            result["output_grad_norm"] = {
                name: (float(parts[name].grad.detach().norm()) if parts[name].grad is not None else None)
                for name in ("linear", "residual_raw", "residual", "logits")}
        return result

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def spec(self) -> dict:
        return self.model_spec()

    def set_prior_bias(self, weighted_prevalence, *, where: Literal["linear", "readout"] = "linear") -> None:
        """Set one intercept from fit-only weighted prevalence probabilities.

        This changes no slope or field parameter. For a gated residual readout,
        divide the raw intercept by the gate so its output contribution equals
        the requested logit. In the prespecified field-only control the gate is
        off, making the raw intercept equal the prior logit.
        """
        probability = torch.as_tensor(weighted_prevalence, dtype=self.readout[-1].bias.dtype,
                                      device=self.readout[-1].bias.device)
        if probability.shape != (self.output_dim,) or not bool(
                torch.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all()):
            raise ValueError("Expected one finite prevalence strictly between zero and one per output")
        logit = torch.logit(probability)
        if where == "linear":
            if self.initialization != "native":
                raise ValueError("Linear prior bias is reserved for native initialization; keep LR anchor bias")
            if self.linear_skip is None:
                raise ValueError("linear skip is disabled")
            with torch.no_grad():
                self.linear_skip.bias.copy_(logit)
        elif where == "readout":
            gate = (torch.sigmoid(self.gate_logits.detach()) if self.gate_logits is not None
                    else torch.ones_like(logit))
            with torch.no_grad():
                self.readout[-1].bias.copy_(logit / gate)
        else:
            raise ValueError("where must be 'linear' or 'readout'")


class EventPlasticFieldV21(_V21Diagnostics, EventPlasticFieldV19):
    """One shared field with optional final antisymmetric P readout."""

    def __init__(self, input_dim: int, output_dim: int, dimension: int = 16,
                 steps: int = 6, head_width: int = 64, dropout: float = 0.0,
                 feature_norm: Literal["fit", "none"] = "fit", seed: int = 42,
                 initialization: Literal["native", "linear_anchor", "legacy"] = "native",
                 use_linear_skip: bool = True, activation_target: float | None = .15,
                 plasticity: bool = True, quartic: bool = True,
                 plastic_readout: Literal["none", "field", "capacity_control"] = "field",
                 residual_gate: Literal["off", "trainable"] = "trainable",
                 event_mode: Literal["hard", "smooth"] = "hard",
                 numeric_embedding: bool = False) -> None:
        if plastic_readout not in ("none", "field", "capacity_control"):
            raise ValueError("Unknown plastic readout")
        if residual_gate not in ("off", "trainable") or event_mode not in ("hard", "smooth"):
            raise ValueError("Unknown residual gate or event mode")
        if numeric_embedding and input_dim != 72:
            raise ValueError("Numeric embedding requires engineered72")
        super().__init__(input_dim, output_dim, dimension=dimension, steps=steps,
                         head_width=head_width, dropout=dropout, feature_norm=feature_norm,
                         seed=seed, initialization=initialization, use_linear_skip=use_linear_skip,
                         activation_target=activation_target, plasticity=plasticity, quartic=quartic)
        self.plastic_readout = plastic_readout
        self.residual_gate = residual_gate
        self.event_mode = event_mode
        self.numeric_embedding = numeric_embedding
        self.smooth_beta = SMOOTH_EVENT_BETA
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed + 2100)
            if numeric_embedding:
                self.embedding = NumericPiecewiseEmbedding(input_dim)
                old_drive = self.drive
                self.drive = nn.Linear(self.embedding.output_dim, 2 * dimension)
                # Retain the v19 identity coordinates while opening every
                # added embedding channel to first-backward gradients.
                with torch.no_grad():
                    self.drive.weight.normal_(0, .01)
                    self.drive.bias.copy_(old_drive.bias)
                    remaining = self.embedding.other_columns.tolist()
                    self.drive.weight[:, :len(remaining)].copy_(old_drive.weight[:, remaining])
                    for j, column in enumerate(NUMERIC_COLUMNS):
                        offset = len(remaining) + EMBEDDING_WIDTH * j
                        self.drive.weight[:, offset].copy_(old_drive.weight[:, column])
            else:
                self.embedding = None
            if plastic_readout != "none":
                self.plastic_u = nn.Parameter(torch.randn(4, dimension) / math.sqrt(dimension))
                self.plastic_v = nn.Parameter(torch.randn(4, dimension) / math.sqrt(dimension))
                first = self.readout[0]
                expanded = nn.Linear(5 * dimension + 4, head_width)
                with torch.no_grad():
                    expanded.weight[:, :5 * dimension].copy_(first.weight)
                    expanded.weight[:, 5 * dimension:].normal_(0, .01)
                    expanded.bias.copy_(first.bias)
                self.readout[0] = expanded
                self.register_buffer("feature_mean_extra", torch.zeros(4))
                self.register_buffer("feature_scale_extra", torch.ones(4))
            else:
                self.register_parameter("plastic_u", None)
                self.register_parameter("plastic_v", None)
            if residual_gate == "trainable":
                self.gate_logits = nn.Parameter(torch.full((output_dim,), math.log(.1 / .9)))
            else:
                self.register_parameter("gate_logits", None)
        # The ablations retain the same serialized architecture but prevent
        # optimizer steps on parameters with no path to the output.
        if not quartic:
            self.quartic_raw.requires_grad_(False)
        if not plasticity:
            self.plastic_gain_raw.requires_grad_(False)
            self.plastic_decay_raw.requires_grad_(False)
            if plastic_readout == "field":
                self.plastic_u.requires_grad_(False)
                self.plastic_v.requires_grad_(False)

    def _nonlinear_input(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.embedding is None else self.embedding(x)

    def _plastic_features(self, z: torch.Tensor, trace: torch.Tensor,
                          plastic: torch.Tensor) -> torch.Tensor:
        if self.plastic_readout == "none":
            return torch.empty((len(z), 0), dtype=trace.dtype, device=trace.device)
        u = F.normalize(self.plastic_u, dim=1, eps=1e-8)
        v = F.normalize(self.plastic_v, dim=1, eps=1e-8)
        if self.plastic_readout == "field":
            return torch.einsum("rd,bdk,rk->br", u, plastic, v)
        real = z.real
        return (real @ u.T) * (trace @ v.T) - (real @ v.T) * (trace @ u.T)

    def _field_pass(self, x: torch.Tensor,
                    diagnostics: Literal["none", "rate", "full"] = "none"):
        """The v19 recurrence, changing only the selectable event function."""
        batch = len(x)
        z, trace, plastic = self.initial_state(batch, x.device)
        d = self.dimension
        dt = F.softplus(self.dt_raw).clamp(.025, .5)
        eye = torch.eye(d, dtype=torch.complex64, device=x.device)
        base_real = (self.h_real + self.h_real.T) / 2
        base_imag = (self.h_imag - self.h_imag.T) / 2
        static_h = torch.complex(base_real, base_imag)
        base_unitary = torch.linalg.solve(eye + 1j * dt * static_h,
                                          eye - 1j * dt * static_h)
        encoded = .1 * self.drive(self._nonlinear_input(x)).reshape(-1, d, 2)
        drive = torch.view_as_complex(encoded.contiguous())
        decay = .5 + .499 * torch.sigmoid(self.decay_raw)
        coefficient = F.softplus(self.quartic_raw) if self.quartic else torch.zeros_like(self.quartic_raw)
        reset = .05 + .9 * torch.sigmoid(self.reset_raw)
        homeostasis = F.softplus(self.homeostasis_raw)
        threshold_base = F.softplus(self.threshold_raw) + .01
        trace_decay = .5 + .499 * torch.sigmoid(self.trace_raw)
        plastic_decay = .5 + .499 * torch.sigmoid(self.plastic_decay_raw)
        gain = .2 * torch.sigmoid(self.plastic_gain_raw)
        events = [] if diagnostics == "full" else None
        energies = [] if diagnostics == "full" else None
        event_total = None
        solve_infos = []
        for step in range(self.steps):
            if step < 2 or not self.plasticity:
                propagated = z @ base_unitary.T
            else:
                h = torch.complex(base_real.expand(batch, -1, -1),
                                  base_imag.expand(batch, -1, -1) + plastic)
                numerator = (eye - 1j * dt * h) @ z.unsqueeze(-1)
                denominator = eye + 1j * dt * h
                if x.device.type == "cuda":
                    solved, info = torch.linalg.solve_ex(denominator, numerator, check_errors=False)
                    solve_infos.append(info)
                else:
                    solved = torch.linalg.solve(denominator, numerator)
                propagated = solved.squeeze(-1)
            driven = decay * propagated + drive
            energy = driven.abs().square()
            evolved = driven * torch.exp(-1j * dt * coefficient * energy)
            threshold = threshold_base * (1 + homeostasis * trace)
            q = (energy - threshold) / (threshold.detach() + .1)
            event = SurrogateEvent.apply(q) if self.event_mode == "hard" else torch.sigmoid(self.smooth_beta * q)
            denominator = torch.maximum(energy, threshold.detach()).clamp_min(1e-6)
            fraction = (1 - reset * threshold * event / denominator).clamp_min(.01)
            next_z = evolved * torch.sqrt(fraction)
            next_trace = trace_decay * trace + (1 - trace_decay) * event
            timing = trace.unsqueeze(2) * event.unsqueeze(1) - event.unsqueeze(2) * trace.unsqueeze(1)
            if self.plasticity:
                next_plastic = .5 * torch.tanh((plastic_decay * plastic + gain * timing) / .5)
            else:
                next_plastic = torch.zeros_like(plastic)
            z, trace, plastic = next_z, next_trace, next_plastic
            if diagnostics == "full":
                events.append(event)
                energies.append(energy)
            elif diagnostics == "rate":
                event_total = event.sum() if event_total is None else event_total + event.sum()
        if solve_infos:
            invalid = torch.cat([info.reshape(-1) for info in solve_infos]).ne(0).any()
            if bool(invalid):
                raise FloatingPointError("Cayley solve failed before optimizer update")
        features = torch.cat([z.real, z.imag, z.abs().square(), trace, energy], dim=1)
        plastic_features = self._plastic_features(z, trace, plastic)
        if self.plastic_readout != "none":
            features = torch.cat((features, plastic_features), dim=1)
        if diagnostics == "none":
            return features, None
        if diagnostics == "rate":
            return features, {"event_rate": event_total / (self.steps * batch * d)}
        return features, {
            "event_rate": torch.stack(events).mean(),
            "field_energy": torch.stack(energies).mean(),
            "plastic_norm": plastic.norm(dim=(1, 2)).mean(),
            "field_features": features,
            "plastic_features": plastic_features,
            "final_plastic": plastic,
            "event_mode": self.event_mode,
        }

    def forward_parts(self, x: torch.Tensor, return_diagnostics: bool = False) -> dict:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong field input shape")
        features, diagnostics = self._field_pass(x, "full" if return_diagnostics else "none")
        if self.feature_norm == "fit":
            if self.plastic_readout == "none":
                features = (features - self.feature_mean) / self.feature_scale
            else:
                mean = torch.cat((self.feature_mean, self.feature_mean_extra))
                scale = torch.cat((self.feature_scale, self.feature_scale_extra))
                features = (features - mean) / scale
        residual_raw = self.readout(features)
        gate = (torch.sigmoid(self.gate_logits) if self.gate_logits is not None
                else residual_raw.new_ones(self.output_dim))
        residual = residual_raw * gate
        linear = (self.linear_skip(x) if self.linear_skip is not None
                  else torch.zeros_like(residual))
        parts = {"logits": linear + residual, "linear": linear, "residual": residual,
                 "residual_raw": residual_raw, "gate": gate, "diagnostics": diagnostics}
        if diagnostics is not None:
            diagnostics["components"] = self.component_statistics(parts)
        return parts

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        parts = self.forward_parts(x, return_diagnostics=return_diagnostics)
        return (parts["logits"], parts["diagnostics"]) if return_diagnostics else parts["logits"]

    @torch.no_grad()
    def initialize_on_fit(self, x_fit, batch_size: int = 256) -> dict:
        """Fit the six knots, event-rate threshold and feature scale on clean fit rows."""
        if bool(self.fit_initialized):
            raise RuntimeError("Fit initialization already completed")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = _fit_tensor(x_fit, self.input_dim, self.drive.weight.device)
        was_training = self.training
        self.eval()
        try:
            embedding_info = self.embedding.initialize_on_fit(x) if self.embedding is not None else None
            probe = x.index_select(0, self._activation_probe_indices(len(x), x.device))
            threshold_scale = 1.0
            original = F.softplus(self.threshold_raw).detach().clone()

            def rate(scale: float) -> float:
                calibrated = (original * scale).clamp_min(1e-7)
                self.threshold_raw.copy_(torch.log(torch.expm1(calibrated)))
                _, info = self._field_pass(probe, "rate")
                return float(info["event_rate"])

            if self.activation_target is not None:
                candidates = [(1.0, rate(1.0)), (.01, rate(.01))]
                lo, hi = .01, 1.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    observed = rate(mid)
                    candidates.append((mid, observed))
                    if observed > self.activation_target:
                        lo = mid
                    else:
                        hi = mid
                threshold_scale, fit_rate = min(candidates,
                                                key=lambda item: abs(item[1] - self.activation_target))
                calibrated = (original * threshold_scale).clamp_min(1e-7)
                self.threshold_raw.copy_(torch.log(torch.expm1(calibrated)))
            else:
                fit_rate = rate(1.0)
            self.activation_fit_rate.fill_(fit_rate)
            if self.feature_norm == "fit":
                blocks = [self._field_pass(chunk, "none")[0].detach().to("cpu", dtype=torch.float64)
                          for chunk in x.split(batch_size)]
                all_features = torch.cat(blocks)
                mean = all_features.mean(0)
                scale = all_features.std(0, unbiased=False).clamp_min(.02)
                self.feature_mean.copy_(mean[:5 * self.dimension].to(self.feature_mean))
                self.feature_scale.copy_(scale[:5 * self.dimension].to(self.feature_scale))
                if self.plastic_readout != "none":
                    self.feature_mean_extra.copy_(mean[5 * self.dimension:].to(self.feature_mean_extra))
                    self.feature_scale_extra.copy_(scale[5 * self.dimension:].to(self.feature_scale_extra))
            self.fit_initialized.fill_(True)
            return {"rows": len(x), "activation_target": self.activation_target,
                    "activation_fit_rate": fit_rate, "activation_rate_definition":
                    "mean of selected hard or smooth event over fit probe, all steps and field dimensions",
                    "event_mode": self.event_mode, "smooth_beta": self.smooth_beta,
                    "threshold_scale": threshold_scale, "feature_norm": self.feature_norm,
                    "feature_scale_min": float(self.feature_scale.min()),
                    "feature_scale_median": float(self.feature_scale.median()),
                    "plastic_feature_scale": (self.feature_scale_extra.tolist()
                                              if self.plastic_readout != "none" else None),
                    "embedding": embedding_info}
        finally:
            self.train(was_training)

    def initialize_native_linear(self, prior_bias=None) -> None:
        """Set native skip to w=0 and b=0 or fit-weighted prevalence logits.

        The runner must calculate prevalence solely from fit labels/weights.
        Linear-anchor starts are rejected so an LR intercept cannot be replaced.
        """
        if self.initialization != "native":
            raise ValueError("Native linear initialization requires initialization='native'")
        if self.linear_skip is None:
            raise ValueError("linear skip is disabled for this field")
        bias = torch.zeros(self.output_dim) if prior_bias is None else prior_bias
        self.set_linear(torch.zeros_like(self.linear_skip.weight), bias)

    def parameter_groups(self) -> dict[str, list[str]]:
        groups = {"field_dynamics": [], "field_readout": [], "plastic_readout": [],
                  "embedding": [], "residual_gate": [], "linear_skip": [], "inactive": []}
        for name, _ in self.named_parameters():
            if name.startswith("linear_skip."):
                groups["linear_skip"].append(name)
            elif name.startswith("readout."):
                groups["field_readout"].append(name)
            elif name in ("plastic_u", "plastic_v"):
                groups["plastic_readout"].append(name)
                if self.plastic_readout == "field" and not self.plasticity:
                    groups["inactive"].append(name)
            elif name.startswith("embedding."):
                groups["embedding"].append(name)
            elif name == "gate_logits":
                groups["residual_gate"].append(name)
            else:
                groups["field_dynamics"].append(name)
                if not self.quartic and name == "quartic_raw":
                    groups["inactive"].append(name)
                if not self.plasticity and name in ("plastic_gain_raw", "plastic_decay_raw"):
                    groups["inactive"].append(name)
        return groups

    def model_spec(self) -> dict:
        inactive_slices = ([f"readout.0.weight[:, {5 * self.dimension}:] -> final P feature zeros"]
                           if self.plastic_readout == "field" and not self.plasticity else [])
        inactive_reasons = {}
        if not self.quartic:
            inactive_reasons["quartic_raw"] = "quartic phase coefficient replaced by zero"
        if not self.plasticity:
            inactive_reasons["plastic_gain_raw"] = "plastic update disabled"
            inactive_reasons["plastic_decay_raw"] = "plastic update disabled"
            if self.plastic_readout == "field":
                inactive_reasons["plastic_u"] = "final P remains zero"
                inactive_reasons["plastic_v"] = "final P remains zero"
        return {"family": "EPF", "input_dim": self.input_dim,
                "nonlinear_input_dim": self.drive.in_features, "output_dim": self.output_dim,
                "dimension": self.dimension, "steps": self.steps,
                "head_width": self.readout[0].out_features, "dropout": self.dropout,
                "feature_norm": self.feature_norm, "initialization": self.initialization,
                "activation_target": self.activation_target,
                "plasticity": self.plasticity, "quartic": self.quartic,
                "plastic_readout": self.plastic_readout, "residual_gate": self.residual_gate,
                "event_mode": self.event_mode, "smooth_beta": self.smooth_beta,
                "numeric_embedding": self.numeric_embedding, "use_linear_skip": self.use_linear_skip,
                "parameters": self.parameter_count(), "parameters_total": self.parameter_count(),
                "parameters_trainable": self.trainable_parameter_count(),
                "parameter_groups": self.parameter_groups(),
                "inactive_parameters": inactive_reasons,
                "inactive_parameter_slices": inactive_slices}


class MLPControlV21(_V21Diagnostics, MLPControlV19):
    """Matched tabular control with identical skip, gate and numeric input option."""

    def __init__(self, input_dim: int, output_dim: int, parameter_target: int | None = None,
                 hidden_layers: Literal[1, 2] = 1, head_width: int | None = None,
                 normalization: Literal["none", "layer"] = "layer", dropout: float = 0.0,
                 seed: int = 42,
                 initialization: Literal["native", "linear_anchor", "legacy"] = "native",
                 use_linear_skip: bool = True, residual_gate: Literal["off", "trainable"] = "trainable",
                 numeric_embedding: bool = False) -> None:
        if residual_gate not in ("off", "trainable"):
            raise ValueError("Unknown residual gate")
        if numeric_embedding and input_dim != 72:
            raise ValueError("Numeric embedding requires engineered72")
        adjusted_input = 90 if numeric_embedding else input_dim
        # Explicit width avoids using the v19 72-column count for an optional
        # 90-column nonlinear path. The linear skip remains exactly 72 columns.
        if head_width is None and parameter_target is not None:
            def count(width: int) -> int:
                hidden = adjusted_input * width + width
                if hidden_layers == 2:
                    hidden += width * width + width
                if normalization == "layer":
                    hidden += 2 * width * hidden_layers
                hidden += width * output_dim + output_dim
                if use_linear_skip:
                    hidden += (input_dim + 1) * output_dim
                if numeric_embedding:
                    hidden += len(NUMERIC_COLUMNS) * EMBEDDING_KNOTS * EMBEDDING_WIDTH
                if residual_gate == "trainable":
                    hidden += output_dim
                return hidden
            head_width = min(range(4, 1025), key=lambda width: abs(count(width) - parameter_target))
            parameter_target = None
        super().__init__(input_dim, output_dim, parameter_target=parameter_target,
                         hidden_layers=hidden_layers, head_width=head_width,
                         normalization=normalization, dropout=dropout, seed=seed,
                         initialization=initialization, use_linear_skip=use_linear_skip)
        self.residual_gate = residual_gate
        self.numeric_embedding = numeric_embedding
        self.register_buffer("fit_initialized", torch.tensor(False))
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed + 2101)
            if numeric_embedding:
                self.embedding = NumericPiecewiseEmbedding(input_dim)
                old_first = self.readout[0]
                new_first = nn.Linear(self.embedding.output_dim, self.head_width)
                with torch.no_grad():
                    new_first.weight.normal_(0, .01)
                    new_first.bias.copy_(old_first.bias)
                    remaining = self.embedding.other_columns.tolist()
                    new_first.weight[:, :len(remaining)].copy_(old_first.weight[:, remaining])
                    for j, column in enumerate(NUMERIC_COLUMNS):
                        offset = len(remaining) + EMBEDDING_WIDTH * j
                        new_first.weight[:, offset].copy_(old_first.weight[:, column])
                self.readout[0] = new_first
            else:
                self.embedding = None
            if residual_gate == "trainable":
                self.gate_logits = nn.Parameter(torch.full((output_dim,), math.log(.1 / .9)))
            else:
                self.register_parameter("gate_logits", None)

    def initialize_on_fit(self, x_fit) -> dict:
        if bool(self.fit_initialized):
            raise RuntimeError("Fit initialization already completed")
        x = _fit_tensor(x_fit, self.input_dim, self.readout[0].weight.device)
        info = self.embedding.initialize_on_fit(x) if self.embedding is not None else None
        self.fit_initialized.fill_(True)
        return {"rows": len(x), "normalization": self.normalization, "embedding": info}

    def initialize_native_linear(self, prior_bias=None) -> None:
        if self.initialization != "native":
            raise ValueError("Native linear initialization requires initialization='native'")
        if self.linear_skip is None:
            raise ValueError("linear skip is disabled for this MLP")
        bias = torch.zeros(self.output_dim) if prior_bias is None else prior_bias
        self.set_linear(torch.zeros_like(self.linear_skip.weight), bias)

    def forward_parts(self, x: torch.Tensor, return_diagnostics: bool = False) -> dict:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong MLP input shape")
        nonlinear_x = x if self.embedding is None else self.embedding(x)
        residual_raw = self.readout(nonlinear_x)
        gate = (torch.sigmoid(self.gate_logits) if self.gate_logits is not None
                else residual_raw.new_ones(self.output_dim))
        residual = residual_raw * gate
        linear = (self.linear_skip(x) if self.linear_skip is not None
                  else torch.zeros_like(residual))
        return {"logits": linear + residual, "linear": linear, "residual": residual,
                "residual_raw": residual_raw, "gate": gate,
                "diagnostics": self.component_statistics({"logits": linear + residual,
                    "linear": linear, "residual": residual, "residual_raw": residual_raw,
                    "gate": gate}) if return_diagnostics else None}

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        parts = self.forward_parts(x, return_diagnostics=return_diagnostics)
        return (parts["logits"], parts["diagnostics"]) if return_diagnostics else parts["logits"]

    def parameter_groups(self) -> dict[str, list[str]]:
        groups = {"nonlinear_readout": [], "embedding": [], "residual_gate": [],
                  "linear_skip": [], "inactive": []}
        for name, _ in self.named_parameters():
            if name.startswith("linear_skip."):
                groups["linear_skip"].append(name)
            elif name.startswith("embedding."):
                groups["embedding"].append(name)
            elif name == "gate_logits":
                groups["residual_gate"].append(name)
            else:
                groups["nonlinear_readout"].append(name)
        return groups

    def model_spec(self) -> dict:
        return {"family": "MLP", "input_dim": self.input_dim,
                "nonlinear_input_dim": self.readout[0].in_features,
                "output_dim": self.output_dim, "hidden_layers": self.hidden_layers,
                "head_width": self.head_width, "normalization": self.normalization,
                "initialization": self.initialization, "residual_gate": self.residual_gate,
                "numeric_embedding": self.numeric_embedding,
                "use_linear_skip": self.use_linear_skip,
                "parameters": self.parameter_count(), "parameters_total": self.parameter_count(),
                "parameters_trainable": self.trainable_parameter_count(),
                "parameter_groups": self.parameter_groups()}
