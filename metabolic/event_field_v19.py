"""Single event-plastic field with cached dynamics and fit-only initialization.

The recurrent state is exactly (complex field z, event trace a, antisymmetric
plastic coupling P). No independent expert or nonlinear raw-input bypass is
present. The optional trainable linear skip is an explicit experimental factor.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .event_plastic_field import EventPlasticField, SurrogateEvent


def _fit_tensor(x: torch.Tensor, input_dim: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(x, dtype=torch.float32, device=device)
    if value.ndim != 2 or value.shape[1] != input_dim or len(value) == 0:
        raise ValueError(f"Expected nonempty fit input with {input_dim} columns")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Fit input must be finite")
    return value


class EventPlasticFieldV19(EventPlasticField):
    """The v17 field equation, with shared per-forward terms calculated once.

    ``initialize_on_fit`` accepts clean fit-role inputs only and never labels.
    It calibrates the existing threshold parameter to a target-free event rate
    and, for ``feature_norm='fit'``, freezes state-feature mean and scale. Both
    native and linear-anchor starts run the same calibration and use the same
    small nonzero field readout. Only the linear skip initializer differs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dimension: int = 16,
        steps: int = 6,
        head_width: int = 64,
        dropout: float = 0.0,
        feature_norm: Literal["fit", "none"] = "fit",
        seed: int = 42,
        initialization: Literal["native", "linear_anchor", "legacy"] = "native",
        use_linear_skip: bool = True,
        activation_target: float | None = 0.15,
        plasticity: bool = True,
        quartic: bool = True,
    ) -> None:
        if input_dim < 1 or output_dim < 1 or dimension < 2 or steps < 1 or head_width < 1:
            raise ValueError("All model dimensions and steps must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if feature_norm not in ("fit", "none"):
            raise ValueError("feature_norm must be 'fit' or 'none'")
        if initialization not in ("native", "linear_anchor", "legacy"):
            raise ValueError("Unknown initialization")
        if initialization == "linear_anchor" and not use_linear_skip:
            raise ValueError("linear_anchor requires use_linear_skip=True")
        if activation_target is not None and not 0.0 < activation_target < 1.0:
            raise ValueError("activation_target must be in (0, 1) or None")
        # The parent constructor sets a seed globally. Restore the caller's RNG
        # while preserving its historical parameter draw order for parity tests.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            super().__init__(input_dim, output_dim, dimension=dimension, steps=steps,
                             plasticity=plasticity, quartic=quartic, seed=seed,
                             head_width=head_width)
            if dropout:
                self.readout = nn.Sequential(nn.Linear(5 * dimension, head_width),
                                             nn.SiLU(), nn.Dropout(dropout),
                                             nn.Linear(head_width, output_dim))
            if initialization == "legacy":
                nn.init.zeros_(self.readout[-1].weight)
                nn.init.zeros_(self.readout[-1].bias)
            else:
                nn.init.normal_(self.readout[-1].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.readout[-1].bias)
        self.dropout = dropout
        self.feature_norm = feature_norm
        self.initialization = initialization
        self.use_linear_skip = use_linear_skip
        self.activation_target = activation_target
        if not use_linear_skip:
            self.linear_skip = None
        self.register_buffer("feature_mean", torch.zeros(5 * dimension))
        self.register_buffer("feature_scale", torch.ones(5 * dimension))
        self.register_buffer("fit_initialized", torch.tensor(False))
        self.register_buffer("activation_fit_rate", torch.tensor(float("nan")))

    def set_linear(self, coef, bias) -> None:
        if self.linear_skip is None:
            raise ValueError("linear skip is disabled for this field")
        weight = torch.as_tensor(coef, dtype=self.linear_skip.weight.dtype,
                                 device=self.linear_skip.weight.device)
        offset = torch.as_tensor(bias, dtype=self.linear_skip.bias.dtype,
                                 device=self.linear_skip.bias.device)
        if weight.shape != self.linear_skip.weight.shape or offset.shape != self.linear_skip.bias.shape:
            raise ValueError("Linear coefficient or bias shape mismatch")
        if not bool(torch.isfinite(weight).all() and torch.isfinite(offset).all()):
            raise ValueError("Linear initializer must be finite")
        with torch.no_grad():
            self.linear_skip.weight.copy_(weight)
            self.linear_skip.bias.copy_(offset)

    @staticmethod
    def _activation_probe_indices(rows: int, device: torch.device) -> torch.Tensor:
        # The saved cohort is ordered by survey year. Span the whole fit role
        # rather than taking its first 512 rows from the earliest year.
        return torch.linspace(0, rows - 1, min(rows, 512), device="cpu").round().long().to(device)

    def _field_pass(self, x: torch.Tensor, diagnostics: Literal["none", "rate", "full"] = "none"):
        """Return raw state features; aggregate diagnostics only when needed."""
        batch = len(x)
        state = self.initial_state(batch, x.device)
        z, trace, plastic = state
        d = self.dimension
        dt = F.softplus(self.dt_raw).clamp(.025, .5)
        eye = torch.eye(d, dtype=torch.complex64, device=x.device)
        base_real = (self.h_real + self.h_real.T) / 2
        base_imag = (self.h_imag - self.h_imag.T) / 2
        static_h = torch.complex(base_real, base_imag)
        base_unitary = torch.linalg.solve(eye + 1j * dt * static_h,
                                          eye - 1j * dt * static_h)
        # x is fixed across all recurrent transitions. This exact expression
        # replaces steps repeated evaluations of the same linear projection.
        encoded = .1 * self.drive(x).reshape(-1, d, 2)
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
                    solved, info = torch.linalg.solve_ex(denominator, numerator,
                                                         check_errors=False)
                    solve_infos.append(info)
                else:
                    solved = torch.linalg.solve(denominator, numerator)
                propagated = solved.squeeze(-1)
            driven = decay * propagated + drive
            energy = driven.abs().square()
            evolved = driven * torch.exp(-1j * dt * coefficient * energy)
            threshold = threshold_base * (1 + homeostasis * trace)
            event = SurrogateEvent.apply((energy - threshold) / (threshold.detach() + .1))
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
        features = torch.cat([z.real, z.imag, z.abs().square(), trace, energy], 1)
        if diagnostics == "none":
            return features, None
        if diagnostics == "rate":
            return features, {"event_rate": event_total / (self.steps * batch * d)}
        return features, {
            "event_rate": torch.stack(events).mean(),
            "field_energy": torch.stack(energies).mean(),
            "plastic_norm": plastic.norm(dim=(1, 2)).mean(),
            "field_features": features,
        }

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong field input shape")
        features, diagnostics = self._field_pass(x, "full" if return_diagnostics else "none")
        # Before fit initialization these buffers are the identity transform;
        # avoid a per-batch device sync just to inspect the fitted flag.
        if self.feature_norm == "fit":
            features = (features - self.feature_mean) / self.feature_scale
        output = self.readout(features)
        if self.linear_skip is not None:
            output = self.linear_skip(x) + output
        if return_diagnostics:
            return output, diagnostics
        return output

    @torch.no_grad()
    def initialize_on_fit(self, x_fit, batch_size: int = 256) -> dict:
        """Calibrate only from clean fit-role inputs, before optimizer creation.

        The caller owns role selection. Repeated calls are rejected so later
        validation/test data cannot silently replace fitted statistics.
        """
        if bool(self.fit_initialized):
            raise RuntimeError("Fit initialization already completed")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = _fit_tensor(x_fit, self.input_dim, self.drive.weight.device)
        was_training = self.training
        self.eval()
        try:
            threshold_scale = 1.0
            if self.activation_target is not None:
                # A deterministic fit subset keeps the same initializer across
                # augmentation arms and bounds setup cost for repeated trials.
                probe = x.index_select(0, self._activation_probe_indices(len(x), x.device))
                original = F.softplus(self.threshold_raw).detach().clone()

                def rate(scale: float) -> float:
                    calibrated = (original * scale).clamp_min(1e-7)
                    self.threshold_raw.copy_(torch.log(torch.expm1(calibrated)))
                    _, info = self._field_pass(probe, "rate")
                    return float(info["event_rate"])

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
                probe = x.index_select(0, self._activation_probe_indices(len(x), x.device))
                _, info = self._field_pass(probe, "rate")
                fit_rate = float(info["event_rate"])
            self.activation_fit_rate.fill_(fit_rate)
            if self.feature_norm == "fit":
                blocks = [self._field_pass(chunk, "none")[0].detach().to("cpu", dtype=torch.float64)
                          for chunk in x.split(batch_size)]
                all_features = torch.cat(blocks)
                mean = all_features.mean(0)
                scale = all_features.std(0, unbiased=False).clamp_min(.02)
                self.feature_mean.copy_(mean.to(self.feature_mean.device, dtype=self.feature_mean.dtype))
                self.feature_scale.copy_(scale.to(self.feature_scale.device, dtype=self.feature_scale.dtype))
            self.fit_initialized.fill_(True)
            return {
                "rows": len(x),
                "activation_target": self.activation_target,
                "activation_fit_rate": fit_rate,
                "threshold_scale": threshold_scale,
                "feature_norm": self.feature_norm,
                "feature_scale_min": float(self.feature_scale.min()),
                "feature_scale_median": float(self.feature_scale.median()),
            }
        finally:
            self.train(was_training)


class MLPControlV19(nn.Module):
    """Separate tabular MLP control; never called by EventPlasticFieldV19."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        parameter_target: int | None = None,
        hidden_layers: Literal[1, 2] = 1,
        head_width: int | None = None,
        normalization: Literal["none", "layer"] = "layer",
        dropout: float = 0.0,
        seed: int = 42,
        initialization: Literal["native", "linear_anchor", "legacy"] = "native",
        use_linear_skip: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or hidden_layers not in (1, 2):
            raise ValueError("Invalid MLP dimensions or hidden_layers")
        if normalization not in ("none", "layer") or not 0 <= dropout < 1:
            raise ValueError("Invalid MLP normalization or dropout")
        if initialization not in ("native", "linear_anchor", "legacy"):
            raise ValueError("Unknown initialization")
        if initialization == "linear_anchor" and not use_linear_skip:
            raise ValueError("linear_anchor requires use_linear_skip=True")
        if head_width is not None and (head_width < 1 or parameter_target is not None):
            raise ValueError("Choose positive head_width or parameter_target, not both")
        if parameter_target is not None and parameter_target < 1:
            raise ValueError("parameter_target must be positive")

        def count(width: int) -> int:
            hidden = input_dim * width + width
            if hidden_layers == 2:
                hidden += width * width + width
            if normalization == "layer":
                hidden += 2 * width * hidden_layers
            hidden += width * output_dim + output_dim
            if use_linear_skip:
                hidden += (input_dim + 1) * output_dim
            return hidden

        if head_width is None:
            head_width = (min(range(4, 1025), key=lambda width: abs(count(width) - parameter_target))
                          if parameter_target is not None else 64)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_width = head_width
        self.hidden_layers = hidden_layers
        self.normalization = normalization
        self.initialization = initialization
        self.use_linear_skip = use_linear_skip
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed)
            layers: list[nn.Module] = []
            previous = input_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(previous, head_width))
                if normalization == "layer":
                    layers.append(nn.LayerNorm(head_width))
                layers.append(nn.SiLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                previous = head_width
            layers.append(nn.Linear(head_width, output_dim))
            self.readout = nn.Sequential(*layers)
            self.linear_skip = nn.Linear(input_dim, output_dim) if use_linear_skip else None
            if initialization == "legacy":
                nn.init.zeros_(self.readout[-1].weight)
            else:
                nn.init.normal_(self.readout[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.readout[-1].bias)

    def initialize_on_fit(self, x_fit) -> dict:
        # LayerNorm carries trainable parameters and uses no fit statistics.
        x = _fit_tensor(x_fit, self.input_dim, self.readout[0].weight.device)
        return {"rows": len(x), "normalization": self.normalization}

    def set_linear(self, coef, bias) -> None:
        if self.linear_skip is None:
            raise ValueError("linear skip is disabled for this MLP")
        weight = torch.as_tensor(coef, dtype=self.linear_skip.weight.dtype,
                                 device=self.linear_skip.weight.device)
        offset = torch.as_tensor(bias, dtype=self.linear_skip.bias.dtype,
                                 device=self.linear_skip.bias.device)
        if weight.shape != self.linear_skip.weight.shape or offset.shape != self.linear_skip.bias.shape:
            raise ValueError("Linear coefficient or bias shape mismatch")
        if not bool(torch.isfinite(weight).all() and torch.isfinite(offset).all()):
            raise ValueError("Linear initializer must be finite")
        with torch.no_grad():
            self.linear_skip.weight.copy_(weight)
            self.linear_skip.bias.copy_(offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Wrong MLP input shape")
        correction = self.readout(x)
        return correction if self.linear_skip is None else self.linear_skip(x) + correction
