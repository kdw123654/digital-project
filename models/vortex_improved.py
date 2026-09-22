"""Compact Vortex adaptation with an honest, trainable phase-feature path.

The original Cayley and Gross--Pitaevskii modules are used unchanged.  Integer
plaquette winding remains a diagnostic: its derivative is zero almost
everywhere.  A *separate* smooth circulation/frustration feature is optimized
instead.  That feature is not an integer winding or a topological invariant.

All heads, including the direct input head, are jointly trained from scratch;
there is no fitted/frozen logistic anchor.  Input coordinates must already be
encoded using training-only preprocessing (as in ExpandedPreprocessor).
"""
from __future__ import annotations

import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .native_candidates import VORTEX_SOURCE


def _native_wave_components():
    for path in (VORTEX_SOURCE, VORTEX_SOURCE / "model"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from wave_engine import CayleyUnitaryLinear, GrossPitaevskiiNonlinearity
    return CayleyUnitaryLinear, GrossPitaevskiiNonlinearity


def winding_field(psi: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Original wrapped-loop formula, generalized to a smaller latent grid.

    This matches SingularityDetector.calculate_winding_field.  It is retained
    for interpretation, never misrepresented as a useful smooth gradient path.
    At zero complex amplitude phase and hence winding are not well-defined.
    """
    if psi.shape[-2:] != (grid_h * grid_w, 2):
        raise ValueError("Wave shape must match the requested grid")
    phase = torch.atan2(psi[..., 1], psi[..., 0]).reshape(-1, grid_h, grid_w)
    tl, tr = phase[:, :-1, :-1], phase[:, :-1, 1:]
    br, bl = phase[:, 1:, 1:], phase[:, 1:, :-1]
    wrap = lambda difference: (difference + math.pi) % (2 * math.pi) - math.pi
    return (wrap(tr - tl) + wrap(br - tr) + wrap(bl - br) + wrap(tl - bl)) / (2 * math.pi)


def smooth_plaquette_features(psi: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Smooth local phase geometry, shape [batch, 2*(h-1)*(w-1)].

    Directed edge sine circulation and mean edge frustration (1-cos delta)
    are evaluated through normalized complex products, avoiding atan2 branch
    cuts.  Unlike the wrapped angle sum, sine terms do not telescope.  The
    epsilon regularization makes outputs finite at zero amplitude.  This is a
    periodic differentiable proxy, not the topological charge itself.
    """
    if psi.shape[-2:] != (grid_h * grid_w, 2):
        raise ValueError("Wave shape must match the requested grid")
    unit = psi / torch.sqrt(psi.square().sum(-1, keepdim=True) + 1e-6)
    grid = unit.reshape(-1, grid_h, grid_w, 2)
    corners = (grid[:, :-1, :-1], grid[:, :-1, 1:], grid[:, 1:, 1:], grid[:, 1:, :-1])
    sine, cosine = [], []
    for index, a in enumerate(corners):
        b = corners[(index + 1) % 4]
        cosine.append((a * b).sum(-1))
        sine.append(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
    circulation = torch.stack(sine).mean(0)
    frustration = 1 - torch.stack(cosine).mean(0)
    return torch.cat((circulation.flatten(1), frustration.flatten(1)), -1)


class ImprovedVortex(nn.Module):
    """Complex waves, shared unitary dynamics, nonlinear phase and smooth proxy.

    Smaller latent and encoder widths address variance at the study's sample
    size.  A direct head preserves encoded magnitudes; wave and phase heads
    receive separately normalized features.  Dropout acts on readout features,
    never on the unitary dynamics.  Native winding is diagnostic-only.
    """
    def __init__(self, input_dim: int = 72, latent_dim: int = 32,
                 steps: int = 4, dropout: float = .1):
        super().__init__()
        if input_dim < 1 or latent_dim < 8 or latent_dim % 4 or steps < 1:
            raise ValueError("Positive input/steps and latent dimension divisible by 4 (>=8) required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim, self.latent_dim, self.steps = input_dim, latent_dim, steps
        self.grid_h, self.grid_w = 4, latent_dim // 4
        hidden = min(32, latent_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.amplitude = nn.Linear(hidden, latent_dim)
        self.phase = nn.Linear(hidden, latent_dim)
        Cayley, GP = _native_wave_components()
        self.unitary_evolution = Cayley(latent_dim)
        self.non_linear_gp = GP(latent_dim)
        # Positive bounded steps and positive exponential damping avoid unstable
        # negative or excessively large time/damping factors.
        self.dt_raw = nn.Parameter(torch.full((steps,), math.log(.2 / .8)))
        self.damping_raw = nn.Parameter(torch.tensor(-6.))
        wave_dim = 3 * latent_dim
        proxy_dim = 2 * (self.grid_h - 1) * (self.grid_w - 1)
        self.wave_norm = nn.LayerNorm(wave_dim)
        self.proxy_norm = nn.LayerNorm(2 * proxy_dim)
        self.dropout = nn.Dropout(dropout)
        self.raw_head = nn.Linear(input_dim, 4)
        self.wave_head = nn.Linear(wave_dim, 4, bias=False)
        self.topology_head = nn.Linear(2 * proxy_dim, 4, bias=False)
        # Nonzero small heads transmit gradients on the first update while
        # avoiding large arbitrary initial corrections to the direct branch.
        nn.init.normal_(self.wave_head.weight, std=.01)
        nn.init.normal_(self.topology_head.weight, std=.01)

    @property
    def time_steps(self) -> torch.Tensor:
        return .02 + .48 * torch.sigmoid(self.dt_raw)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        amplitude = F.softplus(self.amplitude(h)) + .05
        phase = math.pi * torch.tanh(self.phase(h))
        return torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), -1)

    def feature_views(self, x: torch.Tensor, include_diagnostics: bool = False):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Expected a batch of encoded input rows")
        psi = self.encode(x)
        initial_proxy = smooth_plaquette_features(psi, self.grid_h, self.grid_w)
        winding_trajectory = []
        # Shared Hamiltonian: calculate its exact native Cayley matrix once per
        # forward pass, then reuse for all integration steps.  No detach/cache
        # crosses forward passes, so every step still learns the Hamiltonian.
        unitary = self.unitary_evolution.get_unitary()
        damping = F.softplus(self.damping_raw)
        for dt in self.time_steps:
            complex_wave = torch.view_as_complex(psi.contiguous()) @ unitary.T
            psi = self.non_linear_gp(torch.view_as_real(complex_wave), dt)
            psi = psi * torch.exp(-damping * dt)
            if include_diagnostics:
                winding_trajectory.append(winding_field(psi.detach(), self.grid_h, self.grid_w))
        final_proxy = smooth_plaquette_features(psi, self.grid_h, self.grid_w)
        proxy = torch.cat((final_proxy, final_proxy - initial_proxy), -1)
        intensity = psi.square().sum(-1)
        wave = torch.cat((psi.flatten(1), torch.log1p(intensity)), -1)
        result = {"raw": x, "wave": self.wave_norm(wave), "proxy": self.proxy_norm(proxy)}
        if include_diagnostics:
            result.update(wave_state=psi, winding_trajectory=torch.stack(winding_trajectory, 1),
                          smooth_proxy=final_proxy)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.feature_views(x)
        return (self.raw_head(v["raw"]) + self.wave_head(self.dropout(v["wave"]))
                + self.topology_head(self.dropout(v["proxy"])))

    def diagnostics(self, x: torch.Tensor) -> dict:
        """Tensor diagnostics; gradients remain available on the smooth proxy."""
        return self.feature_views(x, include_diagnostics=True)
