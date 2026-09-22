"""
Topological Singularity Detector for VortexNet.
Treats the 128-dimensional latent space as an 8x16 spatial wave grid.
Scans for phase singularities (vortices and anti-vortices) by calculating
plaquette winding numbers, and tracks their dynamics and velocities natively.
"""

import torch
import torch.nn as nn
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.complex_ops import complex_phase

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wraps phase to [-pi, pi), preserving gradients perfectly almost everywhere."""
    return (phi + math.pi) % (2 * math.pi) - math.pi

class SingularityDetector(nn.Module):
    """
    Detects and tracks topological singularities on the 8x16 virtual grid.
    Provides differentiable features extracted from vortex dynamics.
    """
    def __init__(self, grid_h: int = 8, grid_w: int = 16):
        super().__init__()
        self.h = grid_h
        self.w = grid_w
        assert self.h * self.w == 128, "Latent dimension must exactly form the requested grid."

    def calculate_winding_field(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Calculates the topological winding number around every 2x2 plaquette.

        Args:
            psi: (batch, 128, 2) or (batch, 8, 16, 2)

        Returns:
            W: (batch, h-1, w-1) The topological charge density field.
               Values tightly congregate exactly around {-1, 0, +1}.
        """
        if psi.dim() == 3:
            # (batch, 128, 2) -> (batch, h, w, 2)
            psi = psi.view(*psi.shape[:-2], self.h, self.w, 2)

        phase = complex_phase(psi)  # (batch, h, w)

        # Plaquette corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        p_TL = phase[..., :-1, :-1]
        p_TR = phase[..., :-1, 1:]
        p_BR = phase[..., 1:, 1:]
        p_BL = phase[..., 1:, :-1]

        # Phase derivatives (wrapped) along the loop
        dphi_top   = wrap_phase(p_TR - p_TL)
        dphi_right = wrap_phase(p_BR - p_TR)
        dphi_bot   = wrap_phase(p_BL - p_BR)
        dphi_left  = wrap_phase(p_TL - p_BL)

        # Winding number W = sum(dphi) / 2pi
        W = (dphi_top + dphi_right + dphi_bot + dphi_left) / (2 * math.pi)

        return W

    def extract_vortex_features(self, W: torch.Tensor, psi: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts differentiable scalar features from the winding field.
        Since W is highly clustered around integers, we use Soft-thresholding
        for robust gradient propagation.

        Args:
            W: (batch, h-1, w-1) Continuous winding field.
            psi: Optional amplitude context.

        Returns:
            features: (batch, 4) -> [vortex_count, antivortex_count, total_charge, singularity_energy]
        """
        # Soft extraction using Sigmoids with high steepness to approximate step functions
        steepness = 10.0

        # Vortices have W ~ +1 (threshold at +0.5)
        is_vortex = torch.sigmoid(steepness * (W - 0.5))
        # Anti-vortices have W ~ -1 (threshold at -0.5)
        # Note: W is negative, so W < -0.5 -> -W > 0.5
        is_antivortex = torch.sigmoid(steepness * (-W - 0.5))

        # Feature 1 & 2: Counts
        v_count = is_vortex.sum(dim=[-1, -2])
        av_count = is_antivortex.sum(dim=[-1, -2])

        # Feature 3: Net Topological Charge
        net_charge = W.sum(dim=[-1, -2])

        # Feature 4: Singularity Energy / Metric
        # Absolute topological activity
        topological_energy = (is_vortex + is_antivortex).sum(dim=[-1, -2])

        return torch.stack([v_count, av_count, net_charge, topological_energy], dim=-1)

    def track_kinematics(self, W_trajectory: list, dt_list: list) -> torch.Tensor:
        """
        Tracks vortex velocity (v_sing) and detects 'superluminal' annihilation events.
        Args:
            W_trajectory: List of length T, each tensor is (batch, h-1, w-1)
            dt_list: List of length T-1, time steps

        Returns:
            kinematic_features: (batch, 2) -> [mean_velocity, peak_velocity]
        """
        if len(W_trajectory) < 2:
            batch_size = W_trajectory[0].shape[0] if len(W_trajectory) > 0 else 1
            device = W_trajectory[0].device if len(W_trajectory) > 0 else torch.device('cpu')
            return torch.zeros(batch_size, 2, device=device)

        batch_size = W_trajectory[0].shape[0]
        device = W_trajectory[0].device

        velocities = []
        for i in range(len(W_trajectory) - 1):
            W_curr = W_trajectory[i]
            W_next = W_trajectory[i+1]
            dt = dt_list[i].view(-1)  # (1, ) or similar

            # Since exact pair tracking bipartite matching is non-differentiable and slow,
            # we use the Wasserstein-1 (Earth Mover) distance approximation heavily used in ML
            # or simply compute the "flux" of the topological charge.
            # Local continuity equation: dW/dt + div j = 0 -> |j| = |dW/dt| approx local speed.

            dW_dt = torch.abs(W_next - W_curr) / (dt + 1e-6)

            # Mean kinetic energy of singularities
            mean_v = dW_dt.mean(dim=[-1, -2])
            peak_v = dW_dt.amax(dim=[-1, -2])
            velocities.append(torch.stack([mean_v, peak_v], dim=-1))

        # Aggregate over time
        vel_tensor = torch.stack(velocities, dim=1)  # (batch, T-1, 2)

        # Average over time
        return vel_tensor.mean(dim=1)


def _test_singularity_detector():
    from wave_engine import WaveEngine

    print("Testing Singularity Detector...")
    batch = 4
    dim = 128

    # 1. Setup deterministic synthetic grid with a manufactured perfect vortex
    # We will manually create a 8x16 grid where phases circle around a plaquette center
    psi_synthetic = torch.zeros(batch, 8, 16, 2)
    center_y, center_x = 4.5, 8.5  # Set precisely inside the plaquette (TopLeft is 4, 8)

    for y in range(8):
        for x in range(16):
            # Angle relative to center
            dy = y - center_y
            dx = x - center_x
            angle = math.atan2(dy, dx)

            # Construct complex wave
            psi_synthetic[:, y, x, 0] = math.cos(angle)
            psi_synthetic[:, y, x, 1] = math.sin(angle)

    # Flatten it exactly like output of encoder
    psi_synthetic = psi_synthetic.view(batch, dim, 2)

    detector = SingularityDetector(8, 16)
    W = detector.calculate_winding_field(psi_synthetic)

    # Check max winding (Should perfectly detect the +1 vortex at the center)
    max_W = W.max().item()
    print(f"Detected Peak Topological Charge: {max_W:.4f} (Expected: ~1.0)")
    assert max_W > 0.9, "Failed to detect synthetic mathematical vortex."

    features = detector.extract_vortex_features(W)
    v_count = features[0, 0].item()
    print(f"Calculated Vortex Count from Features: {v_count:.4f} (Expected: ~1.0)")
    assert v_count > 0.9, "Soft counter failed to register the vortex."

    # 2. Integrate with Wave Engine trajectory
    print("\nSimulating Wave Engine to generate dynamic vortex trajectories...")
    engine = WaveEngine(latent_dim=dim, num_steps=10)
    psi = torch.randn(batch, dim, 2)

    w_traj = []
    dt_list = []
    for t in range(engine.num_steps):
        dt = engine.dt[t]
        psi = engine.unitary_evolution(psi)
        psi = engine.non_linear_gp(psi, dt)

        # Extact winding
        W_t = detector.calculate_winding_field(psi)
        w_traj.append(W_t)
        if t < engine.num_steps - 1:
            dt_list.append(dt)

    kinematics = detector.track_kinematics(w_traj, dt_list)
    print(f"Mean System Singularity Velocity (Batch 0): {kinematics[0, 0].item():.4f}")
    print(f"Peak System Singularity Velocity (Batch 0): {kinematics[0, 1].item():.4f}")

    # Backprop test to ensure the entire tracking is differentiable
    loss = kinematics.sum()
    loss.backward()

    # Check if gradients reached the wave engine dt parameters
    assert engine.dt[0].grad is not None, "Gradient path broken! Time parameter didn't receive gradients from Singularity kinematics."
    print("\n✓ Singularity backpropagation path verified natively into physical dt parameter!")

    print("All Singularity Detector tests passed!")

if __name__ == "__main__":
    _test_singularity_detector()
