"""
Vortex Feature Extractor.
Aggregates dynamic wave properties and singularity tracking data over T steps
into a structured feature vector suitable for the prediction head.
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.complex_ops import complex_modulus_sq, complex_phase

class VortexFeatureExtractor(nn.Module):
    """
    Extracts physically grounded features from the wave trajectory.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

    def compute_wave_features(self, psi_final: torch.Tensor) -> torch.Tensor:
        """
        Computes static physical properties from the final wave state.
        Args:
            psi_final: (batch, latent_dim, 2)
        Returns:
            features: (batch, 4)
        """
        # 1. Amplitude statistics
        intensity = complex_modulus_sq(psi_final)  # (batch, latent_dim)
        mean_amp = intensity.mean(dim=1)
        var_amp = intensity.var(dim=1, unbiased=False)

        # 2. Phase Coherence Metric
        # If the phase is highly aligned, the vector sum will have length ~ N.
        # If random, it scales as sqrt(N).
        # Convert Phase -> unit vectors -> sum
        # Since psi is already complex, unit vectors are psi / |psi|
        amps = torch.sqrt(intensity + 1e-8)
        unit_psi = psi_final / amps.unsqueeze(-1)  # (batch, latent_dim, 2)
        vector_sum = unit_psi.sum(dim=1)           # (batch, 2)
        coherence_length = torch.norm(vector_sum, dim=1) / self.latent_dim # (batch,)

        # 3. Overall Phase variance (Wrapped)
        phase = complex_phase(psi_final)
        # Using directional statistics for circular variance: 1 - R (where R is mean resultant length)
        # Actually `coherence_length` is exactly R! Circular variance = 1 - R
        circular_variance = 1.0 - coherence_length

        return torch.stack([mean_amp, var_amp, coherence_length, circular_variance], dim=-1)

    def aggregate_singularity_features(self, topo_features_traj: list, kinematics: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Aggregates the singularity properties tracked over time.
        Args:
            topo_features_traj: List of T elements of shape (batch, 4)
                [v_count, av_count, net_charge, top_energy]
            kinematics: (batch, 2) [mean_velocity, peak_velocity]
            batch_size: Explicit batch size to handle T=0 fallback sizes safely.
            device: Explicit device mapping.
        Returns:
            features: (batch, 6)
        """
        if len(topo_features_traj) == 0:
            return torch.zeros(batch_size, 6, device=device)

        traj_tensor = torch.stack(topo_features_traj, dim=1)  # (batch, T, 4)

        # Extract mean properties over time
        mean_topo = traj_tensor.mean(dim=1)  # (batch, 4)

        # Append kinematics (batch, 2)
        return torch.cat([mean_topo, kinematics], dim=-1)

    def forward(self, psi_final: torch.Tensor, topo_traj: list, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi_final: Output state from wave engine
            topo_traj: Tracked topological properties over time
            kinematics: Kinematic properties over time
        Returns:
            combined_features: (batch, 10)
        """
        batch_size = psi_final.size(0)
        device = psi_final.device

        wave_feats = self.compute_wave_features(psi_final)
        sing_feats = self.aggregate_singularity_features(topo_traj, kinematics, batch_size, device)

        # Flatten psi_final to preserve the raw spatial information
        # (batch, latent_dim * 2)
        batch_size = psi_final.size(0)
        psi_flat = psi_final.view(batch_size, -1)

        return torch.cat([psi_flat, wave_feats, sing_feats], dim=-1)
