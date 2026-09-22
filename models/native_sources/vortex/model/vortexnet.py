"""
VortexNet: The End-to-End Molecular Property Prediction Architecture.
Integrates Complex Encoding, Physics-based Wave Simulation,
Topological Singularity Detection, and Property Prediction.
"""

import torch
import torch.nn as nn
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))
from complex_encoder import ComplexEncoder
from wave_engine import WaveEngine
from singularity_detector import SingularityDetector
from vortex_features import VortexFeatureExtractor
from prediction_head import PredictionHead

class VortexNet(nn.Module):
    def __init__(self,
                 input_dim: int = 217,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 num_steps: int = 32,
                 num_tasks: int = 1,
                 is_classification: bool = False,
                 superluminal_reg: bool = True):
        """
        VortexNet Main Architecture.
        """
        super().__init__()
        assert latent_dim == 128, "Currently constrained to 128 for 8x16 spatial projection."

        self.encoder = ComplexEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.wave_engine = WaveEngine(latent_dim=latent_dim, num_steps=num_steps)
        self.singularity_detector = SingularityDetector(grid_h=8, grid_w=16)
        self.feature_extractor = VortexFeatureExtractor(latent_dim=latent_dim)

        self.superluminal_reg = superluminal_reg
        self.c_sys = 10.0  # Speed limit threshold for topological jumps

        # Calculate feature dim for the prediction head
        # 128*2 = 256 (raw psi vector) + 4 (wave physics) + 6 (singularity features)
        extracted_dim = (latent_dim * 2) + 4 + 6
        self.prediction_head = PredictionHead(in_features=extracted_dim,
                                              hidden_dim=hidden_dim,
                                              num_tasks=num_tasks,
                                              is_classification=is_classification)

    def topological_dropout(self, psi: torch.Tensor, W_trajectory: list, dt_list: list) -> torch.Tensor:
        """
        Implements Superluminal Regularization:
        When singularity velocities exceed the system threshold (annihilation/creation events),
        we induce stochastic coordinate jumps (Topological Dropout) to prevent overfitting.
        """
        if not self.training or len(W_trajectory) < 2:
            return psi

        # Get instant velocity
        W_curr = W_trajectory[-2]
        W_next = W_trajectory[-1]
        dt = dt_list[-1]

        # (batch, h-1, w-1)
        velocity_field = torch.abs(W_next - W_curr) / (dt + 1e-6)

        # Find batches where peak velocity > threshold
        peak_v = velocity_field.amax(dim=[-1, -2])  # (batch,)

        # Boolean mask of superluminal batches
        superluminal_mask = (peak_v > self.c_sys)

        if superluminal_mask.any():
            # Apply topological dropout only to those batches
            # Randomly shuffle or re-scale a small portion of dimensions
            noise = torch.randn_like(psi) * 0.1
            jump_mask = (torch.rand_like(psi[..., 0]) < 0.1).unsqueeze(-1)

            # Apply jump where mask is true AND batch is superluminal
            batch_mask_expanded = superluminal_mask.view(-1, 1, 1).expand_as(psi)
            final_mask = jump_mask & batch_mask_expanded

            psi = torch.where(final_mask, psi + noise, psi)

        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 217) RDKit features
        Returns:
            predictions: (batch, num_tasks)
        """
        # 1. Map to Complex Wave Domain
        psi = self.encoder(x)

        # Tracking vars
        w_traj = []
        dt_list = []
        topo_features = []

        # 2. Integrate Physics / Time Evolution
        for t in range(self.wave_engine.num_steps):
            dt = self.wave_engine.dt[t]

            # Step Wave Engine
            psi = self.wave_engine.unitary_evolution(psi)
            psi = self.wave_engine.non_linear_gp(psi, dt)
            gamma = torch.clamp(self.wave_engine.damping, min=0.0, max=0.5)
            psi = psi * (1.0 - gamma * dt)

            # Extract topological field
            W_t = self.singularity_detector.calculate_winding_field(psi)
            w_traj.append(W_t)

            # Store dt for kinematics later
            if t < self.wave_engine.num_steps - 1:
                dt_list.append(dt)

            # Extract topological properties
            feats_t = self.singularity_detector.extract_vortex_features(W_t)
            topo_features.append(feats_t)

            # Superluminal Regularization
            if self.superluminal_reg and t > 0:
                psi = self.topological_dropout(psi, w_traj, dt_list)

        # 3. Kinematic Tracking
        kinematics = self.singularity_detector.track_kinematics(w_traj, dt_list)

        # 4. Feature Extraction & Aggregation
        extracted = self.feature_extractor(psi, topo_features, kinematics)

        # 5. Predict
        out = self.prediction_head(extracted)

        return out


def _test_vortexnet():
    print("Testing VortexNet End-to-End...")

    batch = 16
    input_dim = 217
    x = torch.randn(batch, input_dim)

    model = VortexNet(input_dim=input_dim, num_steps=5, superluminal_reg=True)

    # 1. Forward Pass
    preds = model(x)
    assert preds.shape == (batch, 1), f"Wrong output shape: {preds.shape}"
    print(f"Forward Pass OK. Output Shape: {preds.shape}")

    # 2. Gradient Flow Check
    target = torch.randn(batch, 1)
    loss = torch.nn.functional.mse_loss(preds, target)
    loss.backward()

    grad_checks = {
        'Encoder': model.encoder.shared[0].weight.grad is not None,
        'WaveEngine dt': model.wave_engine.dt[0].grad is not None,
        'WaveEngine GP': model.wave_engine.non_linear_gp.alpha.grad is not None,
        'PredictionHead': model.prediction_head.net[0].weight.grad is not None
    }

    for name, ok in grad_checks.items():
        print(f"  Gradient flow into {name}: {'OK' if ok else 'FAILED'}")
        assert ok, f"Missing gradients for {name}"

    print("End-to-End Pipeline successfully tested!")


class VortexNetV2(nn.Module):
    """
    VortexNet v2: Atom-level Wave Superposition Architecture.

    Key difference from v1:
        v1: RDKit 217d global descriptors → ComplexEncoder → WaveEngine → ...
        v2: Per-atom 42d features → AtomWaveEncoder (attention + superposition) → WaveEngine → ...

    The physics engine pipeline (WaveEngine, SingularityDetector, VortexFeatures,
    PredictionHead) is 100% identical to v1.
    """
    def __init__(self,
                 atom_dim: int = 42,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 num_steps: int = 32,
                 num_tasks: int = 1,
                 is_classification: bool = False,
                 superluminal_reg: bool = True,
                 num_heads: int = 4):
        super().__init__()
        assert latent_dim == 128, "Currently constrained to 128 for 8x16 spatial projection."

        from atom_wave_encoder import AtomWaveEncoder

        self.encoder = AtomWaveEncoder(
            atom_dim=atom_dim, latent_dim=latent_dim,
            hidden_dim=hidden_dim, num_heads=num_heads
        )
        self.wave_engine = WaveEngine(latent_dim=latent_dim, num_steps=num_steps)
        self.singularity_detector = SingularityDetector(grid_h=8, grid_w=16)
        self.feature_extractor = VortexFeatureExtractor(latent_dim=latent_dim)

        self.superluminal_reg = superluminal_reg
        self.c_sys = 10.0

        extracted_dim = (latent_dim * 2) + 4 + 6
        self.prediction_head = PredictionHead(
            in_features=extracted_dim, hidden_dim=hidden_dim,
            num_tasks=num_tasks, is_classification=is_classification
        )

    def topological_dropout(self, psi, W_trajectory, dt_list):
        """Same topological dropout as v1."""
        if not self.training or len(W_trajectory) < 2:
            return psi
        W_curr = W_trajectory[-2]
        W_next = W_trajectory[-1]
        dt = dt_list[-1]
        velocity_field = torch.abs(W_next - W_curr) / (dt + 1e-6)
        peak_v = velocity_field.amax(dim=[-1, -2])
        superluminal_mask = (peak_v > self.c_sys)
        if superluminal_mask.any():
            noise = torch.randn_like(psi) * 0.1
            jump_mask = (torch.rand_like(psi[..., 0]) < 0.1).unsqueeze(-1)
            batch_mask_expanded = superluminal_mask.view(-1, 1, 1).expand_as(psi)
            final_mask = jump_mask & batch_mask_expanded
            psi = torch.where(final_mask, psi + noise, psi)
        return psi

    def forward(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_features: (batch, max_atoms, atom_dim) padded atom features
            atom_mask: (batch, max_atoms) binary mask
        Returns:
            predictions: (batch, num_tasks)
        """
        # 1. Atom-level Wave Encoding + Superposition
        psi = self.encoder(atom_features, atom_mask)

        # 2-5: Identical physics pipeline as v1
        w_traj = []
        dt_list = []
        topo_features = []

        for t in range(self.wave_engine.num_steps):
            dt = self.wave_engine.dt[t]
            psi = self.wave_engine.unitary_evolution(psi)
            psi = self.wave_engine.non_linear_gp(psi, dt)
            gamma = torch.clamp(self.wave_engine.damping, min=0.0, max=0.5)
            psi = psi * (1.0 - gamma * dt)

            W_t = self.singularity_detector.calculate_winding_field(psi)
            w_traj.append(W_t)

            if t < self.wave_engine.num_steps - 1:
                dt_list.append(dt)

            feats_t = self.singularity_detector.extract_vortex_features(W_t)
            topo_features.append(feats_t)

            if self.superluminal_reg and t > 0:
                psi = self.topological_dropout(psi, w_traj, dt_list)

        kinematics = self.singularity_detector.track_kinematics(w_traj, dt_list)
        extracted = self.feature_extractor(psi, topo_features, kinematics)
        out = self.prediction_head(extracted)

        return out


def _test_vortexnet_v2():
    print("Testing VortexNetV2 End-to-End...")

    batch = 8
    max_atoms = 80
    atom_dim = 42

    atom_feats = torch.randn(batch, max_atoms, atom_dim)
    atom_mask = torch.zeros(batch, max_atoms)
    for i in range(batch):
        n = torch.randint(3, 30, (1,)).item()
        atom_mask[i, :n] = 1.0
        atom_feats[i, n:] = 0.0

    model = VortexNetV2(atom_dim=atom_dim, num_steps=5, superluminal_reg=True)

    preds = model(atom_feats, atom_mask)
    assert preds.shape == (batch, 1), f"Wrong output shape: {preds.shape}"
    print(f"  Forward Pass OK. Output Shape: {preds.shape}")

    target = torch.randn(batch, 1)
    loss = torch.nn.functional.mse_loss(preds, target)
    loss.backward()

    grad_checks = {
        'AtomWaveEncoder MLP': model.encoder.atom_mlp[0].weight.grad is not None,
        'AtomWaveEncoder Amplitude': model.encoder.amplitude_head[0].weight.grad is not None,
        'WaveEngine dt': model.wave_engine.dt[0].grad is not None,
        'WaveEngine GP': model.wave_engine.non_linear_gp.alpha.grad is not None,
        'PredictionHead': model.prediction_head.net[0].weight.grad is not None,
    }

    for name, ok in grad_checks.items():
        print(f"  Gradient flow into {name}: {'OK' if ok else 'FAILED'}")
        assert ok, f"Missing gradients for {name}"

    print("VortexNetV2 End-to-End Pipeline successfully tested!")


class VortexNetV3(nn.Module):
    """
    VortexNet V3: Open Quantum System Architecture.

    Key changes from V2:
      1. OpenWaveEngine: Cayley + learnable amplitude modulation (Lindblad-inspired)
      2. DualPredictionHead: Separate psi and physics branches with gating
      3. Residual skip from encoder to prediction head
      4. Norm regularization loss to prevent wave explosion

    The physics pipeline (SingularityDetector, VortexFeatures) is identical.
    """
    def __init__(self,
                 atom_dim: int = 42,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 num_steps: int = 32,
                 num_tasks: int = 1,
                 is_classification: bool = False,
                 superluminal_reg: bool = True,
                 num_heads: int = 4,
                 norm_reg_weight: float = 0.01):
        super().__init__()
        assert latent_dim == 128, "Currently constrained to 128 for 8x16 spatial projection."

        from atom_wave_encoder import AtomWaveEncoder
        from wave_engine import OpenWaveEngine
        from prediction_head import DualPredictionHead

        self.encoder = AtomWaveEncoder(
            atom_dim=atom_dim, latent_dim=latent_dim,
            hidden_dim=hidden_dim, num_heads=num_heads
        )
        self.wave_engine = OpenWaveEngine(latent_dim=latent_dim, num_steps=num_steps)
        self.singularity_detector = SingularityDetector(grid_h=8, grid_w=16)
        self.feature_extractor = VortexFeatureExtractor(latent_dim=latent_dim)

        self.superluminal_reg = superluminal_reg
        self.c_sys = 10.0
        self.norm_reg_weight = norm_reg_weight

        # Dual-head: psi (256d) and physics (4+6=10d) processed separately
        psi_dim = latent_dim * 2
        physics_dim = 4 + 6  # wave_feats + sing_feats
        self.prediction_head = DualPredictionHead(
            psi_dim=psi_dim, physics_dim=physics_dim,
            hidden_dim=hidden_dim, num_tasks=num_tasks,
            is_classification=is_classification
        )

    def topological_dropout(self, psi, W_trajectory, dt_list):
        """Same topological dropout as v1/v2."""
        if not self.training or len(W_trajectory) < 2:
            return psi
        W_curr = W_trajectory[-2]
        W_next = W_trajectory[-1]
        dt = dt_list[-1]
        velocity_field = torch.abs(W_next - W_curr) / (dt + 1e-6)
        peak_v = velocity_field.amax(dim=[-1, -2])
        superluminal_mask = (peak_v > self.c_sys)
        if superluminal_mask.any():
            noise = torch.randn_like(psi) * 0.1
            jump_mask = (torch.rand_like(psi[..., 0]) < 0.1).unsqueeze(-1)
            batch_mask_expanded = superluminal_mask.view(-1, 1, 1).expand_as(psi)
            final_mask = jump_mask & batch_mask_expanded
            psi = torch.where(final_mask, psi + noise, psi)
        return psi

    def forward(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_features: (batch, max_atoms, atom_dim) padded atom features
            atom_mask: (batch, max_atoms) binary mask
        Returns:
            predictions: (batch, num_tasks)
        """
        # 1. Atom-level encoding
        psi = self.encoder(atom_features, atom_mask)
        psi_init = psi  # Save for residual skip

        # 2. Open quantum evolution (with amplitude modulation)
        w_traj = []
        dt_list = []
        topo_features = []

        for t in range(self.wave_engine.num_steps):
            dt = self.wave_engine.dt[t]
            psi = self.wave_engine.unitary_evolution(psi)
            psi = self.wave_engine.non_linear_gp(psi, dt)

            # Open system amplitude modulation
            epsilon = self.wave_engine.amplitude_mod[t]
            scale = 1.0 + 0.1 * torch.tanh(epsilon)
            psi = psi * scale.unsqueeze(0).unsqueeze(-1)

            # Damping
            gamma = self.wave_engine.damping
            psi = psi * (1.0 - gamma * dt)

            W_t = self.singularity_detector.calculate_winding_field(psi)
            w_traj.append(W_t)
            if t < self.wave_engine.num_steps - 1:
                dt_list.append(dt)
            feats_t = self.singularity_detector.extract_vortex_features(W_t)
            topo_features.append(feats_t)

            if self.superluminal_reg and t > 0:
                psi = self.topological_dropout(psi, w_traj, dt_list)

        # Residual skip: mix initial state into evolved state
        alpha = torch.sigmoid(self.wave_engine.skip_alpha)
        psi = (1.0 - alpha) * psi + alpha * psi_init

        # 3. Kinematics
        kinematics = self.singularity_detector.track_kinematics(w_traj, dt_list)

        # 4. Separate feature extraction
        wave_feats = self.feature_extractor.compute_wave_features(psi)       # (batch, 4)
        sing_feats = self.feature_extractor.aggregate_singularity_features(  # (batch, 6)
            topo_features, kinematics, psi.size(0), psi.device
        )
        physics_feats = torch.cat([wave_feats, sing_feats], dim=-1)  # (batch, 10)

        psi_flat = psi.view(psi.size(0), -1)  # (batch, 256)

        # 5. Dual-head prediction
        out = self.prediction_head(psi_flat, physics_feats)

        return out

    def get_reg_loss(self):
        """Returns norm regularization loss from wave engine."""
        if self.wave_engine.norm_reg_loss is not None:
            return self.norm_reg_weight * self.wave_engine.norm_reg_loss
        return torch.tensor(0.0)


def _test_vortexnet_v3():
    print("Testing VortexNetV3 (Open Quantum System)...")

    batch = 8
    max_atoms = 80
    atom_dim = 42

    atom_feats = torch.randn(batch, max_atoms, atom_dim)
    atom_mask = torch.zeros(batch, max_atoms)
    for i in range(batch):
        n = torch.randint(3, 30, (1,)).item()
        atom_mask[i, :n] = 1.0
        atom_feats[i, n:] = 0.0

    model = VortexNetV3(atom_dim=atom_dim, num_steps=5, superluminal_reg=True)

    preds = model(atom_feats, atom_mask)
    assert preds.shape == (batch, 1), f"Wrong output shape: {preds.shape}"
    print(f"  Forward Pass OK. Output Shape: {preds.shape}")

    # Check reg loss
    reg = model.get_reg_loss()
    print(f"  Norm Reg Loss: {reg.item():.6f}")

    target = torch.randn(batch, 1)
    loss = torch.nn.functional.mse_loss(preds, target) + reg
    loss.backward()

    grad_checks = {
        'AtomWaveEncoder MLP': model.encoder.atom_mlp[0].weight.grad is not None,
        'OpenWaveEngine amplitude_mod': model.wave_engine.amplitude_mod[0].grad is not None,
        'OpenWaveEngine skip_alpha': model.wave_engine.skip_alpha.grad is not None,
        'DualHead psi_branch': model.prediction_head.psi_branch[0].weight.grad is not None,
        'DualHead physics_branch': model.prediction_head.physics_branch[0].weight.grad is not None,
        'DualHead gate': model.prediction_head.gate[0].weight.grad is not None,
    }

    for name, ok in grad_checks.items():
        print(f"  Gradient flow into {name}: {'OK' if ok else 'FAILED'}")
        assert ok, f"Missing gradients for {name}"

    print("VortexNetV3 End-to-End Pipeline successfully tested!")


if __name__ == "__main__":
    _test_vortexnet()
    print()
    _test_vortexnet_v2()
    print()
    _test_vortexnet_v3()
