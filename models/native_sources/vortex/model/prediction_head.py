"""
Prediction Head for VortexNet.
Takes the final concatenated feature vector and applies MLPs to predict
chemical properties (Regression or Classification).
"""

import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    """
    MLP blocks mapping extracted physical features and latent space to target properties.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, num_tasks: int = 1, is_classification: bool = False):
        super().__init__()
        self.is_classification = is_classification

        # Two-layer MLP with LayerNorm
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, in_features)
        Returns:
            output: (batch, num_tasks) - Raw logits if classification, target values if regression.
        """
        # Classification heads intentionally return logits. Training code should
        # use BCEWithLogitsLoss or CrossEntropyLoss; callers that need
        # probabilities can apply sigmoid/softmax outside the model.
        return self.net(features)


class DualPredictionHead(nn.Module):
    """
    Dual-branch Prediction Head for VortexNet V3.

    Separates psi_flat (raw wave state) and physics features (wave + singularity)
    into independent processing branches, then fuses via learnable gating.

    This prevents the 256d psi_flat from overwhelming the 10d physics features
    in a single MLP.
    """
    def __init__(self, psi_dim: int = 256, physics_dim: int = 10,
                 hidden_dim: int = 128, num_tasks: int = 1,
                 is_classification: bool = False):
        super().__init__()
        self.is_classification = is_classification
        self.psi_dim = psi_dim
        self.physics_dim = physics_dim

        # Branch A: Process raw wave state
        self.psi_branch = nn.Sequential(
            nn.Linear(psi_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Branch B: Process physics features (amplified pathway)
        self.physics_branch = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Learnable gate: decides how much physics vs psi to use
        # Initialized slightly toward physics to encourage learning
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Final prediction from fused representation
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_tasks)
        )

    def forward(self, psi_flat: torch.Tensor, physics_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi_flat: (batch, psi_dim) raw wave state
            physics_feats: (batch, physics_dim) wave + singularity features
        Returns:
            output: (batch, num_tasks)
        """
        h_psi = self.psi_branch(psi_flat)         # (batch, hidden//2)
        h_phys = self.physics_branch(physics_feats)  # (batch, hidden//2)

        # Compute gate from concatenation of both branches
        gate_input = torch.cat([h_psi, h_phys], dim=-1)  # (batch, hidden)
        g = self.gate(gate_input)  # (batch, 1) in [0, 1]

        # g=1 -> physics, g=0 -> psi
        fused = (1.0 - g) * h_psi + g * h_phys  # (batch, hidden//2)

        out = self.output_head(fused)
        return out
