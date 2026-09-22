"""
Complex Encoder for VortexNet.
Encodes molecular descriptors (real-valued) into complex-valued wave functions
in a 128-dimensional latent space: psi = A * exp(i * phi).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.complex_ops import (
    complex_from_polar, complex_modulus, complex_phase,
    ComplexLinear, ComplexLayerNorm
)


class ComplexEncoder(nn.Module):
    """Encodes real-valued molecular descriptors into complex wave functions.

    Architecture:
        Input (D_in) -> Shared MLP -> Branch into Amplitude + Phase
        -> Combine as psi = A * exp(i*phi) in C^128
    """

    def __init__(self, input_dim: int = 217, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Amplitude branch: must be positive (use softplus)
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),
        )

        # Phase branch: unbounded [-pi, pi] expected
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # Output in [-1, 1], scale to [-pi, pi]
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable complex gradients."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) real-valued molecular descriptors

        Returns:
            psi: (batch, latent_dim, 2) complex wave function
                 psi[..., 0] = real part, psi[..., 1] = imaginary part
        """
        h = self.shared(x)  # (batch, hidden_dim)

        amplitude = self.amplitude_head(h)       # (batch, latent_dim), positive
        phase = self.phase_head(h) * torch.pi    # (batch, latent_dim), [-pi, pi]

        # psi = A * exp(i*phi)
        psi = complex_from_polar(amplitude, phase)  # (batch, latent_dim, 2)

        return psi

    def get_amplitude_phase(self, x: torch.Tensor):
        """Return amplitude and phase separately for analysis."""
        h = self.shared(x)
        amplitude = self.amplitude_head(h)
        phase = self.phase_head(h) * torch.pi
        return amplitude, phase


class ComplexDecoder(nn.Module):
    """Optional decoder for reconstruction loss (autoencoder pretraining).
    Takes complex wave function back to molecular descriptor space."""

    def __init__(self, output_dim: int = 217, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # Takes both amplitude and phase as input (2 * latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (batch, latent_dim, 2) complex wave function

        Returns:
            x_hat: (batch, output_dim) reconstructed descriptors
        """
        amplitude = complex_modulus(psi)    # (batch, latent_dim)
        phase = complex_phase(psi)          # (batch, latent_dim)
        features = torch.cat([amplitude, phase], dim=-1)  # (batch, 2*latent_dim)
        return self.decoder(features)


def test_complex_encoder():
    """Test encoder gradient flow and output properties."""
    print("Testing ComplexEncoder...")

    batch_size = 16
    input_dim = 217
    latent_dim = 128

    encoder = ComplexEncoder(input_dim=input_dim, latent_dim=latent_dim)
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    psi = encoder(x)
    assert psi.shape == (batch_size, latent_dim, 2), f"Wrong shape: {psi.shape}"

    # Amplitude should be positive
    amp = complex_modulus(psi)
    assert (amp > 0).all(), "Amplitude should be positive"

    # Phase should be in [-pi, pi]
    phase = complex_phase(psi)
    assert (phase >= -torch.pi - 0.01).all() and (phase <= torch.pi + 0.01).all(), \
        f"Phase out of range: [{phase.min()}, {phase.max()}]"

    # Gradient flow
    loss = complex_modulus(psi).sum()
    loss.backward()

    grad_ok = True
    for name, param in encoder.named_parameters():
        if param.grad is None:
            print(f"  WARNING: No gradient for {name}")
            grad_ok = False

    assert grad_ok, "Some parameters have no gradient"

    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder parameters: {n_params:,}")
    print(f"  Output shape: {psi.shape}")
    print(f"  Amplitude range: [{amp.min():.4f}, {amp.max():.4f}]")
    print(f"  Phase range: [{phase.min():.4f}, {phase.max():.4f}]")

    # Test decoder
    print("Testing ComplexDecoder...")
    decoder = ComplexDecoder(output_dim=input_dim, latent_dim=latent_dim)
    x_hat = decoder(psi.detach())
    assert x_hat.shape == (batch_size, input_dim), f"Decoder wrong shape: {x_hat.shape}"

    # Sanity check: overfit on single sample
    print("Testing overfit on single sample...")
    encoder_single = ComplexEncoder(input_dim=input_dim, latent_dim=latent_dim)
    decoder_single = ComplexDecoder(output_dim=input_dim, latent_dim=latent_dim)

    x_single = torch.randn(1, input_dim)
    optimizer = torch.optim.Adam(
        list(encoder_single.parameters()) + list(decoder_single.parameters()),
        lr=1e-3
    )

    for step in range(200):
        psi_s = encoder_single(x_single)
        x_hat_s = decoder_single(psi_s)
        loss = F.mse_loss(x_hat_s, x_single)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = F.mse_loss(decoder_single(encoder_single(x_single)), x_single).item()
    print(f"  Overfit loss after 200 steps: {final_loss:.6f}")
    assert final_loss < 0.1, f"Failed to overfit: loss={final_loss}"

    print("All encoder tests passed!")


if __name__ == "__main__":
    test_complex_encoder()
