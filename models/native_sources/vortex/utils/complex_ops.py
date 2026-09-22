"""
Complex-valued operations for VortexNet.
Implements complex tensor arithmetic, Wirtinger derivatives support,
and phase manipulation utilities using PyTorch.
"""

import torch
import torch.nn as nn
import math


# ===== Basic Complex Operations =====

def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two complex tensors represented as (..., 2) real tensors.
    a[..., 0] = real, a[..., 1] = imaginary."""
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack([real, imag], dim=-1)


def complex_conjugate(z: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: (a + bi) -> (a - bi)."""
    return torch.stack([z[..., 0], -z[..., 1]], dim=-1)


def complex_modulus(z: torch.Tensor) -> torch.Tensor:
    """Modulus |z| = sqrt(a^2 + b^2)."""
    return torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-8)


def complex_modulus_sq(z: torch.Tensor) -> torch.Tensor:
    """Squared modulus |z|^2 = a^2 + b^2."""
    return z[..., 0] ** 2 + z[..., 1] ** 2


def complex_phase(z: torch.Tensor) -> torch.Tensor:
    """Phase angle: atan2(imag, real) in [-pi, pi]."""
    return torch.atan2(z[..., 1], z[..., 0])


def complex_from_polar(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Create complex tensor from amplitude and phase.
    psi = A * exp(i * phi) = A*cos(phi) + i*A*sin(phi)."""
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    return torch.stack([real, imag], dim=-1)


def complex_exp(z: torch.Tensor) -> torch.Tensor:
    """Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))."""
    exp_real = torch.exp(z[..., 0])
    real = exp_real * torch.cos(z[..., 1])
    imag = exp_real * torch.sin(z[..., 1])
    return torch.stack([real, imag], dim=-1)


def complex_norm(z: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """L2 norm of complex vector along specified dimension."""
    return torch.sqrt(complex_modulus_sq(z).sum(dim=dim) + 1e-8)


# ===== Phase Utilities =====

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap phase angle to [-pi, pi]."""
    return (phi + math.pi) % (2 * math.pi) - math.pi


def phase_difference(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    """Compute wrapped phase difference."""
    return wrap_phase(phi2 - phi1)


def phase_coherence(phases: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute phase coherence (0 = random, 1 = aligned).
    R = |mean(exp(i*phi))|."""
    cos_sum = torch.cos(phases).mean(dim=dim)
    sin_sum = torch.sin(phases).mean(dim=dim)
    return torch.sqrt(cos_sum ** 2 + sin_sum ** 2 + 1e-8)


# ===== Winding Number =====

def winding_number_2d(phases: torch.Tensor) -> torch.Tensor:
    """Compute winding number for a closed path of phase values.

    Args:
        phases: (..., N) tensor of phase values along a closed path.

    Returns:
        (...,) tensor of integer winding numbers.
    """
    # Phase differences along the path
    dphi = phase_difference(phases[..., :-1], phases[..., 1:])
    # Close the loop
    dphi_close = phase_difference(phases[..., -1], phases[..., 0])
    dphi = torch.cat([dphi, dphi_close.unsqueeze(-1)], dim=-1)
    # Winding number = sum of phase differences / (2*pi)
    return dphi.sum(dim=-1) / (2 * math.pi)


def detect_singularities_pairwise(z: torch.Tensor) -> torch.Tensor:
    """Detect topological singularities in complex field by computing
    winding numbers across pairs of dimensions.

    Args:
        z: (..., D, 2) complex tensor where D is the latent dimension.

    Returns:
        winding: (..., D//2) tensor of winding numbers for each dimension pair.
    """
    phases = complex_phase(z)  # (..., D)
    D = phases.shape[-1]

    windings = []
    for k in range(0, D - 1, 2):
        # Create a small closed path around each dimension pair
        phi_pair = phases[..., k:k+2]  # (..., 2)

        # 4-point path: (k, k+1) -> (k+delta, k+1) -> (k+delta, k+1+delta) -> (k, k+1+delta)
        path = torch.stack([
            phi_pair[..., 0],
            phi_pair[..., 1],
            wrap_phase(phi_pair[..., 0] + math.pi),
            wrap_phase(phi_pair[..., 1] + math.pi),
        ], dim=-1)

        w = winding_number_2d(path)
        windings.append(w)

    return torch.stack(windings, dim=-1)


# ===== Complex Linear Layer =====

class ComplexLinear(nn.Module):
    """Linear layer for complex-valued inputs.
    Implements: W_complex * z = (W_r + i*W_i)(x + iy)
                              = (W_r*x - W_i*y) + i(W_r*y + W_i*x)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features, bias=bias)
        self.W_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., in_features, 2)"""
        x, y = z[..., 0], z[..., 1]
        out_real = self.W_real(x) - self.W_imag(y)
        out_imag = self.W_real(y) + self.W_imag(x)
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex tensors.
    Normalizes the modulus while preserving phase."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D, 2)"""
        mod = complex_modulus(z)  # (..., D)
        mod_norm = (mod - mod.mean(dim=-1, keepdim=True)) / (mod.std(dim=-1, keepdim=True) + 1e-8)
        mod_scaled = mod_norm * self.gamma + self.beta

        # Reconstruct with normalized modulus and original phase
        phase = complex_phase(z)
        return complex_from_polar(torch.abs(mod_scaled), phase)


# ===== Tests =====

def run_tests():
    """Verify all complex operations."""
    print("Testing complex operations...")

    # Test 1: complex_mul
    a = torch.tensor([3.0, 4.0])  # 3 + 4i
    b = torch.tensor([1.0, 2.0])  # 1 + 2i
    result = complex_mul(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    # (3+4i)(1+2i) = 3+6i+4i+8i^2 = -5+10i
    assert torch.allclose(result, torch.tensor([-5.0, 10.0]), atol=1e-5), f"complex_mul failed: {result}"

    # Test 2: modulus
    z = torch.tensor([3.0, 4.0])
    mod = complex_modulus(z.unsqueeze(0)).squeeze(0)
    assert torch.allclose(mod, torch.tensor(5.0), atol=1e-3), f"modulus failed: {mod}"

    # Test 3: phase
    z = torch.tensor([1.0, 1.0])  # 45 degrees
    phi = complex_phase(z.unsqueeze(0)).squeeze(0)
    assert torch.allclose(phi, torch.tensor(math.pi / 4), atol=1e-5), f"phase failed: {phi}"

    # Test 4: polar roundtrip
    A = torch.tensor([2.0, 3.0])
    phi = torch.tensor([0.5, 1.0])
    z = complex_from_polar(A, phi)
    A_back = complex_modulus(z)
    phi_back = complex_phase(z)
    assert torch.allclose(A, A_back, atol=1e-3), f"polar roundtrip A failed"
    assert torch.allclose(phi, phi_back, atol=1e-5), f"polar roundtrip phi failed"

    # Test 5: wrap_phase
    phi = torch.tensor([4.0 * math.pi + 0.1])
    wrapped = wrap_phase(phi)
    assert torch.allclose(wrapped, torch.tensor([0.1]), atol=1e-5), f"wrap failed: {wrapped}"

    # Test 6: winding number
    # Full circle path should give winding number = 1
    N = 100
    angles = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    w = winding_number_2d(angles.unsqueeze(0)).squeeze(0)
    assert torch.allclose(w, torch.tensor(1.0), atol=0.1), f"winding failed: {w}"

    # Test 7: ComplexLinear gradient flow
    layer = ComplexLinear(8, 4)
    z = torch.randn(2, 8, 2, requires_grad=True)
    out = layer(z)
    loss = complex_modulus(out).sum()
    loss.backward()
    assert z.grad is not None, "No gradient through ComplexLinear"

    # Test 8: phase_coherence
    aligned = torch.zeros(10)  # All same phase
    c = phase_coherence(aligned)
    assert c > 0.99, f"coherence of aligned phases should be ~1, got {c}"

    random_phases = torch.rand(1000) * 2 * math.pi - math.pi
    c_rand = phase_coherence(random_phases)
    assert c_rand < 0.2, f"coherence of random phases should be ~0, got {c_rand}"

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
