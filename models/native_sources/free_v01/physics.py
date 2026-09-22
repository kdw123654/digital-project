"""Dimensionless quartic fields and Langevin dynamics (k_B = 1).

U(x, u) = sum(a*x^2/2 + b*x^4/4 - h*x) - x' W x/2 - x' B u.
W is symmetric, sparse and has zero diagonal. This is a classical numerical
model, not a claim to simulate quantum physics or remove discretization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np


def finite(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numbers")
    return arr


def positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def ports(value, n, name="ports"):
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and (raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= n))):
        raise ValueError(f"{name} must contain valid integer indices")
    result = raw.astype(int)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must contain distinct indices")
    return result


@dataclass
class Medium:
    quadratic: np.ndarray
    quartic: np.ndarray
    bias: np.ndarray
    coupling: np.ndarray
    drive: np.ndarray
    mass: np.ndarray
    damping: np.ndarray
    positions: np.ndarray
    allowed: np.ndarray
    backbone: np.ndarray

    def __post_init__(self):
        for name in ("quadratic", "quartic", "bias", "coupling", "drive", "mass", "damping", "positions"):
            setattr(self, name, finite(getattr(self, name), name).copy())
        self.allowed = np.asarray(self.allowed, dtype=bool).copy()
        self.backbone = np.asarray(self.backbone, dtype=bool).copy()
        n = self.quadratic.size
        if n < 1 or any(getattr(self, k).shape != (n,) for k in ("quadratic", "quartic", "bias", "mass", "damping")):
            raise ValueError("Medium vectors must share shape (n,)")
        if self.coupling.shape != (n, n) or self.allowed.shape != (n, n) or self.backbone.shape != (n, n):
            raise ValueError("coupling and topology must have shape (n, n)")
        if self.drive.ndim != 2 or self.drive.shape[0] != n or self.positions.shape != (n, 2):
            raise ValueError("drive must be (n, inputs) and positions must be (n, 2)")
        if not np.allclose(self.coupling, self.coupling.T, rtol=0, atol=1e-12) or np.any(np.diag(self.coupling) != 0):
            raise ValueError("coupling must be symmetric with zero diagonal")
        for mask in (self.allowed, self.backbone):
            if not np.array_equal(mask, mask.T) or np.any(np.diag(mask)):
                raise ValueError("topology must be symmetric with no self edges")
        if np.any(self.backbone & ~self.allowed) or np.any(self.coupling[~self.allowed] != 0):
            raise ValueError("coupling/backbone must stay inside the allowed topology")
        if np.any(self.mass <= 0) or np.any(self.damping <= 0):
            raise ValueError("mass and damping must be positive")
        if np.any(self.quartic < 0):
            raise ValueError("quartic coefficients must be nonnegative")
        if not np.all(self.quartic > 0):
            # Mixed/linear media need a positive quadratic form for confinement.
            if np.linalg.eigvalsh(np.diag(self.quadratic) - self.coupling)[0] <= 0:
                raise ValueError("zero quartic coefficients require a positive definite quadratic form")

    @property
    def n(self):
        return len(self.quadratic)

    @property
    def input_dim(self):
        return self.drive.shape[1]

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def random(cls, n=48, input_dim=1, seed=7, coupling=0.65, quartic=0.6, neighbors=5):
        n = integer(n, "n")
        input_dim = integer(input_dim, "input_dim", 0)
        neighbors = integer(neighbors, "neighbors")
        positive(coupling, "coupling", zero=True)
        positive(quartic, "quartic", zero=True)
        rng = np.random.default_rng(seed)
        positions = rng.uniform(0.08, 0.92, (n, 2))
        distance = np.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
        np.fill_diagonal(distance, np.inf)
        allowed = np.zeros((n, n), dtype=bool)
        for i in range(n):
            allowed[i, np.argsort(distance[i])[:min(neighbors, n - 1)]] = True
        # A geometric spanning chain prevents accidental isolated components.
        order = np.argsort(np.arctan2(positions[:, 1] - 0.5, positions[:, 0] - 0.5))
        backbone = np.zeros_like(allowed)
        for i, j in zip(order[:-1], order[1:]):
            backbone[i, j] = backbone[j, i] = True
        allowed = allowed | allowed.T | backbone
        raw = rng.normal(size=(n, n))
        weights = (raw + raw.T) * allowed
        radius = np.max(np.abs(np.linalg.eigvalsh(weights)))
        weights *= coupling / max(radius, 1e-12)
        quadratic = rng.uniform(0.7, 2.2, n)
        if quartic == 0:
            quadratic += coupling
        return cls(quadratic, np.full(n, quartic), rng.normal(0, 0.35, n), weights,
                   rng.uniform(-1.5, 1.5, (n, input_dim)), rng.uniform(0.7, 1.3, n),
                   rng.uniform(0.3, 0.9, n), positions, allowed, backbone)

    def field(self, inputs=None):
        if inputs is None:
            return np.zeros(self.n)
        inputs = finite(inputs, "inputs")
        if inputs.ndim not in (1, 2) or inputs.shape[-1] != self.input_dim:
            raise ValueError(f"inputs must end in dimension {self.input_dim}")
        return inputs @ self.drive.T

    def energy(self, x, inputs=None):
        x = finite(x, "state")
        if x.ndim not in (1, 2) or x.shape[-1] != self.n:
            raise ValueError("state must have shape (n,) or (batch, n)")
        return self._energy(x, self.field(inputs))

    def _energy(self, x, field):
        onsite = self.quadratic * x**2 / 2 + self.quartic * x**4 / 4 - (self.bias + field) * x
        return np.sum(onsite, axis=-1) - np.sum((x @ self.coupling) * x, axis=-1) / 2

    def gradient(self, x, inputs=None):
        x = finite(x, "state")
        if x.ndim not in (1, 2) or x.shape[-1] != self.n:
            raise ValueError("state must have shape (n,) or (batch, n)")
        return self._gradient(x, self.field(inputs))

    def _gradient(self, x, field):
        return self.quadratic * x + self.quartic * x**3 - self.bias - field - x @ self.coupling


class Simulator:
    """Stateful external-input transducer with reproducible thermal noise.

    inertial: BAOAB Langevin splitting; overdamped: Euler-Maruyama.
    All state and RNG changes are transactional on a failed numerical step.
    """

    def __init__(self, medium: Medium, *, dt=0.025, mode="inertial", seed=0, initial=None, batch=1):
        self.medium = medium
        self.dt = positive(dt, "dt")
        if mode not in ("inertial", "overdamped"):
            raise ValueError("mode must be inertial or overdamped")
        self.mode = mode
        batch = integer(batch, "batch")
        self.x = np.zeros((batch, medium.n)) if initial is None else np.atleast_2d(finite(initial, "initial")).copy()
        if self.x.ndim != 2 or self.x.shape[1] != medium.n or self.x.shape[0] < 1:
            raise ValueError("initial must be (n,) or (batch, n)")
        self.v = np.zeros_like(self.x)
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.steps = 0
        self.mean = np.zeros(medium.n)
        self.correlation = np.zeros((medium.n, medium.n))
        self.parameter_work = 0.0

    def step(self, inputs=None, *, temperature=0.0):
        temperature = positive(temperature, "temperature", zero=True)
        m, dt = self.medium, self.dt
        field = m.field(inputs)
        if field.ndim == 2 and field.shape[0] not in (1, self.x.shape[0]):
            raise ValueError("input batch must match state batch")
        rng_state = copy.deepcopy(self.rng.bit_generator.state) if temperature else None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.mode == "inertial":
                    v = self.v - 0.5 * dt * m._gradient(self.x, field) / m.mass
                    x = self.x + 0.5 * dt * v
                    decay = np.exp(-m.damping * dt)
                    v = decay * v
                    if temperature:
                        v += np.sqrt(temperature * (1 - decay**2) / m.mass) * self.rng.normal(size=v.shape)
                    x += 0.5 * dt * v
                    v -= 0.5 * dt * m._gradient(x, field) / m.mass
                else:
                    mobility = 1 / m.damping
                    x = self.x - dt * mobility * m._gradient(self.x, field)
                    if temperature:
                        x += np.sqrt(2 * temperature * mobility * dt) * self.rng.normal(size=x.shape)
                    v = np.zeros_like(x)
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)) or np.max(np.abs(x)) > 1e6 or np.max(np.abs(v)) > 1e9:
                    raise FloatingPointError("divergent state")
        except FloatingPointError as exc:
            if rng_state is not None:
                self.rng.bit_generator.state = rng_state
            raise FloatingPointError("Integration failed without changing state; reduce dt/input/coupling") from exc
        self.x, self.v = x, v
        self.time += dt
        self.steps += 1
        return self.x.copy()

    def total_energy(self, inputs=None):
        kinetic = np.sum(self.medium.mass * self.v**2, axis=-1) / 2 if self.mode == "inertial" else 0.0
        return self.medium.energy(self.x, inputs) + kinetic

    def adapt(self, *, rate=0.005, trace_rate=0.03, decay=0.2, threshold=0.015, max_weight=0.35):
        """Target-free local correlation rule; heuristic, not an optimality claim.

        Candidate edges are spatially restricted. A protected spanning chain
        maintains contact while weak non-backbone edges can disappear/reappear.
        Changing the potential performs parameter work; it is reported explicitly.
        """
        for value, name in ((rate, "rate"), (trace_rate, "trace_rate"), (max_weight, "max_weight")):
            positive(value, name)
        positive(decay, "decay", zero=True)
        positive(threshold, "threshold", zero=True)
        if rate > 1 or trace_rate > 1 or threshold >= max_weight:
            raise ValueError("rates must be <= 1 and threshold < max_weight")
        if not np.all(self.medium.quartic > 0):
            raise ValueError("plasticity requires a confining positive quartic potential")
        m = self.medium
        energy_before = float(np.mean(m.energy(self.x)))
        signal = np.tanh(self.x)  # bounded local sensor, not a neural activation layer
        self.mean += trace_rate * (signal.mean(axis=0) - self.mean)
        centered = signal - self.mean
        self.correlation += trace_rate * (centered.T @ centered / len(signal) - self.correlation)
        weights = m.coupling + rate * (self.correlation - decay * m.coupling) * m.allowed
        # Hysteresis lets an absent contact regrow. Without this rule, a small
        # learning rate followed by hard thresholding would trap all zero edges.
        grow = m.allowed & (m.coupling == 0) & (np.abs(self.correlation) > 5 * threshold)
        weights[grow] = np.sign(self.correlation[grow]) * min(1.1 * threshold, max_weight)
        weights = np.clip(weights, -max_weight, max_weight)
        weights[(np.abs(weights) < threshold) & ~m.backbone] = 0
        weak = m.backbone & (np.abs(weights) < threshold)
        weights[weak] = np.where(weights[weak] < 0, -threshold, threshold)
        m.coupling = ((weights + weights.T) / 2) * m.allowed
        work = float(np.mean(m.energy(self.x))) - energy_before
        self.parameter_work += work
        return {"parameter_work": work, "active_edges": int(np.count_nonzero(np.triu(m.coupling, 1)))}


@dataclass
class Relaxation:
    state: np.ndarray
    converged: np.ndarray
    residual: np.ndarray
    iterations: int
    energies: np.ndarray


def relax(medium: Medium, *, inputs=None, initial=None, clamp_indices=(), clamp_values=None,
          output_indices=(), targets=None, beta=0.0, tolerance=1e-7, max_steps=4000, step_size=0.2):
    """Deterministic zero-temperature relaxation with Armijo step control.

    A convergence result is always returned; callers must check it. Neither a
    finite iteration limit nor low output error is evidence of equilibrium.
    """
    positive(tolerance, "tolerance")
    positive(step_size, "step_size")
    integer(max_steps, "max_steps")
    beta = float(finite(beta, "beta"))
    field = medium.field(inputs)
    clamps = ports(clamp_indices, medium.n, "clamp_indices")
    outputs = ports(output_indices, medium.n, "output_indices")
    if np.intersect1d(clamps, outputs).size:
        raise ValueError("input and output ports cannot overlap")
    values = np.atleast_2d(finite(clamp_values, "clamp_values")) if len(clamps) else None
    target = np.atleast_2d(finite(targets, "targets")) if len(outputs) else None
    batch = max(field.shape[0] if field.ndim == 2 else 1, len(values) if values is not None else 1,
                len(target) if target is not None else 1)
    x = np.zeros((batch, medium.n)) if initial is None else np.atleast_2d(finite(initial, "initial")).copy()
    if x.ndim != 2 or x.shape[1] != medium.n:
        raise ValueError("initial must have shape (batch, n)")
    batch = len(x)
    if field.ndim == 2 and len(field) not in (1, batch):
        raise ValueError("input batch mismatch")
    for val, ids in ((values, clamps), (target, outputs)):
        if val is not None and (val.ndim != 2 or val.shape[-1] != len(ids) or len(val) not in (1, batch)):
            raise ValueError("boundary values must match ports and batch")
    if values is not None:
        x[:, clamps] = values

    def energy(z):
        result = medium._energy(z, field)
        if target is not None:
            result = result + beta * np.sum((z[:, outputs] - target)**2, axis=1) / 2
        return result

    history = [energy(x)]
    residual = np.full(batch, np.inf)
    for iteration in range(max_steps + 1):
        grad = medium._gradient(x, field)
        if target is not None:
            grad[:, outputs] += beta * (x[:, outputs] - target)
        grad[:, clamps] = 0
        residual = np.max(np.abs(grad), axis=1)
        if np.all(residual <= tolerance) or iteration == max_steps:
            return Relaxation(x, residual <= tolerance, residual, iteration, np.asarray(history))
        scale = np.full((batch, 1), step_size)
        old_energy = energy(x)
        norm = np.sum(grad**2, axis=1)
        for _ in range(60):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = x - scale * grad
                candidate_energy = energy(candidate)
            bad = ~np.isfinite(candidate_energy) | (candidate_energy > old_energy - 1e-4 * scale[:, 0] * norm + 1e-14)
            if not np.any(bad):
                break
            scale[bad] *= 0.5
        else:
            raise FloatingPointError("Relaxation could not find a finite descending step")
        x = candidate
        history.append(candidate_energy)
    raise AssertionError("unreachable")
