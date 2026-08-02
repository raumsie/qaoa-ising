"""
egger_ws.py
============

Warm-Start QAOA (WS-QAOA), Continuous and Rounded variants -- D. J. Egger,
J. Marecek, S. Woerner, "Warm-starting quantum optimization," Quantum 5,
479 (2021), arXiv:2009.10095.

Reuses `ws_ansatz.py`'s WS-QAOA initial state `|c> = RY(2*arcsin(sqrt(c)))|0>`
and mixer `H_M^i = 2*sqrt(c_i(1-c_i))*X_i + (1-2*c_i)*Z_i` (Eq. A5-A8;
chosen so `|c>` is the mixer's exact ground state) unchanged. New here: the
classical "continuous relaxation" front end and a uniform
`WarmStartComparisonResult` interface so standard QAOA and Egger-WS can be
run and compared interchangeably.

`run_egger_ws`/`run_standard_qaoa` return a common
`WarmStartComparisonResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.hamiltonian import build_ising_hamiltonian
from src.optimizer import optimize_qaoa, run_qaoa_multi_restart
from src.qaoa_circuit import build_qaoa_circuit_for_instance, build_qaoa_circuit_from_ising
from src.warm_start.ws_ansatz import bias_from_bitstring, build_ws_ansatz

VALID_EGGER_VARIANTS = ("continuous", "rounded")

DEFAULT_MEAN_FIELD_KWARGS = dict(
    n_steps=200,
    inner_steps=5,
    beta_start=None,  # auto-scaled from J/h magnitude -- see mean_field_relaxation
    beta_end=None,
    damping=0.5,
    symmetry_break_scale=1e-2,
)


# --------------------------------------------------------------------------
# Classical front end: naive mean-field relaxation
# --------------------------------------------------------------------------


def _local_field(m: np.ndarray, J: np.ndarray, h: np.ndarray, boundary: str) -> np.ndarray:
    """`L_i = h_i + sum_{j ~ i} J_ij * m_j`, vectorized via array shifts
    (same pattern as `exact_solver._bond_products`: this chain's bonds are
    always consecutive, `0-1, 1-2, ...` plus the PBC wraparound, so slicing
    suffices and avoids a per-bond Python loop)."""
    if boundary == "PBC":
        return h + J * np.roll(m, -1) + np.roll(J, 1) * np.roll(m, 1)
    local_field = h.copy()
    local_field[:-1] += J * m[1:]
    local_field[1:] += J * m[:-1]
    return local_field


def mean_field_relaxation(
    J: np.ndarray,
    h: np.ndarray,
    boundary: str = "OBC",
    n_spins: Optional[int] = None,
    n_steps: int = 200,
    inner_steps: int = 5,
    beta_start: Optional[float] = None,
    beta_end: Optional[float] = None,
    damping: float = 0.5,
    symmetry_break_scale: float = 1e-2,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Naive mean-field / deterministic-annealing relaxation -- this
    module's classical analog of Egger et al.'s continuous relaxation.

        L_i = h_i + sum_{j ~ i} J_ij * m_j        (local field on qubit i)
        m_i <- (1 - damping) * m_i + damping * tanh(-beta * L_i)
    """
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = n_spins if n_spins is not None else len(h)
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if inner_steps < 1:
        raise ValueError(f"inner_steps must be >= 1, got {inner_steps}")

    rng = rng if rng is not None else np.random.default_rng(seed)
    expected_len = n if boundary == "PBC" else n - 1
    if len(J) != expected_len:
        raise ValueError(
            f"boundary={boundary!r} with n_spins={n} expects len(J) == {expected_len}, got {len(J)}"
        )
    if len(h) != n:
        raise ValueError(f"expects len(h) == n_spins ({n}), got {len(h)}")

    scale = float(
        max(
            np.max(np.abs(J)) if len(J) else 0.0,
            np.max(np.abs(h)) if len(h) else 0.0,
            1e-9,
        )
    )
    if beta_start is None:
        beta_start = 0.5 / scale
    if beta_end is None:
        beta_end = 5.0 / scale
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError(f"beta_start/beta_end must be > 0, got {beta_start}, {beta_end}")

    m = symmetry_break_scale * rng.uniform(-1.0, 1.0, size=n)

    betas = np.linspace(beta_start, beta_end, n_steps)
    for beta_mf in betas:
        for _ in range(inner_steps):
            local_field = _local_field(m, J, h, boundary)
            m_target = np.tanh(-beta_mf * local_field)
            m = (1.0 - damping) * m + damping * m_target

    return m


def magnetization_to_bias(m) -> np.ndarray:
    """The per-qubit `P(bit_i = 1)` implied by a magnetization `m_i`,
    matching hamiltonian.py's `spin = 1 - 2*bit` convention.

        c_i = (1 - m_i) / 2
    """
    m = np.asarray(m, dtype=float)
    return np.clip((1.0 - m) / 2.0, 0.0, 1.0)


def round_to_bitstring(m) -> np.ndarray:
    """Majority rule: `bit_i = 0` (spin `+1`) if `m_i >= 0`, else `bit_i =
    1` (spin `-1`) -- matching hamiltonian.py's `spin = 1 - 2*bit`
    convention. Used by Rounded WS-QAOA before applying `bias_from_bitstring`."""
    m = np.asarray(m, dtype=float)
    return (m < 0.0).astype(np.int64)


# --------------------------------------------------------------------------
# Uniform comparison interface
# --------------------------------------------------------------------------


@dataclass
class WarmStartComparisonResult:
    """Result shape shared by all three comparisons;
    `statevector`/`cost_hamiltonian` are always the original physical frame."""

    method: str
    instance_name: str
    p: int
    optimal_params: np.ndarray  # flat (beta, gamma), alphabetical bind order
    optimal_energy: float
    statevector: np.ndarray
    ansatz: QuantumCircuit
    cost_hamiltonian: SparsePauliOp  # ORIGINAL, physical cost Hamiltonian
    extra: dict = field(default_factory=dict)


def _bind_and_get_statevector(ansatz: QuantumCircuit, params: np.ndarray) -> np.ndarray:
    bind = dict(zip(ansatz.parameters, np.asarray(params, dtype=float)))
    bound = ansatz.assign_parameters(bind).decompose(reps=6)
    return Statevector(bound).data


def run_standard_qaoa(
    instance,
    p: int = 1,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    n_restarts: int = 3,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    estimator=None,
) -> WarmStartComparisonResult:
    """Plain (unwarmed) QAOA, in the same `WarmStartComparisonResult` shape
    as `run_egger_ws` -- the baseline for the comparison."""
    ansatz, H = build_qaoa_circuit_for_instance(instance, p=p)
    best, _all = run_qaoa_multi_restart(
        ansatz,
        H,
        optimizer_method=optimizer_method,
        device=device,
        n_restarts=n_restarts,
        minimize_options=minimize_options,
        rng=rng,
        seed=seed,
        estimator=estimator,
    )
    sv = _bind_and_get_statevector(ansatz, best.optimal_params)
    return WarmStartComparisonResult(
        method="standard",
        instance_name=getattr(instance, "name", "unknown"),
        p=p,
        optimal_params=best.optimal_params,
        optimal_energy=best.optimal_energy,
        statevector=sv,
        ansatz=ansatz,
        cost_hamiltonian=H,
        extra={},
    )


def run_egger_ws(
    instance,
    p: int = 1,
    variant: str = "continuous",
    eps: float = 0.1,
    relaxation_kwargs: Optional[dict] = None,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    n_restarts: int = 3,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    estimator=None,
) -> WarmStartComparisonResult:
    """Egger et al. WS-QAOA entry point; `variant` in
    `VALID_EGGER_VARIANTS` selects continuous vs. rounded bias."""
    if variant not in VALID_EGGER_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_EGGER_VARIANTS}, got {variant!r}")

    J, h, boundary = instance.as_tuple()
    n = instance.n_spins
    rng = rng if rng is not None else np.random.default_rng(seed)

    mf_kwargs = dict(DEFAULT_MEAN_FIELD_KWARGS)
    mf_kwargs.update(relaxation_kwargs or {})
    mf_kwargs.pop("rng", None)
    mf_kwargs.pop("seed", None)

    m = mean_field_relaxation(J, h, boundary=boundary, n_spins=n, rng=rng, **mf_kwargs)

    rounded_bits = None
    if variant == "continuous":
        c = magnetization_to_bias(m)
    else:  # "rounded"
        rounded_bits = round_to_bitstring(m)
        c = bias_from_bitstring(rounded_bits, eps)

    H = build_ising_hamiltonian(J, h, boundary)
    ansatz = build_ws_ansatz(H, c, p=p, n_spins=n)

    best, _all = run_qaoa_multi_restart(
        ansatz,
        H,
        optimizer_method=optimizer_method,
        device=device,
        n_restarts=n_restarts,
        minimize_options=minimize_options,
        rng=rng,
        estimator=estimator,
    )
    sv = _bind_and_get_statevector(ansatz, best.optimal_params)

    return WarmStartComparisonResult(
        method=f"egger_{variant}",
        instance_name=getattr(instance, "name", "unknown"),
        p=p,
        optimal_params=best.optimal_params,
        optimal_energy=best.optimal_energy,
        statevector=sv,
        ansatz=ansatz,
        cost_hamiltonian=H,
        extra={
            "c": c,
            "magnetization": m,
            "rounded_bits": rounded_bits,
            "eps": eps if variant == "rounded" else None,
        },
    )


# --------------------------------------------------------------------------
# warm_start_fn adapter (unused entry point)
# --------------------------------------------------------------------------


def make_egger_ws_warm_start_fn(
    variant: str = "continuous",
    eps: float = 0.1,
    relaxation_kwargs: Optional[dict] = None,
    optimizer_method: str = "COBYLA",
    minimize_options: Optional[dict] = None,
    n_restarts: int = 2,
    device: str = "CPU",
    seed: Optional[int] = None,
) -> Callable:
    """`run_egger_ws()` above is the real entry point."""

    def warm_start_fn(instance, p: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed)
        egger_result = run_egger_ws(
            instance,
            p=p,
            variant=variant,
            eps=eps,
            relaxation_kwargs=relaxation_kwargs,
            optimizer_method=optimizer_method,
            n_restarts=n_restarts,
            minimize_options=minimize_options,
            rng=local_rng,
        )
        # Same alphabetical beta-then-gamma bind order as the default
        # ansatz (both built via qaoa_circuit.build_qaoa_circuit).
        gamma_values = egger_result.optimal_params[p:]

        J, h, boundary = instance.as_tuple()
        H = build_ising_hamiltonian(J, h, boundary)
        default_ansatz, _ = build_qaoa_circuit_from_ising(J, h, boundary=boundary, p=p)
        beta_seed = local_rng.uniform(0.0, 0.1, size=p)  # small, non-zero
        x0 = np.concatenate([beta_seed, gamma_values])

        refine = optimize_qaoa(
            default_ansatz,
            H,
            optimizer_method=optimizer_method,
            device=device,
            initial_params=x0,
            minimize_options={"maxiter": 30},
        )
        return refine.optimal_params

    return warm_start_fn
