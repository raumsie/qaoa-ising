"""
noise.py
========

Amplitude-damping noise model used to demonstrate ND-AWS's noise-adaptivity
mechanism (Sec. II E / Fig. 2 of ND-AWS).

Method: ND-AWS, Eq. (10).
F. B. Maciejewski, S. Hadfield, O. Wallis, G. Pennington, S. Brandhofer,
S. Woerner, D. J. Egger, D. Venturelli, "Quantum Approximate Optimization
via Noise-Directed Adaptive Warm-Starting," arXiv:2607.09368 (2026).

Independent reimplementation written directly from the paper. Not derived
from quapopt, the ND-AWS authors' reference implementation:
https://github.com/usra-riacs/quantum-approximate-optimization

Eq. (10): single-qubit Kraus operators

    K_0 = |0><0| + sqrt(1-q) |1><1|,   K_1 = sqrt(q) |0><1|

applied uncorrelated and identical after EVERY gate in the circuit (the
paper sweeps q in {0.01, 0.02, 0.03, 0.04, 0.05, 0.1}; this repo's tests use
a subset of that same grid at n=4-8 qubits rather than n=20/100). Because
amplitude damping is a non-unitary (dissipative) channel, it requires a
density-matrix (or equivalent) simulation method -- NOT the pure-state
`method="statevector"` used elsewhere in this repo's `optimizer.py`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

DEFAULT_ONE_QUBIT_GATES = ("u", "rx", "ry", "rz", "h", "x", "sx", "id")
DEFAULT_TWO_QUBIT_GATES = ("cx", "rzz", "cz", "ecr", "rxx", "ryy")


def amplitude_damping_noise_model(
    q: float,
    one_qubit_gates: Iterable[str] = DEFAULT_ONE_QUBIT_GATES,
    two_qubit_gates: Iterable[str] = DEFAULT_TWO_QUBIT_GATES,
) -> NoiseModel:
    """Eq. (10): build a `qiskit_aer.noise.NoiseModel` applying amplitude
    damping of strength `q` after every gate."""
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q (amplitude-damping strength) must be in [0, 1], got {q}")

    error_1q = amplitude_damping_error(q)
    error_2q = error_1q.tensor(error_1q)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, list(one_qubit_gates))
    noise_model.add_all_qubit_quantum_error(error_2q, list(two_qubit_gates))
    return noise_model


def build_noisy_backend(q: float, method: str = "density_matrix") -> AerSimulator:
    """`AerSimulator` with `amplitude_damping_noise_model(q)` attached;
    `method="density_matrix"` is required (amplitude damping is
    non-unitary)."""
    return AerSimulator(method=method, noise_model=amplitude_damping_noise_model(q))


def build_noisy_estimator(q: float, method: str = "density_matrix"):
    """Noisy `EstimatorV2` (via `qiskit_aer.primitives.EstimatorV2.from_backend`)
    for computing `<H>` expectation values under Eq. (10) amplitude damping.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    backend = build_noisy_backend(q, method=method)
    return AerEstimatorV2.from_backend(backend)


def build_noisy_sampler(q: float, method: str = "density_matrix"):
    """Noisy `SamplerV2` (via `qiskit_aer.primitives.SamplerV2.from_backend`)
    for drawing bitstring samples under Eq. (10) amplitude damping."""
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    backend = build_noisy_backend(q, method=method)
    return AerSamplerV2.from_backend(backend)


def prepare_ansatz_for_aer(ansatz: QuantumCircuit) -> QuantumCircuit:
    """`ansatz.decompose(reps=6)`, with a check that parameter names/order
    are unchanged."""
    decomposed = ansatz.decompose(reps=6)
    original_names = [str(p) for p in ansatz.parameters]
    decomposed_names = [str(p) for p in decomposed.parameters]
    if decomposed_names != original_names:
        raise RuntimeError(
            "ansatz.decompose(reps=6) changed the parameter order/names "
            f"({original_names} -> {decomposed_names}); refusing to "
            "proceed since this would silently corrupt parameter binding."
        )
    return decomposed


def noisy_expectation(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    observable,
    q: float,
    method: str = "density_matrix",
) -> float:
    """Returns `<observable>` for `ansatz` bound at `params`, evaluated
    through a noisy (Eq. 10 amplitude-damping) `EstimatorV2`."""
    estimator = build_noisy_estimator(q, method=method)
    run_ansatz = prepare_ansatz_for_aer(ansatz)
    result = estimator.run([(run_ansatz, observable, params)]).result()
    return float(result[0].data.evs)


def optimize_qaoa_noisy(
    ansatz: QuantumCircuit,
    cost_hamiltonian,
    q: float,
    method: str = "density_matrix",
    **optimize_qaoa_kwargs,
):
    """Optimizes variational parameters directly against a noisy (Eq. 10)
    expectation value via `optimizer.optimize_qaoa`'s `estimator` parameter."""
    from src.optimizer import optimize_qaoa

    estimator = build_noisy_estimator(q, method=method)
    optimize_qaoa_kwargs.setdefault("device", "CPU_AER")
    return optimize_qaoa(ansatz, cost_hamiltonian, estimator=estimator, **optimize_qaoa_kwargs)
