"""
optimizer.py
============

Classical optimization loop for the QAOA ansatz built by `qaoa_circuit.py`:
minimizes energy(params) = <psi(params)| H_cost |psi(params)> via
`scipy.optimize.minimize` (`"COBYLA"` or `"L-BFGS-B"`), on `device` in
`"CPU"`/`"GPU"`/`"CPU_AER"`. See `notes/src/optimizer.md` for the full
design rationale (estimator/device selection, the unconditional
`ansatz.decompose(reps=6)` perf fix, and the warm-start extension points).

Parameter ordering -- READ THIS BEFORE TOUCHING `initial_params`:
Qiskit's `ansatz.parameters` sorts alphabetically, not by layer.
s"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from src.qaoa_circuit import build_qaoa_circuit_for_instance, build_qaoa_circuit_from_ising

VALID_OPTIMIZER_METHODS = ("COBYLA", "L-BFGS-B")
VALID_DEVICES = ("CPU", "GPU", "CPU_AER")

# --------------------------------------------------------------------------
# Estimator construction
# --------------------------------------------------------------------------


def build_estimator(device: str = "CPU"):
    """Build an Estimator-primitive instance for the requested device.
    See `notes/src/optimizer.md` Sec. 3 & Sec. 8."""
    if device == "CPU":
        return StatevectorEstimator()
    elif device == "GPU":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        backend = AerSimulator(method="statevector", device="GPU")
        return AerEstimatorV2.from_backend(backend)
    elif device == "CPU_AER":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        backend = AerSimulator(method="statevector", device="CPU")
        return AerEstimatorV2.from_backend(backend)
    else:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")


def _prepare_ansatz_for_estimator(ansatz: QuantumCircuit, device: str) -> QuantumCircuit:
    """Return the circuit for `estimator.run()`, decomposed to basis gates."""
    decomposed = ansatz.decompose(reps=6)
    # Guard against decompose() ever silently reordering/dropping the
    # parameters that params/initial_params/optimal_params are indexed against
    original_names = [str(p) for p in ansatz.parameters]
    decomposed_names = [str(p) for p in decomposed.parameters]
    if decomposed_names != original_names:
        raise RuntimeError(
            "ansatz.decompose(reps=6) changed the parameter order/names "
            f"relative to the original ansatz ({original_names} -> "
            f"{decomposed_names}); refusing to proceed since this would "
            "silently corrupt the beta/gamma parameter binding on the "
            f"{device!r} path."
        )
    return decomposed


# --------------------------------------------------------------------------
# Cost function
# --------------------------------------------------------------------------


def make_cost_function(
    ansatz: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    estimator,
    device: str = "CPU",
) -> Callable[[np.ndarray], float]:
    """Build the `params -> float` objective for `scipy.optimize.minimize`."""
    run_ansatz = _prepare_ansatz_for_estimator(ansatz, device)

    def cost_function(params: np.ndarray) -> float:
        result = estimator.run([(run_ansatz, cost_hamiltonian, params)]).result()
        return float(result[0].data.evs)

    return cost_function


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class QAOAOptimizationResult:
    """Outcome of the `run_qaoa_ising`/`optimize_qaoa` call."""

    optimal_params: np.ndarray  # flat vector, ansatz.parameters bind order
    optimal_energy: float
    ansatz: QuantumCircuit
    cost_hamiltonian: SparsePauliOp
    optimizer_method: str
    device: str
    initial_params: np.ndarray  # the x0 actually used (random or supplied)
    n_iterations: int
    n_function_evals: int
    success: bool
    message: str
    scipy_result: object  # raw scipy.optimize.OptimizeResult, for inspection


# --------------------------------------------------------------------------
# Core optimization loop
# --------------------------------------------------------------------------


def optimize_qaoa(
    ansatz: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    initial_params: Optional[np.ndarray] = None,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    estimator=None,
) -> QAOAOptimizationResult:
    """Minimize a QAOA ansatz's energy against `cost_hamiltonian` via
    `scipy.optimize.minimize`.
    """
    if optimizer_method not in VALID_OPTIMIZER_METHODS:
        raise ValueError(
            f"optimizer_method must be one of {VALID_OPTIMIZER_METHODS}, got {optimizer_method!r}"
        )

    n_params = ansatz.num_parameters

    if initial_params is None:
        rng = rng if rng is not None else np.random.default_rng(seed)
        x0 = rng.uniform(0.0, 2.0 * np.pi, size=n_params)
    else:
        x0 = np.asarray(initial_params, dtype=float)
        if x0.shape != (n_params,):
            raise ValueError(
                f"initial_params must have shape ({n_params},) to match "
                f"ansatz.num_parameters, got {x0.shape}"
            )

    if estimator is None:
        estimator = build_estimator(device)
    cost_function = make_cost_function(ansatz, cost_hamiltonian, estimator, device=device)

    scipy_result = minimize(
        cost_function,
        x0,
        method=optimizer_method,
        options=minimize_options,
    )

    return QAOAOptimizationResult(
        optimal_params=np.asarray(scipy_result.x, dtype=float),
        optimal_energy=float(scipy_result.fun),
        ansatz=ansatz,
        cost_hamiltonian=cost_hamiltonian,
        optimizer_method=optimizer_method,
        device=device,
        initial_params=x0,
        n_iterations=int(getattr(scipy_result, "nit", -1)),
        n_function_evals=int(getattr(scipy_result, "nfev", -1)),
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        scipy_result=scipy_result,
    )


# --------------------------------------------------------------------------
# Convenience wrappers matching qaoa_circuit.py
# --------------------------------------------------------------------------


def run_qaoa_ising(
    J,
    h,
    boundary: str = "OBC",
    p: int = 1,
    n_spins: Optional[int] = None,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    initial_params: Optional[np.ndarray] = None,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    estimator=None,
    **circuit_kwargs,
) -> QAOAOptimizationResult:
    """Build the QAOA circuit for a `(J, h, boundary)` Ising instance and
    optimize it."""
    ansatz, cost_hamiltonian = build_qaoa_circuit_from_ising(
        J, h, boundary=boundary, p=p, n_spins=n_spins, **circuit_kwargs
    )
    return optimize_qaoa(
        ansatz,
        cost_hamiltonian,
        optimizer_method=optimizer_method,
        device=device,
        initial_params=initial_params,
        minimize_options=minimize_options,
        rng=rng,
        seed=seed,
        estimator=estimator,
    )


def run_qaoa_for_instance(
    instance,
    p: int = 1,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    initial_params: Optional[np.ndarray] = None,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    estimator=None,
    **circuit_kwargs,
) -> QAOAOptimizationResult:
    """Same as `run_qaoa_ising`, but takes an `ising_model.IsingInstance` directly."""
    ansatz, cost_hamiltonian = build_qaoa_circuit_for_instance(instance, p=p, **circuit_kwargs)
    return optimize_qaoa(
        ansatz,
        cost_hamiltonian,
        optimizer_method=optimizer_method,
        device=device,
        initial_params=initial_params,
        minimize_options=minimize_options,
        rng=rng,
        seed=seed,
        estimator=estimator,
    )


# --------------------------------------------------------------------------
# Multi-restart sweep (for experiment.py)
# --------------------------------------------------------------------------


def run_qaoa_multi_restart(
    ansatz: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    n_restarts: int = 5,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    warm_start_fn: Optional[Callable[..., np.ndarray]] = None,
    estimator=None,
) -> Tuple[QAOAOptimizationResult, list]:
    """Run `optimize_qaoa` `n_restarts` times and return the best result;
    `warm_start_fn` is accepted but is just a placeholder for now."""
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

    rng = rng if rng is not None else np.random.default_rng(seed)

    all_results = []
    for _ in range(n_restarts):
        result = optimize_qaoa(
            ansatz,
            cost_hamiltonian,
            optimizer_method=optimizer_method,
            device=device,
            initial_params=None,
            minimize_options=minimize_options,
            rng=rng,
            estimator=estimator,
        )
        all_results.append(result)

    best_result = min(all_results, key=lambda r: r.optimal_energy)
    return best_result, all_results
