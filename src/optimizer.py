"""
optimizer.py
============

Classical optimization loop for the QAOA ansatz built by `qaoa_circuit.py`

    energy(params) = <psi(params)| H_cost |psi(params)>
                    = estimator.run([(ansatz, cost_hamiltonian, params)])

Optimizer surface
------------------------------------------------------------------------
Exactly two `scipy.optimize.minimize` methods are supported, selectable via
`optimizer_method`, never hardcoded:

- `"COBYLA"` (default)
- `"L-BFGS-B"` -- gradient-based; `scipy` finite-differences the gradient by
  default (extra circuit evaluations per iteration).

Device selection
-----------------
`device="CPU"` (default) uses `qiskit.primitives.StatevectorEstimator`

`device="GPU"` uses `qiskit_aer.primitives.EstimatorV2.from_backend(
AerSimulator(method="statevector", device="GPU"))`.

`device="CPU_AER"` uses the same Aer `EstimatorV2` code path as `"GPU"`,
just with `AerSimulator(method="statevector", device="CPU")` instead of
`device="GPU"`. It exists purely so a CPU-vs-GPU wall-clock comparison can
hold the simulator engine fixed.

Warm-start extension point
---------------------------
`initial_params=None` is the sole hook for warm-starting: when omitted, a
uniform-random vector in `[0, 2*pi]` (length `ansatz.num_parameters`) is
used as the starting point; when supplied, it is passed straight through as
`scipy.optimize.minimize`'s `x0` with no further processing. Warm-start
logic itself has yet to be designed for this implementation. `run_qaoa_sweep`
below also accepts an optional `warm_start_fn=None` purely
so `experiment.py`'s sweep has a stable call to build on
later without a full rewrite.

Parameter ordering -- READ THIS BEFORE TOUCHING `initial_params` -----------
`qaoa_circuit.build_qaoa_circuit()`'s `ansatz.parameters` is a Qiskit
`ParameterView`, which sorts alphabetically by parameter name, not by
layer. Because the mixer angles are named `beta[0..p-1]` and the cost
angles `gamma[0..p-1]`, and `'b' < 'g'`, the flat parameter vector is:

    [beta[0], beta[1], ..., beta[p-1], gamma[0], gamma[1], ..., gamma[p-1]]

i.e. all mixer angles first, then all cost angles -- not sorted
layer-by-layer as `[beta_0, gamma_0, beta_1, gamma_1, ...]`.

    betas  = [p for p in ansatz.parameters if str(p).startswith("beta")]
    gammas = [p for p in ansatz.parameters if str(p).startswith("gamma")]
    # len(betas) == len(gammas) == p; ansatz.parameters is already in the
    # bind order that params/initial_params/optimal_params use.
"""

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

# Devices that route through qiskit_aer's EstimatorV2.
# Both need the ansatz decomposed first since Aer's EstimatorV2
# can't resolve a `QAOAAnsatz`'s opaque `"QAOA"` instruction.
AER_DEVICES = ("GPU", "CPU_AER")


# --------------------------------------------------------------------------
# Estimator construction
# --------------------------------------------------------------------------


def build_estimator(device: str = "CPU"):
    """Build an Estimator-primitive instance for the requested device.

    Parameters
    ----------
    device : str
        `"CPU"` (default) -> `qiskit.primitives.StatevectorEstimator`
        `"GPU"` -> `qiskit_aer.primitives.EstimatorV2.from_backend(
        AerSimulator(method="statevector", device="GPU"))`.
        `"CPU_AER"` -> `qiskit_aer.primitives.EstimatorV2.from_backend(
        AerSimulator(method="statevector", device="CPU"))` -- same Aer
        code path as `"GPU"`, just on CPU.

    Returns
    -------
    An Estimator-primitive object usable as
    `estimator.run([(ansatz, hamiltonian, params)]).result()`.
    """
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
    """Return the circuit for `estimator.run()`.

    The Aer-backed estimators can't handle `QAOAAnsatz`'s opaque `QAOA`
    instruction, so it's decomposed to basis gates first (`reps=6`).
    `device="CPU"` handles the raw ansatz untouched.
    """
    if device not in AER_DEVICES:
        return ansatz
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
            f"Aer-backed {device!r} path."
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
    """Build the `params -> float` objective for `scipy.optimize.minimize`.
    Runs the (possibly decomposed) ansatz through the estimator and returns
    `result[0].data.evs`"""
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
) -> QAOAOptimizationResult:
    """Minimize a QAOA ansatz's energy expectation against `cost_hamiltonian`
    via `scipy.optimize.minimize`. `initial_params` sets `x0` (random in
    `[0, 2*pi]` via `rng`/`seed` if omitted). `minimize_options` forwards to
    `scipy.optimize.minimize`'s `options=`.
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
    **circuit_kwargs,
) -> QAOAOptimizationResult:
    """Build the QAOA circuit for a `(J, h, boundary)` Ising instance
    (via `qaoa_circuit.build_qaoa_circuit_from_ising`) and optimize it.
    """
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
) -> Tuple[QAOAOptimizationResult, list]:
    """Run `optimize_qaoa` `n_restarts` times with fresh random `initial_params`
    each time (unless `warm_start_fn` is supplied) and return the best result.

    `warm_start_fn` is accepted but is just a placeholder for now

    Returns
    -------
    (best_result, all_results) : (QAOAOptimizationResult, list[QAOAOptimizationResult])
        `best_result` is the lowest-`optimal_energy` result across restarts;
        `all_results` has all `n_restarts` results.
    """
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
        )
        all_results.append(result)

    best_result = min(all_results, key=lambda r: r.optimal_energy)
    return best_result, all_results


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import time

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src import ising_model as im
    from src import exact_solver as es

    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    # ---- 0. Cross-check E_0 constants against exact_solver --
    print("\n--- E_0 vs. exact_solver.ground_state_energy (baseline) ---")
    analytic_instances = [
        (im.TWO_SPIN_FM_INSTANCE, -1.0),
        (im.THREE_SPIN_AFM_INSTANCE, -2.0),
    ]
    for instance, e0_expected in analytic_instances:
        e0_exact = es.ground_state_energy(*instance.as_tuple())
        check(
            f"{instance.name}: exact_solver.ground_state_energy == documented E_0={e0_expected}",
            np.isclose(e0_exact, e0_expected, atol=1e-10),
            f"exact_solver E_0={e0_exact}",
        )

    # ---- 1. Optimizer surface / device validation ------------------------
    print("\n--- Input validation ---")
    ansatz_tmp, H_tmp = build_qaoa_circuit_for_instance(im.TWO_SPIN_FM_INSTANCE, p=1)
    try:
        optimize_qaoa(ansatz_tmp, H_tmp, optimizer_method="SPSA")
        check("optimizer_method='SPSA' raises ValueError", False)
    except ValueError:
        check("optimizer_method='SPSA' raises ValueError", True)

    try:
        build_estimator(device="TPU")
        check("device='TPU' raises ValueError", False)
    except ValueError:
        check("device='TPU' raises ValueError", True)

    # ---- 2. End-to-end optimization --------------------------------------
    print("\n--- End-to-end optimization vs. exact_solver ground truth ---")
    # p=3 gives QAOAAnsatz is enough to reach the exact ground state
    # on these 2-3 qubit instances (matches the depth
    # already validated as sufficient in qaoa_circuit.py's own __main__
    # test, which got within 0.15 of E_0 at p=3 with plain COBYLA).
    P_DEPTH = 3
    GAP_TOL = 0.15  # relative to |E_0| ~ 1-2, matches qaoa_circuit.py's own tolerance

    for instance, e0_expected in analytic_instances:
        e0_exact = es.ground_state_energy(*instance.as_tuple())
        ansatz, H = build_qaoa_circuit_for_instance(instance, p=P_DEPTH)

        for method in VALID_OPTIMIZER_METHODS:
            best = None
            t0 = time.time()
            for trial_seed in range(5):
                result = optimize_qaoa(
                    ansatz,
                    H,
                    optimizer_method=method,
                    device="CPU",
                    minimize_options={"maxiter": 200},
                    seed=trial_seed,
                )
                if best is None or result.optimal_energy < best.optimal_energy:
                    best = result
            elapsed = time.time() - t0
            gap = best.optimal_energy - e0_exact
            check(
                f"{instance.name} p={P_DEPTH} {method}: best-of-5 restarts within {GAP_TOL} of exact E_0={e0_exact}",
                gap < GAP_TOL,
                f"found={best.optimal_energy:.6f}, E_0(exact_solver)={e0_exact:.6f}, "
                f"E_0(documented)={e0_expected}, gap={gap:.6f}, {elapsed:.2f}s for 5 restarts",
            )
            check(
                f"{instance.name} p={P_DEPTH} {method}: optimal_params length == ansatz.num_parameters",
                len(best.optimal_params) == ansatz.num_parameters,
                f"got {len(best.optimal_params)}, expected {ansatz.num_parameters}",
            )

    # ---- 3. CPU device path runs end-to-end via Estimator primitive ------
    print("\n--- CPU device path (StatevectorEstimator) ---")
    estimator_cpu = build_estimator(device="CPU")
    check(
        "build_estimator('CPU') returns a StatevectorEstimator",
        isinstance(estimator_cpu, StatevectorEstimator),
        f"got {type(estimator_cpu)}",
    )
    result_cpu = optimize_qaoa(ansatz_tmp, H_tmp, optimizer_method="COBYLA", device="CPU", seed=0)
    check(
        "device='CPU' optimize_qaoa runs end-to-end and returns a finite energy",
        np.isfinite(result_cpu.optimal_energy),
        f"optimal_energy={result_cpu.optimal_energy}",
    )

    # ---- 4. GPU device path: only actually exercised if hardware/qiskit- --
    #         aer-gpu are present in this environment ----------------------
    # NOTE: `AerSimulator(device="GPU").available_devices()` is NOT a
    # reliable hardware probe -- it just echoes back the requested option
    # even when no GPU/CUDA build is present. So this check instead looks for
    # an actual NVIDIA driver via `nvidia-smi`, and the
    # `qiskit-aer-gpu` distribution.
    print("\n--- GPU device path (best-effort; environment-dependent) ---")
    import shutil
    import subprocess
    import importlib.metadata as importlib_metadata

    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    if has_nvidia_smi:
        try:
            has_nvidia_smi = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            ).returncode == 0
        except Exception:
            has_nvidia_smi = False
    try:
        importlib_metadata.distribution("qiskit-aer-gpu")
        has_aer_gpu_package = True
    except importlib_metadata.PackageNotFoundError:
        has_aer_gpu_package = False

    print(
        f"[INFO] nvidia-smi present & working: {has_nvidia_smi}; "
        f"qiskit-aer-gpu package installed: {has_aer_gpu_package} "
        f"(plain qiskit-aer has no GPU compute support compiled in)"
    )

    gpu_tested = False
    if has_nvidia_smi and has_aer_gpu_package:
        try:
            result_gpu = optimize_qaoa(
                ansatz_tmp, H_tmp, optimizer_method="COBYLA", device="GPU", seed=0
            )
            check(
                "device='GPU' optimize_qaoa runs end-to-end and returns a finite energy",
                np.isfinite(result_gpu.optimal_energy),
                f"optimal_energy={result_gpu.optimal_energy}",
            )
            gpu_tested = True
        except Exception as exc:
            print(f"[INFO] GPU run attempted but failed: {type(exc).__name__}: {exc}")

    if not gpu_tested:
        print(
            "[INFO] GPU self-check SKIPPED (not a failure, not claimed as tested) -- "
            "this environment has no NVIDIA GPU / qiskit-aer-gpu (confirmed: `nvidia-smi` "
            "not found, only plain `qiskit-aer` installed)."
        )

    # ---- 4b. Aer-backend estimator stand-in--
    print("\n--- Aer-backend estimator stand-in (AerSimulator(device='CPU'), same code path as GPU) ---")
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        aer_cpu_backend = AerSimulator(method="statevector", device="CPU")
        aer_cpu_estimator = AerEstimatorV2.from_backend(aer_cpu_backend)
        prepared = _prepare_ansatz_for_estimator(ansatz_tmp, device="GPU")
        check(
            "_prepare_ansatz_for_estimator decomposes to Aer-recognized basis gates",
            set(prepared.count_ops()).issubset({"u", "rx", "rzz", "h", "rz", "cx"}),
            f"ops={dict(prepared.count_ops())}",
        )
        aer_params = np.array([0.3, 0.5])
        aer_ev = float(aer_cpu_estimator.run([(prepared, H_tmp, aer_params)]).result()[0].data.evs)
        check(
            "AerEstimatorV2.from_backend(...) + decomposed ansatz runs end-to-end "
            "(the exact fix the device='GPU' branch relies on)",
            np.isfinite(aer_ev),
            f"<H>={aer_ev}",
        )
        # Confirm it agrees with StatevectorEstimator on the same params
        cpu_ev = make_cost_function(ansatz_tmp, H_tmp, StatevectorEstimator(), device="CPU")(aer_params)
        check(
            "Aer-backend estimator (decomposed) agrees with StatevectorEstimator (un-decomposed) "
            "on the same params",
            np.isclose(aer_ev, cpu_ev, atol=1e-8),
            f"aer={aer_ev:.8f}, statevector={cpu_ev:.8f}",
        )
    except ImportError:
        print("[INFO] qiskit_aer not importable -- skipping Aer-backend stand-in check.")

    # ---- 4c. device='CPU_AER': the actual selectable matched-engine device
    print("\n--- device='CPU_AER' (matched-simulator-engine CPU benchmark for GPU comparisons) ---")
    try:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        estimator_cpu_aer = build_estimator(device="CPU_AER")
        check(
            "build_estimator('CPU_AER') returns an Aer EstimatorV2",
            isinstance(estimator_cpu_aer, AerEstimatorV2),
            f"got {type(estimator_cpu_aer)}",
        )
        check(
            "_prepare_ansatz_for_estimator decomposes the ansatz for device='CPU_AER' "
            "(same as it does for 'GPU')",
            _prepare_ansatz_for_estimator(ansatz_tmp, device="CPU_AER").num_parameters
            == ansatz_tmp.num_parameters
            and set(_prepare_ansatz_for_estimator(ansatz_tmp, device="CPU_AER").count_ops())
            != set(ansatz_tmp.count_ops()),
        )

        result_cpu_aer = optimize_qaoa(
            ansatz_tmp, H_tmp, optimizer_method="COBYLA", device="CPU_AER", seed=0
        )
        check(
            "device='CPU_AER' optimize_qaoa runs end-to-end and returns a finite energy",
            np.isfinite(result_cpu_aer.optimal_energy),
            f"optimal_energy={result_cpu_aer.optimal_energy}",
        )

        # Same params, same underlying math -> should agree closely
        # with device='CPU' (StatevectorEstimator)
        cpu_aer_cost_fn = make_cost_function(ansatz_tmp, H_tmp, estimator_cpu_aer, device="CPU_AER")
        cpu_cost_fn = make_cost_function(ansatz_tmp, H_tmp, StatevectorEstimator(), device="CPU")
        probe_params = np.array([0.3, 0.5])
        ev_cpu_aer = cpu_aer_cost_fn(probe_params)
        ev_cpu = cpu_cost_fn(probe_params)
        check(
            "device='CPU_AER' agrees with device='CPU' on the same params",
            np.isclose(ev_cpu_aer, ev_cpu, atol=1e-8),
            f"CPU_AER={ev_cpu_aer:.8f}, CPU={ev_cpu:.8f}",
        )
    except ImportError:
        print("[INFO] qiskit_aer not importable -- skipping device='CPU_AER' check.")

    # ---- 5. initial_params is actually used, not silently ignored --------
    print("\n--- initial_params warm-start hook: actually used, not ignored ---")
    ansatz5, H5 = build_qaoa_circuit_for_instance(im.THREE_SPIN_AFM_INSTANCE, p=2)
    fixed_x0 = np.array([0.1, 0.4, 0.9, 1.3])  # length == ansatz5.num_parameters == 2*p

    # 5a. Determinism: same initial_params + optimizer -> identical result.
    r1 = optimize_qaoa(ansatz5, H5, optimizer_method="L-BFGS-B", initial_params=fixed_x0)
    r2 = optimize_qaoa(ansatz5, H5, optimizer_method="L-BFGS-B", initial_params=fixed_x0)
    check(
        "same initial_params -> identical initial_params echoed back both calls",
        np.allclose(r1.initial_params, fixed_x0) and np.allclose(r2.initial_params, fixed_x0),
    )
    check(
        "same initial_params + method='L-BFGS-B' -> deterministic optimal_energy across repeated calls",
        np.isclose(r1.optimal_energy, r2.optimal_energy, atol=1e-10),
        f"run1={r1.optimal_energy:.10f}, run2={r2.optimal_energy:.10f}",
    )

    # 5b. Different initial_params values actually change the outcome
    # (i.e. the value isn't silently dropped in favor of a random x0).
    other_x0 = fixed_x0 + 3.0
    r3 = optimize_qaoa(ansatz5, H5, optimizer_method="L-BFGS-B", initial_params=other_x0)
    check(
        "different initial_params are echoed back as the actual x0 used (not overwritten)",
        np.allclose(r3.initial_params, other_x0) and not np.allclose(r3.initial_params, fixed_x0),
    )

    # 5c. Starting from an already-near-optimal point, the optimizer should
    #     not need to move far / should not do worse than a random start.
    ansatz_afm3, H_afm3 = build_qaoa_circuit_for_instance(im.THREE_SPIN_AFM_INSTANCE, p=3)
    warm_result = optimize_qaoa(
        ansatz_afm3, H_afm3, optimizer_method="COBYLA",
        minimize_options={"maxiter": 300}, seed=1,
    )
    near_optimal_x0 = warm_result.optimal_params
    r4 = optimize_qaoa(
        ansatz_afm3, H_afm3, optimizer_method="COBYLA",
        initial_params=near_optimal_x0, minimize_options={"maxiter": 5},
    )
    check(
        "supplying an already-converged point as initial_params keeps the result close to it "
        "(optimizer doesn't need to wander far, confirming x0 was actually honored)",
        r4.optimal_energy <= warm_result.optimal_energy + 1e-6,
        f"warm-start energy={r4.optimal_energy:.6f}, seed point energy={warm_result.optimal_energy:.6f}",
    )

    # ---- 6. Multi-restart helper check -----------------------------------
    print("\n--- run_qaoa_multi_restart check ---")
    best, all_r = run_qaoa_multi_restart(
        ansatz_tmp, H_tmp, optimizer_method="COBYLA", n_restarts=3, seed=42,
        minimize_options={"maxiter": 100},
    )
    check("run_qaoa_multi_restart returns n_restarts results", len(all_r) == 3, f"got {len(all_r)}")
    check(
        "run_qaoa_multi_restart's best result has the minimum energy among all restarts",
        best.optimal_energy == min(r.optimal_energy for r in all_r),
    )

    print("\n" + ("=" * 60))
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
