"""Depth-sweep, optimizer, & warm-start comparison experiment runner.

Part A -- depth sweep: for each (Ising instance, QAOA depth `p`, optimizer
method), runs a multi-restart QAOA optimization
(`optimizer.run_qaoa_multi_restart`) and records the best energy found
against the exact ground-state energy `E_0`
(`exact_solver.ground_state_energy`) as the relative energy error:

    epsilon(p) = (E_qaoa(p) - E_0) / |E_0|

`run_depth_sweep` returns a `list[SweepRecord]`, one per (instance, p,
optimizer_method) combination.

Part B -- warm-start comparison: runs standard QAOA against Egger-WS and
ND-AWS on the uniform AFM ring (`ising_model.generate_uniform_AFM_ring`),
emitting one CSV row per `(n, p, method)`. `run_warm_start_comparison_point`
builds one row, `run_warm_start_comparison` sweeps `(n, p)`, and
`write_warm_start_comparison_csv` writes the result.
"""

from __future__ import annotations

import csv
import itertools
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from src.exact_solver import (
    ground_state_energy,
    ground_state_success_probability,
    two_point_correlators,
)
from src.ising_model import IsingInstance, generate_test_instances, generate_uniform_AFM_ring
from src.qaoa_circuit import build_qaoa_circuit_for_instance
from src import optimizer as optimizer_module
from src.optimizer import (
    VALID_DEVICES,
    VALID_OPTIMIZER_METHODS,
    run_qaoa_multi_restart,
)

# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------


@dataclass
class SweepRecord:
    """One (instance, p, optimizer_method) point of the depth sweep --
    everything a plotting script needs for an `epsilon(p)` vs. `p` curve."""

    instance_name: str
    n_spins: int
    boundary: str
    instance_description: str

    p: int
    optimizer_method: str
    device: str
    n_restarts: int

    E_0: float
    best_energy: float
    epsilon: float

    best_params: np.ndarray
    restart_energies: List[float]  # optimal_energy of every restart, run order
    success: bool
    message: str

    n_function_evals_total: int
    wall_time_s: float


def relative_energy_error(E_qaoa: float, E_0: float) -> float:
    """`epsilon = (E_qaoa - E_0) / |E_0|`. Warns and returns `nan` if
    `E_0 ~ 0` rather than dividing by zero."""
    if abs(E_0) < 1e-12:
        warnings.warn(
            f"relative_energy_error: E_0={E_0!r} is ~0, relative error is "
            "undefined; returning nan.",
            stacklevel=2,
        )
        return float("nan")
    return (E_qaoa - E_0) / abs(E_0)


# --------------------------------------------------------------------------
# Single sweep point
# --------------------------------------------------------------------------


def run_sweep_point(
    instance: IsingInstance,
    p: int,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    n_restarts: int = 5,
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    warm_start_fn: Optional[Callable] = None,
    E_0: Optional[float] = None,
) -> SweepRecord:
    """Run one (instance, p, optimizer_method) point: build the ansatz,
    run `optimizer.run_qaoa_multi_restart`, record against `E_0`."""
    if optimizer_method not in VALID_OPTIMIZER_METHODS:
        raise ValueError(
            f"optimizer_method must be one of {VALID_OPTIMIZER_METHODS}, got {optimizer_method!r}"
        )
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")

    if E_0 is None:
        E_0 = ground_state_energy(*instance.as_tuple())

    ansatz, cost_hamiltonian = build_qaoa_circuit_for_instance(instance, p=p)

    t0 = time.time()
    best_result, all_results = run_qaoa_multi_restart(
        ansatz,
        cost_hamiltonian,
        optimizer_method=optimizer_method,
        device=device,
        n_restarts=n_restarts,
        minimize_options=minimize_options,
        rng=rng,
        seed=seed,
        warm_start_fn=warm_start_fn,
    )
    wall_time_s = time.time() - t0

    epsilon = relative_energy_error(best_result.optimal_energy, E_0)

    return SweepRecord(
        instance_name=instance.name,
        n_spins=instance.n_spins,
        boundary=instance.boundary,
        instance_description=instance.description,
        p=p,
        optimizer_method=optimizer_method,
        device=device,
        n_restarts=n_restarts,
        E_0=E_0,
        best_energy=best_result.optimal_energy,
        epsilon=epsilon,
        best_params=best_result.optimal_params,
        restart_energies=[r.optimal_energy for r in all_results],
        success=best_result.success,
        message=best_result.message,
        n_function_evals_total=sum(r.n_function_evals for r in all_results),
        wall_time_s=wall_time_s,
    )


# --------------------------------------------------------------------------
# Full depth sweep
# --------------------------------------------------------------------------


def run_depth_sweep(
    instances: Optional[Union[Dict[str, IsingInstance], Iterable[IsingInstance]]] = None,
    p_values: Iterable[int] = range(1, 6),
    optimizer_methods: Iterable[str] = VALID_OPTIMIZER_METHODS,
    device: str = "CPU",
    n_restarts: int = 5,
    minimize_options: Optional[dict] = None,
    seed: Optional[int] = None,
    warm_start_fn: Optional[Callable] = None,
    verbose: bool = False,
) -> List[SweepRecord]:
    """Run the full instance x p x optimizer depth sweep; one `SweepRecord`
    per combination, one independent RNG stream per point (`seed`)."""
    if instances is None:
        instance_map: Dict[str, IsingInstance] = generate_test_instances()
    elif isinstance(instances, dict):
        instance_map = instances
    else:
        instance_map = {inst.name: inst for inst in instances}

    p_list = list(p_values)
    optimizer_list = list(optimizer_methods)
    for method in optimizer_list:
        if method not in VALID_OPTIMIZER_METHODS:
            raise ValueError(
                f"optimizer_methods entries must be one of {VALID_OPTIMIZER_METHODS}, got {method!r}"
            )
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")

    e0_by_instance = {
        name: ground_state_energy(*inst.as_tuple()) for name, inst in instance_map.items()
    }

    combos = list(itertools.product(instance_map.items(), p_list, optimizer_list))
    seed_seq = np.random.SeedSequence(seed)
    child_seed_seqs = seed_seq.spawn(len(combos))

    records: List[SweepRecord] = []
    for ((name, instance), p, optimizer_method), child_seed_seq in zip(combos, child_seed_seqs):
        rng = np.random.default_rng(child_seed_seq)
        record = run_sweep_point(
            instance,
            p,
            optimizer_method=optimizer_method,
            device=device,
            n_restarts=n_restarts,
            minimize_options=minimize_options,
            rng=rng,
            warm_start_fn=warm_start_fn,
            E_0=e0_by_instance[name],
        )
        records.append(record)
        if verbose:
            print(
                f"[{name}] p={p} {optimizer_method}: best_energy={record.best_energy:.6f} "
                f"E_0={record.E_0:.6f} epsilon={record.epsilon:.6f} "
                f"({record.wall_time_s:.2f}s, {record.n_function_evals_total} fevals)"
            )

    return records


# --------------------------------------------------------------------------
# Conversion helper (for json.dump / pandas.DataFrame / plotting)
# --------------------------------------------------------------------------


def records_to_dicts(records: Iterable[SweepRecord]) -> List[dict]:
    """`SweepRecord`s -> plain-Python dicts (numpy arrays -> lists, numpy
    scalars -> float/int/bool), for `json.dump`/`pandas.DataFrame`."""
    out = []
    for r in records:
        out.append(
            {
                "instance_name": r.instance_name,
                "n_spins": int(r.n_spins),
                "boundary": r.boundary,
                "instance_description": r.instance_description,
                "p": int(r.p),
                "optimizer_method": r.optimizer_method,
                "device": r.device,
                "n_restarts": int(r.n_restarts),
                "E_0": float(r.E_0),
                "best_energy": float(r.best_energy),
                "epsilon": float(r.epsilon),
                "best_params": [float(x) for x in r.best_params],
                "restart_energies": [float(x) for x in r.restart_energies],
                "success": bool(r.success),
                "message": r.message,
                "n_function_evals_total": int(r.n_function_evals_total),
                "wall_time_s": float(r.wall_time_s),
            }
        )
    return out


# --------------------------------------------------------------------------
# Warm-start comparison runner (standard QAOA vs. Egger-WS vs. ND-AWS)
# --------------------------------------------------------------------------
#
# Compares `src.warm_start.egger_ws.run_standard_qaoa` / `run_egger_ws` /
# `nd_aws_to_comparison_result` head-to-head on the uniform AFM
# ring (`ising_model.generate_uniform_AFM_ring`), emitting one CSV row per (n, p).

VALID_WS_METHODS = ("standard", "egger_continuous", "egger_rounded", "nd_aws")

WS_CSV_COLUMNS = (
    ["n", "p", "topology", "E_gs", "E_sim", "delta_E", "P_gs"]
    + [f"C{i}" for i in range(1, 11)]
    + [f"gamma_{i}" for i in range(4)]
    + [f"beta_{i}" for i in range(4)]
    + ["method", "n_circuit_evals", "wall_time_s"]
)

DEFAULT_WS_CSV_FILENAME = "ising_results.csv"


class _CircuitCallCounter:
    """Shared mutable `.n_calls` tally, incremented by every
    `_CountingEstimator`/`_CountingSampler` routed through it."""

    def __init__(self) -> None:
        self.n_calls = 0


class _CountingEstimator:
    """Wraps an Estimator, counting `.run()` calls against a shared
    `_CircuitCallCounter`."""

    def __init__(self, inner, counter: _CircuitCallCounter) -> None:
        self._inner = inner
        self._counter = counter

    def run(self, *args, **kwargs):
        self._counter.n_calls += 1
        return self._inner.run(*args, **kwargs)


class _CountingSampler:
    """Same wrapping pattern as `_CountingEstimator`, for `Sampler`
    primitives (ND-AWS's bitstring-sampling step, `nd_aws.sample_bitstrings`)."""

    def __init__(self, inner, counter: _CircuitCallCounter) -> None:
        self._inner = inner
        self._counter = counter

    def run(self, *args, **kwargs):
        self._counter.n_calls += 1
        return self._inner.run(*args, **kwargs)


def _run_ws_method(
    method: str,
    instance: IsingInstance,
    p: int,
    device: str,
    optimizer_method: str,
    n_restarts: int,
    minimize_options: Optional[dict],
    seed: Optional[int],
    egger_variant: str,
    egger_eps: float,
    egger_relaxation_kwargs: Optional[dict],
    ndaws_kwargs: Optional[dict],
):
    """Dispatches to one of the three `warm_start.egger_ws` methods.
    Returns `(result, n_circuit_evals, notes)`."""
    from src.warm_start.egger_ws import (
        VALID_EGGER_VARIANTS,
        nd_aws_to_comparison_result,
        run_egger_ws,
        run_standard_qaoa,
    )
    from src.warm_start.nd_aws import build_default_sampler

    notes: Dict[str, str] = {}
    counter = _CircuitCallCounter()

    if method == "standard":
        estimator = _CountingEstimator(optimizer_module.build_estimator(device), counter)
        result = run_standard_qaoa(
            instance,
            p=p,
            optimizer_method=optimizer_method,
            device=device,
            n_restarts=n_restarts,
            minimize_options=minimize_options,
            seed=seed,
            estimator=estimator,
        )
    elif method in ("egger_continuous", "egger_rounded"):
        variant = "continuous" if method == "egger_continuous" else "rounded"
        if egger_variant not in VALID_EGGER_VARIANTS:
            raise ValueError(
                f"egger_variant must be one of {VALID_EGGER_VARIANTS}, got {egger_variant!r}"
            )
        estimator = _CountingEstimator(optimizer_module.build_estimator(device), counter)
        result = run_egger_ws(
            instance,
            p=p,
            variant=variant,
            eps=egger_eps,
            relaxation_kwargs=egger_relaxation_kwargs,
            optimizer_method=optimizer_method,
            device=device,
            n_restarts=n_restarts,
            minimize_options=minimize_options,
            seed=seed,
            estimator=estimator,
        )
    elif method == "nd_aws":
        kwargs = dict(ndaws_kwargs or {})
        kwargs.setdefault("seed", seed)
        kwargs.setdefault("device", device)
        ndaws_device = kwargs["device"]

        sampler_counter = _CircuitCallCounter()
        # An explicitly supplied sampler still wins,
        # otherwise default to the one matching the requested device.
        inner_sampler = kwargs.get("sampler") or build_default_sampler(ndaws_device)
        kwargs["sampler"] = _CountingSampler(inner_sampler, sampler_counter)
        kwargs["estimator"] = _CountingEstimator(
            optimizer_module.build_estimator(ndaws_device), counter
        )

        result = nd_aws_to_comparison_result(instance, p=p, ndaws_kwargs=kwargs)
        counter.n_calls += sampler_counter.n_calls
    else:
        raise ValueError(f"method must be one of {VALID_WS_METHODS}, got {method!r}")

    return result, counter.n_calls, notes


def run_warm_start_comparison_point(
    n_spins: int,
    p: int,
    method: str,
    J_value: float = 1.0,
    h_value: float = 0.0,
    device: str = "CPU",
    optimizer_method: str = "L-BFGS-B",
    n_restarts: int = 3,
    minimize_options: Optional[dict] = None,
    seed: Optional[int] = None,
    egger_variant: str = "continuous",
    egger_eps: float = 0.1,
    egger_relaxation_kwargs: Optional[dict] = None,
    ndaws_kwargs: Optional[dict] = None,
    assert_energy_correlator_identity: bool = True,
) -> dict:
    """One (n, p, method) CSV row on the uniform AFM ring; `method` in
    `VALID_WS_METHODS` dispatches via `_run_ws_method`. `p` must be `<= 4`."""
    if method not in VALID_WS_METHODS:
        raise ValueError(f"method must be one of {VALID_WS_METHODS}, got {method!r}")
    if not (1 <= p <= 4):
        raise ValueError(f"p must be in 1..4 to fit this CSV schema's fixed gamma/beta slots, got {p}")
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")

    instance = generate_uniform_AFM_ring(n_spins, J_value=J_value, h_value=h_value)
    J, h, boundary = instance.as_tuple()

    E_gs = ground_state_energy(J, h, boundary, n_spins)

    t0 = time.time()
    result, n_circuit_evals, notes = _run_ws_method(
        method,
        instance,
        p,
        device,
        optimizer_method,
        n_restarts,
        minimize_options,
        seed,
        egger_variant,
        egger_eps,
        egger_relaxation_kwargs,
        ndaws_kwargs,
    )
    wall_time_s = time.time() - t0

    E_sim = float(result.optimal_energy)
    delta_E = E_sim - E_gs
    P_gs = ground_state_success_probability(result.statevector, J, h, boundary, n_spins)
    correlators = two_point_correlators(result.statevector, n_spins, r_max=10)

    if abs(h_value) < 1e-12:
        expected_E = J_value * n_spins * float(correlators[0])
        if not np.isclose(E_sim, expected_E, atol=1e-6, rtol=1e-6):
            msg = (
                f"E_sim == n_spins * J_value * C_1 identity failed: "
                f"n={n_spins}, p={p}, method={method!r}: E_sim={E_sim}, "
                f"n*J*C_1={expected_E} (C_1={correlators[0]})"
            )
            if assert_energy_correlator_identity:
                raise AssertionError(msg)
            warnings.warn(msg, stacklevel=2)

    params = np.asarray(result.optimal_params, dtype=float)
    beta, gamma = params[:p], params[p : 2 * p]

    row: Dict[str, object] = {
        "n": int(n_spins),
        "p": int(p),
        "topology": boundary,
        "E_gs": float(E_gs),
        "E_sim": E_sim,
        "delta_E": float(delta_E),
        "P_gs": float(P_gs),
    }
    for i in range(10):
        row[f"C{i + 1}"] = float(correlators[i])
    for i in range(4):
        row[f"gamma_{i}"] = float(gamma[i]) if i < p else ""
    for i in range(4):
        row[f"beta_{i}"] = float(beta[i]) if i < p else ""
    row["method"] = method
    row["n_circuit_evals"] = int(n_circuit_evals)
    row["wall_time_s"] = float(wall_time_s)

    return row


def run_warm_start_comparison(
    n_values: Iterable[int],
    p_values: Iterable[int],
    method: str,
    seed: Optional[int] = None,
    verbose: bool = False,
    **point_kwargs,
) -> List[dict]:
    """Runs `run_warm_start_comparison_point` over every `(n, p)`, one
    independent RNG child per point; returns a flat `list[dict]` of rows."""
    combos = list(itertools.product(n_values, p_values))
    seed_seq = np.random.SeedSequence(seed)
    child_seed_seqs = seed_seq.spawn(len(combos))

    rows = []
    for (n_spins, p), child_seed_seq in zip(combos, child_seed_seqs):
        child_seed = int(child_seed_seq.generate_state(1)[0])
        row = run_warm_start_comparison_point(
            n_spins, p, method, seed=child_seed, **point_kwargs
        )
        rows.append(row)
        if verbose:
            notes = row.pop("_notes", None)
            print(
                f"[n={row['n']}, p={row['p']}, method={method}] "
                f"E_gs={row['E_gs']:.6f} E_sim={row['E_sim']:.6f} "
                f"P_gs={row['P_gs']:.6f} n_circuit_evals={row['n_circuit_evals']} "
                f"wall_time_s={row['wall_time_s']:.2f}"
                + (f"  [note: {notes}]" if notes else "")
            )
    for row in rows:
        row.pop("_notes", None)
    return rows


def _default_results_dir() -> str:
    """`<repo_root>/results`, derived from `__file__` so it's cwd-independent."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def write_warm_start_comparison_csv(
    rows: Iterable[dict],
    filename: str = DEFAULT_WS_CSV_FILENAME,
    results_dir: Optional[str] = None,
) -> str:
    """Writes `rows` to `<results_dir>/<filename>` in `WS_CSV_COLUMNS`
    order; `results_dir` defaults to `<repo_root>/results`. Returns the path."""
    if results_dir is None:
        results_dir = _default_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path
