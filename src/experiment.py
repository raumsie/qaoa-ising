"""Depth-sweep, optimizer, & coupling experiment runner.

For each (Ising instance, QAOA depth `p`, optimizer method), runs a
multi-restart QAOA optimization (`optimizer.run_qaoa_multi_restart`) and
records the best energy found against the exact ground-state energy `E_0`
(`exact_solver.ground_state_energy`) as the relative energy error:

    epsilon(p) = (E_qaoa(p) - E_0) / |E_0|

Only calls into `ising_model.py`/`exact_solver.py`/`qaoa_circuit.py`/
`optimizer.py` -- no instance generation, Hamiltonian/circuit construction,
or optimizer logic lives here. `device` and `warm_start_fn` (inert) are
passed straight through to `run_qaoa_multi_restart`.

`run_depth_sweep` returns a `list[SweepRecord]`, one per (instance, p,
optimizer_method) combination, containing everything a plotting script needs
(`epsilon(p)` vs `p`, per-restart energies, cost/timing, diagnostics).
`records_to_dicts` converts that to plain-Python dicts for
`json.dump`/`pandas.DataFrame`.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from src.exact_solver import ground_state_energy
from src.ising_model import IsingInstance, generate_test_instances
from src.qaoa_circuit import build_qaoa_circuit_for_instance
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
    """One (instance, p, optimizer_method) point of the depth sweep.

    Everything a plotting script needs for an `epsilon(p)` vs. `p` curve
    (per instance, per optimizer).
    """

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
    """`epsilon = (E_qaoa - E_0) / |E_0|`

    `E_0 == 0` makes the relative error undefined (division by zero); this
    is not expected for any instance in `ising_model.generate_test_instances`
    (couplings are never all-zero), but is still checked for,
    since a degenerate all-zero-coupling/all-zero-field instance is
    still constructible by hand.
    """
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
    """Run one (instance, p, optimizer_method) point of the depth sweep.

    Builds a fresh `(ansatz, cost_hamiltonian)` pair via
    `qaoa_circuit.build_qaoa_circuit_for_instance(instance, p=p)`, runs
    `optimizer.run_qaoa_multi_restart()` with `device`/`warm_start_fn`
    forwarded, and records the result against `E_0` (computed via
    `exact_solver.ground_state_energy` if not supplied).

    Returns
    -------
    SweepRecord
    """
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
    """Run the full instance x p x optimizer depth sweep.

    Parameters
    ----------
    instances : dict[str, IsingInstance], iterable of IsingInstance, or None
        Defaults to `ising_model.generate_test_instances()`.
    p_values : iterable of int
        QAOA depths to sweep. Design spec default is `1..5`
        (`range(1, 6)`).
    optimizer_methods : iterable of str
        Optimizer methods to sweep, each one of
        `optimizer.VALID_OPTIMIZER_METHODS` (`"COBYLA"`, `"L-BFGS-B"`).
        Defaults to both.
    device : str
        `"CPU"` or `"GPU"`, forwarded to
        `optimizer.run_qaoa_multi_restart` for every sweep point. Device
        choice is not decided here.
    n_restarts : int
        Random restarts per (instance, p, optimizer) point, forwarded to
        `run_qaoa_multi_restart`. Default is 5.
    minimize_options : dict, optional
        Forwarded to `scipy.optimize.minimize` via
        `optimizer.optimize_qaoa`.
    seed : int, optional
        Seeds a `numpy.random.SeedSequence`
    warm_start_fn : callable, optional
        Forwarded to `run_qaoa_multi_restart` for every sweep
        point. Unused within `run_qaoa_multi_restart` this pass (see
        `optimizer.py`).
    verbose : bool
        If True, print a one-line progress/result summary per sweep point
        (best energy, epsilon, wall time) as it completes.

    Returns
    -------
    list[SweepRecord]
        One record per (instance, p, optimizer_method) combination, in
        the order: outer loop over instances (dict/iteration order),
        then `p_values`, then `optimizer_methods`.
    """
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
# Plain-Python conversion helper (for json.dump / pandas.DataFrame / plotting)
# --------------------------------------------------------------------------


def records_to_dicts(records: Iterable[SweepRecord]) -> List[dict]:
    """Convert `SweepRecord`s to plain-Python dicts (numpy arrays -> lists,
    numpy scalars -> float/int/bool) -- directly usable as
    `json.dump(records_to_dicts(records), f)` or
    `pandas.DataFrame(records_to_dicts(records))`.
    """
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
# Self-check (small-scale -- NOT the full design-spec sweep,
# exercises the same code path at reduced p/restarts/instances)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src import ising_model as im
    from src import exact_solver as es

    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    # ---- 0. relative_energy_error arithmetic check -----------------------
    print("\n--- relative_energy_error arithmetic check ---")
    check(
        "relative_energy_error(-1.7, -2.0) == (-1.7 - -2.0)/abs(-2.0) == 0.15",
        np.isclose(relative_energy_error(-1.7, -2.0), 0.15, atol=1e-12),
        f"got {relative_energy_error(-1.7, -2.0)}",
    )
    check(
        "relative_energy_error(E_0, E_0) == 0",
        np.isclose(relative_energy_error(-2.0, -2.0), 0.0, atol=1e-12),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_nan = relative_energy_error(1.0, 0.0)
        check(
            "relative_energy_error(_, 0.0) warns and returns nan (guarded, not a crash)",
            len(w) == 1 and np.isnan(result_nan),
            f"warnings={len(w)}, result={result_nan}",
        )

    # ---- 1. Small-scale depth sweep on two easy analytic instances --------
    # NOT the full design-spec sweep --
    # reduced p range and restart count, but enough to exercise run_depth_sweep's full
    # code path and see a real epsilon(p) -> 0 trend on instances QAOA should solve easily.
    print("\n--- Small-scale depth sweep: TWO_SPIN_FM (E_0=-1.0), THREE_SPIN_AFM (E_0=-2.0) ---")
    easy_instances = {
        "two_spin_FM": im.TWO_SPIN_FM_INSTANCE,
        "three_spin_AFM": im.THREE_SPIN_AFM_INSTANCE,
    }
    P_VALUES = (1, 2, 3)
    N_RESTARTS = 3
    t_sweep0 = time.time()
    records = run_depth_sweep(
        instances=easy_instances,
        p_values=P_VALUES,
        optimizer_methods=("COBYLA", "L-BFGS-B"),
        device="CPU",
        n_restarts=N_RESTARTS,
        minimize_options={"maxiter": 200},
        seed=7,
        warm_start_fn=None,
        verbose=True,
    )
    t_sweep = time.time() - t_sweep0
    print(f"[INFO] small-scale sweep took {t_sweep:.2f}s total for {len(records)} records")

    n_expected = len(easy_instances) * len(P_VALUES) * 2
    check(
        f"run_depth_sweep returns instances x p x optimizers == {n_expected} records",
        len(records) == n_expected,
        f"got {len(records)}",
    )

    # ---- 2. E_0 cross-check: sweep's E_0 vs. an independently-recomputed ------
    print("\n--- E_0 cross-check (sweep-recorded vs. independently recomputed) ---")
    expected_e0 = {"two_spin_FM": -1.0, "three_spin_AFM": -2.0}
    for name, instance in easy_instances.items():
        e0_independent = es.ground_state_energy(*instance.as_tuple())
        recs_for_instance = [r for r in records if r.instance_name == name]
        e0_in_records = {r.E_0 for r in recs_for_instance}
        check(
            f"{name}: exact_solver.ground_state_energy (independent call) == documented E_0={expected_e0[name]}",
            np.isclose(e0_independent, expected_e0[name], atol=1e-10),
            f"got {e0_independent}",
        )
        check(
            f"{name}: every sweep record's E_0 == independently-recomputed E_0 (single consistent value)",
            len(e0_in_records) == 1 and np.isclose(e0_in_records.pop(), e0_independent, atol=1e-10),
            f"E_0 values seen in records={ {r.E_0 for r in recs_for_instance} }",
        )

    # ---- 3. epsilon(p) trend: report actual numbers, confirm it heads toward 0 as p increases -----
    print("\n--- epsilon(p) trend (best-of-optimizers per p) ---")
    for name in easy_instances:
        print(f"  instance={name}")
        eps_by_p = {}
        for p in P_VALUES:
            recs_p = [r for r in records if r.instance_name == name and r.p == p]
            best_eps = min(r.epsilon for r in recs_p)
            eps_by_p[p] = best_eps
            per_optimizer = ", ".join(
                f"{r.optimizer_method}: eps={r.epsilon:.6f} (best_energy={r.best_energy:.6f})"
                for r in sorted(recs_p, key=lambda r: r.optimizer_method)
            )
            print(f"    p={p}: {per_optimizer}  -> best_eps={best_eps:.6f}")
        check(
            f"{name}: epsilon(p) reaches (near-)0 by p={P_VALUES[-1]} (best-of-optimizers <= 0.05)",
            eps_by_p[P_VALUES[-1]] <= 0.05,
            f"eps by p={eps_by_p}",
        )
        check(
            f"{name}: epsilon is non-negative (best_energy can't beat E_0, check on the exact-baseline direction)",
            all(v >= -1e-6 for v in eps_by_p.values()),
            f"eps by p={eps_by_p}",
        )

    # ---- 4. Checked epsilon arithmetic for one real record ----------
    print("\n--- Checked epsilon arithmetic ---")
    sample = records[0]
    hand_epsilon = (sample.best_energy - sample.E_0) / abs(sample.E_0)
    check(
        f"record[0] ({sample.instance_name}, p={sample.p}, {sample.optimizer_method}): "
        f"computed (E_qaoa - E_0)/|E_0| == record.epsilon",
        np.isclose(hand_epsilon, sample.epsilon, atol=1e-12),
        f"best_energy={sample.best_energy}, E_0={sample.E_0}, "
        f"computed epsilon={hand_epsilon}, record.epsilon={sample.epsilon}",
    )

    # ---- 5. Record schema spot-check: every field a plotting script would -
    #         need is present and correctly typed/shaped.
    print("\n--- Record schema spot-check ---")
    r = records[0]
    check("record.instance_name is str", isinstance(r.instance_name, str), f"{r.instance_name!r}")
    check("record.n_spins is int-like and > 0", int(r.n_spins) > 0, f"{r.n_spins}")
    check("record.boundary in ('OBC', 'PBC')", r.boundary in ("OBC", "PBC"), f"{r.boundary!r}")
    check("record.p is int-like and > 0", int(r.p) > 0, f"{r.p}")
    check(
        "record.optimizer_method in VALID_OPTIMIZER_METHODS",
        r.optimizer_method in VALID_OPTIMIZER_METHODS,
        f"{r.optimizer_method!r}",
    )
    check("record.device == 'CPU' (as requested)", r.device == "CPU", f"{r.device!r}")
    check(
        "record.n_restarts == N_RESTARTS requested",
        int(r.n_restarts) == N_RESTARTS,
        f"{r.n_restarts}",
    )
    check("record.E_0 is finite float", np.isfinite(r.E_0), f"{r.E_0}")
    check("record.best_energy is finite float", np.isfinite(r.best_energy), f"{r.best_energy}")
    check("record.epsilon is finite float", np.isfinite(r.epsilon), f"{r.epsilon}")
    check(
        "record.best_params has length == 2*p (beta+gamma per QAOA layer)",
        len(r.best_params) == 2 * r.p,
        f"len={len(r.best_params)}, expected {2 * r.p}",
    )
    check(
        "record.restart_energies has length == n_restarts",
        len(r.restart_energies) == r.n_restarts,
        f"len={len(r.restart_energies)}, expected {r.n_restarts}",
    )
    check(
        "record.best_energy == min(restart_energies) (best-of-restarts is actually the min)",
        np.isclose(r.best_energy, min(r.restart_energies), atol=1e-12),
        f"best_energy={r.best_energy}, min(restart_energies)={min(r.restart_energies)}",
    )
    check("record.success is bool", isinstance(r.success, bool), f"{type(r.success)}")
    check("record.message is str", isinstance(r.message, str), f"{r.message!r}")
    check(
        "record.n_function_evals_total > 0",
        int(r.n_function_evals_total) > 0,
        f"{r.n_function_evals_total}",
    )
    check("record.wall_time_s >= 0", r.wall_time_s >= 0, f"{r.wall_time_s}")

    # ---- 6. records_to_dicts round-trips cleanly (plain-Python, no numpy) -
    print("\n--- records_to_dicts conversion sanity ---")
    dicts = records_to_dicts(records)
    check("records_to_dicts returns same length as records", len(dicts) == len(records))
    d0 = dicts[0]
    check(
        "records_to_dicts: best_params is a plain list of floats (not ndarray)",
        isinstance(d0["best_params"], list) and all(isinstance(x, float) for x in d0["best_params"]),
        f"type={type(d0['best_params'])}",
    )
    check(
        "records_to_dicts: epsilon value matches source record",
        np.isclose(d0["epsilon"], records[0].epsilon, atol=1e-12),
    )
    import json

    try:
        json.dumps(dicts)
        check("records_to_dicts output is JSON-serializable", True)
    except TypeError as exc:
        check("records_to_dicts output is JSON-serializable", False, str(exc))

    # ---- 7. device/warm_start_fn are pure pass-throughs, not decided here -
    print("\n--- device / warm_start_fn pass-through sanity ---")
    passed_warm_start_calls = []

    def _tracking_warm_start_fn(**kwargs):
        # Deliberately never actually used by run_qaoa_multi_restart this pass (see optimizer.py).
        # This just confirms experiment.py wires the argument through without breaking.
        passed_warm_start_calls.append(kwargs)
        return None

    rec_ws = run_sweep_point(
        im.TWO_SPIN_FM_INSTANCE,
        p=1,
        optimizer_method="COBYLA",
        device="CPU",
        n_restarts=2,
        minimize_options={"maxiter": 50},
        seed=0,
        warm_start_fn=_tracking_warm_start_fn,
    )
    check(
        "run_sweep_point accepts warm_start_fn without error (pass-through, unused downstream this pass)",
        np.isfinite(rec_ws.best_energy),
        f"best_energy={rec_ws.best_energy}",
    )
    check("record.device reflects the requested device ('CPU')", rec_ws.device == "CPU")

    try:
        run_sweep_point(im.TWO_SPIN_FM_INSTANCE, p=1, device="TPU", n_restarts=1)
        check("run_sweep_point(device='TPU') raises ValueError", False)
    except ValueError:
        check("run_sweep_point(device='TPU') raises ValueError", True)

    try:
        run_sweep_point(im.TWO_SPIN_FM_INSTANCE, p=1, optimizer_method="SPSA", n_restarts=1)
        check("run_sweep_point(optimizer_method='SPSA') raises ValueError", False)
    except ValueError:
        check("run_sweep_point(optimizer_method='SPSA') raises ValueError", True)

    print("\n" + ("=" * 60))
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
