"""Build the parameterized QAOA ansatz for a 1D Ising cost Hamiltonian

`H = sum_i J_i Z_i Z_(i+1) + sum_i h_i Z_i`

usable directly as an Estimator ansatz:
`estimator.run([(ansatz, cost_hamiltonian, params)])`.

Built with `qiskit.circuit.library.QAOAAnsatz`, which
evolves the cost-layer unitary via a `PauliEvolutionGate`.

- `J_i`/`h_i` are handled directly (no Max-Cut unit-weight assumption)
- No Trotter error. Every term is diagonal (Z or Z*Z), so all terms commute.
"""

from __future__ import annotations

from typing import Optional, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from src.hamiltonian import build_ising_hamiltonian

# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------


def build_qaoa_circuit(
    cost_hamiltonian: SparsePauliOp,
    p: int = 1,
    mixer_operator: Optional[SparsePauliOp] = None,
    initial_state: Optional[QuantumCircuit] = None,
    name: str = "QAOA",
) -> QuantumCircuit:
    """Build a depth-`p` QAOA ansatz for `cost_hamiltonian`.

    Defaults follow Qiskit's `QAOAAnsatz`:
    https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.QAOAAnsatz

    Parameters
    ----------
    cost_hamiltonian : SparsePauliOp
        `hamiltonian.build_ising_hamiltonian(J, h, boundary)`
    p : int
        QAOA depth
    mixer_operator, initial_state : optional
        Override the default mixer / initial state
    name : str
        Circuit name

    Returns
    -------
    QuantumCircuit
    """
    if p < 0:
        raise ValueError(f"p (QAOA depth) must be >= 0, got {p}")

    circuit = QAOAAnsatz(
        cost_operator=cost_hamiltonian,
        reps=p,
        initial_state=initial_state,
        mixer_operator=mixer_operator,
        name=name,
    )
    return circuit


def build_qaoa_circuit_from_ising(
    J,
    h,
    boundary: str = "OBC",
    p: int = 1,
    n_spins: Optional[int] = None,
    **kwargs,
) -> Tuple[QuantumCircuit, SparsePauliOp]:
    """Wrapper: build both the cost `SparsePauliOp` (via
    `hamiltonian.build_ising_hamiltonian`) and the matching QAOA ansatz
    circuit, for a `(J, h, boundary)` Ising instance.

    `n_spins` is inferred from `J`/`h`. If given, it is checked against the
    inferred size and a `ValueError` is raised on mismatch.

    Returns
    -------
    (QuantumCircuit, SparsePauliOp)
        The ansatz circuit and the cost Hamiltonian both pass
         to the Estimator: `estimator.run([(circuit, H, params)])`.
    """
    cost_hamiltonian = build_ising_hamiltonian(J, h, boundary)
    if n_spins is not None and cost_hamiltonian.num_qubits != n_spins:
        raise ValueError(
            f"n_spins={n_spins} does not match the size inferred from J/h "
            f"(got {cost_hamiltonian.num_qubits})"
        )
    circuit = build_qaoa_circuit(cost_hamiltonian, p=p, **kwargs)
    return circuit, cost_hamiltonian


def build_qaoa_circuit_for_instance(instance, p: int = 1, **kwargs) -> Tuple[QuantumCircuit, SparsePauliOp]:
    """Same as `build_qaoa_circuit_from_ising`, but takes an
    `ising_model.IsingInstance` directly."""
    return build_qaoa_circuit_from_ising(*instance.as_tuple(), p=p, **kwargs)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import numpy as np
    from qiskit.circuit.library import RZZGate, RZGate, RXGate
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.primitives import StatevectorEstimator

    from src import ising_model as im
    from src import exact_solver as es
    from src.hamiltonian import build_ising_hamiltonian

    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    rng = np.random.default_rng(0)

    # ---- 1. Circuit builds for p=1,2,3 with the expected parameter count --
    print("\n--- Circuit construction / parameter count checks ---")
    for instance in (im.TWO_SPIN_FM_INSTANCE, im.THREE_SPIN_AFM_INSTANCE):
        H = build_ising_hamiltonian(*instance.as_tuple())
        for p in (1, 2, 3):
            qc = build_qaoa_circuit(H, p=p)
            check(
                f"{instance.name} p={p}: builds, num_qubits == n_spins",
                qc.num_qubits == instance.n_spins,
                f"num_qubits={qc.num_qubits}",
            )
            check(
                f"{instance.name} p={p}: num_parameters == 2*p",
                qc.num_parameters == 2 * p,
                f"num_parameters={qc.num_parameters}, expected {2 * p}",
            )

    # ---- 2. p=0: equal-superposition state, 0 parameters ------
    print("\n--- p=0 (minimal depth) check ---")
    instance = im.THREE_SPIN_AFM_INSTANCE
    H = build_ising_hamiltonian(*instance.as_tuple())
    qc0 = build_qaoa_circuit(H, p=0)
    check("p=0: num_parameters == 0", qc0.num_parameters == 0, f"got {qc0.num_parameters}")
    check("p=0: num_qubits == n_spins", qc0.num_qubits == instance.n_spins)
    sv0 = Statevector(qc0.decompose(reps=5))
    uniform = np.full(2 ** instance.n_spins, 1.0 / np.sqrt(2 ** instance.n_spins))
    check(
        "p=0: statevector == uniform superposition (up to global phase)",
        np.allclose(np.abs(sv0.data), uniform, atol=1e-8),
        f"max|amp| deviation={np.max(np.abs(np.abs(sv0.data) - uniform))}",
    )
    est = StatevectorEstimator()
    ev0 = est.run([(qc0, H, [])]).result()[0].data.evs  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    mean_energy = float(np.mean(es.all_state_energies(*instance.as_tuple())))
    check(
        "p=0: Estimator expectation == mean over all basis-state energies",
        np.isclose(float(ev0), mean_energy, atol=1e-8),
        f"estimator={float(ev0)}, mean_energy={mean_energy}",
    )

    # ---- 3. Gate-level equivalence: QAOAAnsatz cost/mixer layers vs a
    #         manually constructed RZZ/RZ/RX circuit (w/ same angles & same unitary) -------
    print("\n--- QAOAAnsatz cost+mixer layer vs. explicit RZZ/RZ/RX gates (p=1) ---")
    instance = im.THREE_SPIN_AFM_INSTANCE  # H = Z0Z1 + Z1Z2, J=[1,1], h=[0,0,0]
    J, h, boundary = instance.as_tuple()
    n = instance.n_spins
    H = build_ising_hamiltonian(J, h, boundary)
    gamma, beta = 0.37, -0.81

    qc_ansatz = build_qaoa_circuit(H, p=1)
    # bind order: circuit.parameters is sorted alphabetically -> beta first
    param_order = list(qc_ansatz.parameters)
    values = {}
    for prm in param_order:
        values[prm] = beta if str(prm).startswith("β") or str(prm).startswith("beta") else gamma
    qc_ansatz_bound = qc_ansatz.assign_parameters(values)
    u_ansatz = Operator(qc_ansatz_bound.decompose(reps=6))

    # H^{\otimes n}, then RZZ(2*gamma*J_k) per bond, RZ(2*gamma*h_i)
    # per field term, then RX(2*beta) per qubit.
    qc_manual = QuantumCircuit(n)
    for q in range(n):
        qc_manual.h(q)
    pairs = [(i, i + 1) for i in range(n - 1)] if boundary == "OBC" else [(i, (i + 1) % n) for i in range(n)]
    for k, (i, j) in enumerate(pairs):
        qc_manual.append(RZZGate(2 * gamma * J[k]), [i, j])
    for i in range(n):
        if h[i] != 0.0:
            qc_manual.append(RZGate(2 * gamma * h[i]), [i])
    for q in range(n):
        qc_manual.append(RXGate(2 * beta), [q])
    u_manual = Operator(qc_manual)

    check(
        "QAOAAnsatz(p=1) unitary == hand-built RZZ/RZ/RX circuit unitary (up to global phase)",
        u_ansatz.equiv(u_manual),
        "Operator.equiv() (global-phase-insensitive) comparison",
    )

    # ---- 4. End-to-end Estimator correctness: expectation values bounded
    #         between the analytic ground and maximum energy --------------
    print("\n--- End-to-end Estimator expectation-value bounds ---")
    for instance in (im.TWO_SPIN_FM_INSTANCE, im.THREE_SPIN_AFM_INSTANCE, im.THREE_SPIN_FRUSTRATED_RING_INSTANCE):
        J, h, boundary = instance.as_tuple()
        H = build_ising_hamiltonian(J, h, boundary)
        energies = es.all_state_energies(J, h, boundary)
        e_min, e_max = float(np.min(energies)), float(np.max(energies))
        for p in (1, 2, 3):
            circuit = build_qaoa_circuit(H, p=p)
            n_params = circuit.num_parameters
            for trial in range(5):
                params = rng.uniform(-np.pi, np.pi, size=n_params)
                ev = float(est.run([(circuit, H, params)]).result()[0].data.evs)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
                in_bounds = (e_min - 1e-6) <= ev <= (e_max + 1e-6)
                check(
                    f"{instance.name} p={p} trial={trial}: E_min <= <H> <= E_max",
                    in_bounds,
                    f"<H>={ev:.6f}, bounds=[{e_min:.6f}, {e_max:.6f}]",
                )

    # ---- 5. Optimization test (random search) -----------
    print("\n--- Optimization test (random search, not optimizer.py) ---")
    from scipy.optimize import minimize

    for instance, e0_expected in (
        (im.TWO_SPIN_FM_INSTANCE, -1.0),
        (im.THREE_SPIN_AFM_INSTANCE, -2.0),
    ):
        J, h, boundary = instance.as_tuple()
        H = build_ising_hamiltonian(J, h, boundary)
        p = 3
        circuit = build_qaoa_circuit(H, p=p)

        def cost_fn(params):
            return float(est.run([(circuit, H, params)]).result()[0].data.evs)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

        best = None
        x0 = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)
        result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 150})
        best = result.fun
        gap = best - e0_expected
        check(
            f"{instance.name} p={p}: COBYLA (150 iters) gets within 0.15 of E_0={e0_expected}",
            gap < 0.15,
            f"best found={best:.6f}, gap={gap:.6f}",
        )

    print("\n" + ("=" * 60))
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
