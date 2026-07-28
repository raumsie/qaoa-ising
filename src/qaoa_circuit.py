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
    """Build a depth-`p` QAOA ansatz for `cost_hamiltonian` via Qiskit's `QAOAAnsatz`."""
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
    """Wrapper: build both the cost `SparsePauliOp` and the matching QAOA
    ansatz circuit, for a `(J, h, boundary)` Ising instance."""
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
