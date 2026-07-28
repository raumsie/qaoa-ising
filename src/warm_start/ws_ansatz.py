"""
ws_ansatz.py
============

Warm-Start QAOA (WS-QAOA) initial state, mixer, and bitstring-derived bias,
generalized as in ND-AWS.

Methods implemented:
- WS-QAOA initial state / mixer (Eq. 6-7 of ND-AWS; originally Eq. A5-A8 of
  Egger, Marecek, Woerner, "Warm-starting quantum optimization," Quantum 5,
  479 (2021), arXiv:2009.10095).
- Bias-from-bitstring generalization (Eqs. A9-A11 of ND-AWS):
  F. B. Maciejewski, et al., "Quantum Approximate Optimization
  via Noise-Directed Adaptive Warm-Starting," arXiv:2607.09368 (2026).

Independent reimplementation written directly from the papers. Not derived
from quapopt, the ND-AWS authors' reference implementation:
https://github.com/usra-riacs/quantum-approximate-optimization

Built on top of `qaoa_circuit.build_qaoa_circuit`'s `mixer_operator` and
`initial_state` parameters.

Eq. (6): |c> = RY(theta_c)|0> = sqrt(1-c)|0> + sqrt(c)|1>, theta_c =
2*arcsin(sqrt(c)); full state |c>^{tensor n}; c in [0, 1]

Eq. (7): H_M = sum_i H_M^i, H_M^i = 2*sqrt(c_i(1-c_i))*X_i + (1-2*c_i)*Z_i.

The initial state |c> is chosen as the exact ground state of H_M, to
preserve the mixer/initial-state alignment that gives WS-QAOA its
adiabatic performance guarantee (Zichang He, et al., "Alignment between initial state and
mixer improves QAOA performance for constrained optimization," arXiv:2305.03857v3 (2024)).

Why the mixer is built as a QuantumCircuit, not a SparsePauliOp: X_i and Z_i
on the same qubit don't commute. Handing Eq. (7)'s literal `SparsePauliOp`
to `QAOAAnsatz` would compile it via first-order Lie-Trotter, as two
separate sequential gates -- an approximation of the true single-qubit
rotation. That would silently break the "|c> is H_M's exact ground state"
guarantee. `ws_mixer_circuit` instead builds that exact rotation directly,
via the `mixer_operator` parameter's existing `QuantumCircuit` support (no change to
`qaoa_circuit.py`). `ws_mixer_hamiltonian` (a real `SparsePauliOp`) is kept
for testing.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from src.qaoa_circuit import build_qaoa_circuit

CLike = Union[float, np.ndarray]


# --------------------------------------------------------------------------
# Broadcasting helper
# --------------------------------------------------------------------------


def _broadcast_c(c: CLike, n_spins: Optional[int] = None) -> np.ndarray:
    """Scalar `c` (same bias every qubit) or a length-`n_spins` array."""
    if np.isscalar(c):
        if n_spins is None:
            raise ValueError("n_spins is required when c is given as a scalar")
        arr = np.full(n_spins, float(c))
    else:
        arr = np.asarray(c, dtype=float)
        if n_spins is not None and len(arr) != n_spins:
            raise ValueError(f"len(c)={len(arr)} does not match n_spins={n_spins}")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"c must be in [0, 1] elementwise, got {arr}")
    return arr


# --------------------------------------------------------------------------
# Eq. (6): WS initial state |c>
# --------------------------------------------------------------------------


def ws_bias_state(c: CLike, n_spins: Optional[int] = None, name: str = "WS-init") -> QuantumCircuit:
    """Eq. (6)/(A5-A6): `|c> = RY(theta_c)|0>`.

        theta_c = 2*arcsin(sqrt(c))
    """
    c_arr = _broadcast_c(c, n_spins)
    n = len(c_arr)
    qc = QuantumCircuit(n, name=name)
    for i in range(n):
        theta = 2.0 * float(np.arcsin(np.sqrt(c_arr[i])))
        qc.ry(theta, i)
    return qc


# --------------------------------------------------------------------------
# Eq. (7): WS mixer H_M -- SparsePauliOp (for testing) and the exact
# QuantumCircuit realization actually used as the mixer_operator parameter
# --------------------------------------------------------------------------


def ws_mixer_hamiltonian(c: CLike, n_spins: Optional[int] = None) -> SparsePauliOp:
    """Eq. (7)/(A7-A8) as a `SparsePauliOp` -- testing only,
    NOT the actual ansatz mixer (Trotter bug).

        H_M = sum_i H_M^i,  H_M^i = 2*sqrt(c_i(1-c_i))*X_i + (1-2*c_i)*Z_i
    """
    c_arr = _broadcast_c(c, n_spins)
    n = len(c_arr)
    pauli_list = []
    for i in range(n):
        ci = float(c_arr[i])
        a_i = 2.0 * np.sqrt(ci * (1.0 - ci))
        b_i = 1.0 - 2.0 * ci
        if a_i != 0.0:
            label = ["I"] * n
            label[n - 1 - i] = "X"
            pauli_list.append(("".join(label), a_i))
        if b_i != 0.0:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            pauli_list.append(("".join(label), b_i))
    if not pauli_list:
        pauli_list = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(pauli_list).simplify()


def ws_mixer_circuit(c: CLike, n_spins: Optional[int] = None, name: str = "WS-mixer") -> QuantumCircuit:
    """Exact single-qubit realization of `exp(-i*beta*H_M)` for Eq. (7)'s
    `H_M` -- this is what should be passed as the `mixer_operator` parameter."""
    c_arr = _broadcast_c(c, n_spins)
    n = len(c_arr)
    beta = Parameter("beta_ws")
    qc = QuantumCircuit(n, name=name)
    for i in range(n):
        ci = float(c_arr[i])
        a_i = 2.0 * np.sqrt(ci * (1.0 - ci))
        b_i = 1.0 - 2.0 * ci
        r_i = float(np.hypot(a_i, b_i))
        # r_i == 0 is impossible -- guarded against anyway.
        if r_i == 0.0:
            continue
        phi_i = float(np.arctan2(a_i, b_i))
        qc.ry(-phi_i, i)
        qc.rz(2.0 * r_i * beta, i)
        qc.ry(phi_i, i)
    return qc


# --------------------------------------------------------------------------
# Eqs. (A9)-(A11): bias vector from a target bitstring
# --------------------------------------------------------------------------


def bias_from_bitstring(x, t: CLike) -> np.ndarray:
    """Eqs. (A9)-(A11): generalizes the continuous-relaxation bias `c` to a
    bitstring-derived bias. Raises on `t_i == 0` (App. A3 triviality).

        c_i = f_i(x_i) = t_i        if x_i == 0
                       = 1 - t_i    if x_i == 1
    """
    x_bits = np.asarray(x, dtype=np.int64)
    n = len(x_bits)
    if np.isscalar(t):
        t_arr = np.full(n, float(t))
    else:
        t_arr = np.asarray(t, dtype=float)
        if len(t_arr) != n:
            raise ValueError(f"len(t)={len(t_arr)} does not match len(x)={n}")

    if np.any(t_arr < 0.0) or np.any(t_arr > 0.5):
        raise ValueError(f"t_i must be in [0, 0.5] (Eq. A11), got t={t_arr}")
    if np.any(t_arr == 0.0):
        raise ValueError(
            "t_i == 0 makes the WS ansatz logically trivial on that qubit "
            "(App. A3: the mixer becomes diagonal, the whole circuit just "
            "adds a global phase to the fixed input state |x_i>) -- never "
            "use t_i = 0; pick a small positive value instead."
        )

    return np.where(x_bits == 0, t_arr, 1.0 - t_arr)


# --------------------------------------------------------------------------
# Full ansatz builder
# --------------------------------------------------------------------------


def build_ws_ansatz(
    phase_separator_hamiltonian: SparsePauliOp,
    c: CLike,
    p: int = 1,
    n_spins: Optional[int] = None,
    name: str = "WS-QAOA",
) -> QuantumCircuit:
    """Build the WS-QAOA ansatz (Eq. 2) for a given phase-separator/cost
    Hamiltonian and bias `c`; `c=0.5` recovers standard QAOA exactly."""
    n = n_spins if n_spins is not None else phase_separator_hamiltonian.num_qubits
    mixer = ws_mixer_circuit(c, n_spins=n)
    init = ws_bias_state(c, n_spins=n)
    return build_qaoa_circuit(
        phase_separator_hamiltonian,
        p=p,
        mixer_operator=mixer,
        initial_state=init,
        name=name,
    )
