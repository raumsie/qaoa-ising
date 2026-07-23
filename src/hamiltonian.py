"""
hamiltonian.py
==============

Converts a 1D Ising instance `(J, h, boundary)` into a
`qiskit.quantum_info.SparsePauliOp` cost Hamiltonian:

    H = sum_i J_i * Z_i * Z_(i+1)  +  sum_i h_i * Z_i

Also provides an independent (no `SparsePauliOp`) energy
function for a single bitstring, for cross-checking against
`exact_solver.bitstring_energy`.

Convention (must match `ising_model.py` and `exact_solver.py`):

- `h` has length `n_spins`. `J` has length `n_spins - 1` for open boundary
  conditions (bonds `0-1, 1-2, ..., (n-2)-(n-1)`), or length `n_spins` for
  periodic boundary conditions (adds a wraparound bond `(n-1, 0)` as the
  last entry, `J[n_spins - 1]`).
- `boundary` is a string, either `"OBC"` (open) or `"PBC"` (periodic) --
  matching `ising_model.py`'s and `exact_solver.py`'s `VALID_BOUNDARIES`.
- Qubit/bit ordering matches Qiskit's little-endian convention: for basis
  state integer index `k`, qubit `i` is bit `i` of `k` (qubit 0 = least
  significant bit). `SparsePauliOp` Pauli labels are read left-to-right as
  qubit `n-1 ... 0`, so a `Z` on qubit `i` sits at label position
  `n - 1 - i`.
- Spin convention: bit == 0 -> Z eigenvalue +1 ("spin up"); bit == 1 ->
  Z eigenvalue -1 ("spin down"). Since qiskit's `Z|0> = +|0>`, the `Z_i`
  Pauli operator's eigenvalue *is* the classical spin, so no extra sign
  flip is needed when building the Hamiltonian.
- Sign convention: `J_i < 0` favors ferromagnetic (aligned neighbors),
  `J_i > 0` favors antiferromagnetic (anti-aligned neighbors).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from qiskit.quantum_info import SparsePauliOp

VALID_BOUNDARIES = ("OBC", "PBC")


def _normalize_boundary(boundary: str) -> bool:
    """Return True if PBC, False if OBC.
    Raise ValueError for anything else."""
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got {boundary!r}")
    return boundary == "PBC"


def _infer_n_spins(J: np.ndarray, h: np.ndarray, periodic: bool) -> int:
    h = np.asarray(h)
    J = np.asarray(J)
    n = len(h)
    expected_J_len = n if periodic else n - 1
    if len(J) != expected_J_len:
        raise ValueError(
            f"boundary={'PBC' if periodic else 'OBC'!r} with n_spins={n} "
            f"(inferred from len(h)) expects len(J) == {expected_J_len}, got {len(J)}"
        )
    return n


def _boundary_pairs(n_spins: int, periodic: bool):
    """(i, j) index pairs for each coupling term."""
    if periodic:
        return [(i, (i + 1) % n_spins) for i in range(n_spins)]
    return [(i, i + 1) for i in range(n_spins - 1)]


# --------------------------------------------------------------------------
# SparsePauliOp construction
# --------------------------------------------------------------------------


def build_ising_hamiltonian(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC") -> SparsePauliOp:
    """Build the Ising cost Hamiltonian as a `SparsePauliOp`.

        H = sum_i J_i * Z_i * Z_(i+1)  +  sum_i h_i * Z_i

    Parameters
    ----------
    J :
        Coupling array. Length `n_spins - 1` for open boundary, `n_spins`
        for periodic boundary (last entry is the wraparound bond
        `(n_spins - 1, 0)`).
    h :
        Local field array, length `n_spins`.
    boundary : str
        Either `"OBC"` or `"PBC"`.

    Builds one `(label, coeff)` Pauli term per nonzero `J`/`h` entry:
    1. Coupling terms: for each bond `(i, j)` from `_boundary_pairs`, write
       `Z` at label positions `n-1-i` and `n-1-j`. `SparsePauliOp` labels
       read left-to-right as qubit `n-1 ... 0`, so qubit `i` sits at `n-1-i`.
    2. Field terms: same reversal, one `Z` at `n-1-i` per nonzero `h[i]`.
    3. Zero-coefficient terms are skipped, and all-zero `J`/`h` falls back to
       a single zero-coefficient identity term (`SparsePauliOp` needs >= 1).
    4. `SparsePauliOp.from_list(pauli_list).simplify()` combines/cancels
       duplicate terms and drops near-zero coefficients.

    Returns
    -------
    SparsePauliOp
        Directly usable as an Estimator-primitive observable.
    """
    periodic = _normalize_boundary(boundary)
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    n = _infer_n_spins(J, h, periodic)
    if n < 1:
        raise ValueError(f"n_spins must be >= 1, got {n}")

    pauli_list = []

    for k, (i, j) in enumerate(_boundary_pairs(n, periodic)):
        coeff = float(J[k])
        if coeff == 0.0:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        label[n - 1 - j] = "Z"
        pauli_list.append(("".join(label), coeff))

    for i in range(n):
        coeff = float(h[i])
        if coeff == 0.0:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        pauli_list.append(("".join(label), coeff))

    if not pauli_list:
        # All-zero Hamiltonian (J == 0 and h == 0): SparsePauliOp needs at
        # least one term, so fall back to a zero-coefficient identity.
        pauli_list = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list).simplify()


# --------------------------------------------------------------------------
# Classical energy of a single bitstring (independent of SparsePauliOp)
# --------------------------------------------------------------------------


def bitstring_energy(
    bits: Union[str, Sequence[int]],
    J: np.ndarray,
    h: np.ndarray,
    boundary: str = "OBC",
    n_spins: Optional[int] = None,
) -> float:
    """Classical energy of a single spin configuration, computed directly
    (no `SparsePauliOp`/qiskit involved) : for cross-checking against
    `exact_solver.bitstring_energy` and against
    `build_ising_hamiltonian(...).to_matrix()`/Estimator expectation values.

    Parameters
    ----------
    bits : sequence of int (0/1) or str of ("0"/"1"), length n_spins
        `bits[i]` is qubit `i`'s bit value (index 0 = qubit 0), matching
        `exact_solver.bitstring_energy`'s convention
        (i.e. NOT a Qiskit-style reversed bitstring).
    boundary : str
        `"OBC"` or `"PBC"`.
    """
    periodic = _normalize_boundary(boundary)
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    n = n_spins if n_spins is not None else _infer_n_spins(J, h, periodic)

    if isinstance(bits, str):
        bit_arr = np.array([int(c) for c in bits], dtype=np.int64)
    else:
        bit_arr = np.asarray(bits, dtype=np.int64)
    if len(bit_arr) != n:
        raise ValueError(f"bits must have length n_spins ({n}), got {len(bit_arr)}")

    spins = 1 - 2 * bit_arr  # bit 0 -> spin +1, bit 1 -> spin -1

    pairs = _boundary_pairs(n, periodic)
    coupling_energy = sum(J[k] * spins[i] * spins[j] for k, (i, j) in enumerate(pairs))
    field_energy = float(np.dot(spins, h))
    return float(coupling_energy) + field_energy


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src import ising_model
    from src import exact_solver

    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    # ---- 1. Analytic ground-state energy checks -------------
    print("\n--- Analytic ground-state energy checks ---")
    for instance, e0_expected in [
        (ising_model.TWO_SPIN_FM_INSTANCE, -1.0),
        (ising_model.THREE_SPIN_AFM_INSTANCE, -2.0),
        (ising_model.THREE_SPIN_FRUSTRATED_RING_INSTANCE, -1.0),
    ]:
        H = build_ising_hamiltonian(*instance.as_tuple())
        mat = H.to_matrix()
        eigvals = np.linalg.eigvalsh(mat)
        e0 = float(np.min(eigvals.real))
        check(
            f"{instance.name}: E_0 == {e0_expected}",
            np.isclose(e0, e0_expected, atol=1e-8),
            f"got E_0={e0}",
        )

    # ---- 2. bitstring_energy agreement vs exact_solver ---------------
    print("\n--- bitstring_energy vs exact_solver.bitstring_energy ---")
    obc_instance = ising_model.THREE_SPIN_AFM_INSTANCE  # OBC, n=3
    pbc_instance = ising_model.THREE_SPIN_FRUSTRATED_RING_INSTANCE  # PBC, n=3
    test_bits = ["000", "010", "101", "111", "011"]

    for instance in (obc_instance, pbc_instance):
        J, h, boundary = instance.as_tuple()
        for bits in test_bits:
            e_ham = bitstring_energy(bits, J, h, boundary)
            e_exact = exact_solver.bitstring_energy(bits, J, h, boundary)
            check(
                f"{instance.name} bits={bits}: hamiltonian.bitstring_energy == exact_solver.bitstring_energy",
                np.isclose(e_ham, e_exact, atol=1e-10),
                f"hamiltonian.py={e_ham}, exact_solver.py={e_exact}",
            )

    # ---- 3. SparsePauliOp.to_matrix() vs exact_solver full matrix --------
    print("\n--- SparsePauliOp.to_matrix() vs exact_solver.build_full_hamiltonian_matrix() ---")
    test_instances = [
        ising_model.TWO_SPIN_FM_INSTANCE,
        ising_model.THREE_SPIN_AFM_INSTANCE,
        ising_model.THREE_SPIN_FRUSTRATED_RING_INSTANCE,
        ising_model.generate_random_chain(5, boundary="OBC", seed=7, name="random_obc_5"),
        ising_model.generate_frustrated_chain(5, boundary="PBC", seed=11, name="frustrated_pbc_5"),
        ising_model.generate_field_chain(4, boundary="PBC", seed=3, name="field_pbc_4"),
    ]
    for instance in test_instances:
        J, h, boundary = instance.as_tuple()
        H = build_ising_hamiltonian(J, h, boundary)
        mat_ham = H.to_matrix().real
        mat_exact = exact_solver.build_full_hamiltonian_matrix(J, h, boundary)
        check(
            f"{instance.name}: to_matrix() == build_full_hamiltonian_matrix()",
            np.allclose(mat_ham, mat_exact, atol=1e-8),
            f"max abs diff={np.max(np.abs(mat_ham - mat_exact))}",
        )
        # confirm the matrix is diagonal
        off_diag = mat_ham - np.diag(np.diag(mat_ham))
        check(
            f"{instance.name}: SparsePauliOp matrix is diagonal",
            np.allclose(off_diag, 0.0, atol=1e-10),
        )

    # ---- 4. exact_solver.bitstring_energy(*instance.as_tuple()) pass ------
    print("\n--- exact_solver.bitstring_energy(*instance.as_tuple()) direct pass-through ---")
    for instance in (obc_instance, pbc_instance):
        bits = "0" * instance.n_spins
        e_exact_direct = exact_solver.bitstring_energy(bits, *instance.as_tuple())
        e_ham_direct = bitstring_energy(bits, *instance.as_tuple())
        print(
            f"  {instance.name} ({instance.boundary}): "
            f"exact_solver.bitstring_energy(*instance.as_tuple()) = {e_exact_direct} "
            f"(no ValueError raised)"
        )
        check(
            f"{instance.name}: exact_solver.bitstring_energy(*instance.as_tuple()) "
            "== hamiltonian.bitstring_energy(*instance.as_tuple())",
            np.isclose(e_exact_direct, e_ham_direct, atol=1e-10),
            f"exact_solver.py={e_exact_direct}, hamiltonian.py={e_ham_direct}",
        )

    print("\n" + ("=" * 60))
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
