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
    (no `SparsePauliOp`/qiskit involved)
    (for cross-checking against `exact_solver.bitstring_energy`)."""
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
