# noinspection PyPep8Naming
"""
exact_solver.py
================

Exact diagonalization baseline for validating QAOA results.

Builds the full `2**n x 2**n` Ising Hamiltonian matrix from raw `(J, h,
boundary)` arrays and diagonalizes it with `numpy.linalg.eigh`. This
reconstructs the Hamiltonian independently of `hamiltonian.py`,
so the two can be cross-checked against each other.

Convention:

    H = sum_i J_i * Z_i * Z_(i+1)  +  sum_i h_i * Z_i

- boundary: {"OBC", "PBC"}

- Qubit/bit ordering matches Qiskit's little-endian convention:
    Ex.: A 3-qubit quantum register qreg with wave-function
    |ψ⟩=|A⊗B⊗C⟩=|ABC⟩ has qreg[0]=|C⟩, qreg[1]=|B⟩, qreg[2]=|A⟩.

- Spin convention:
    bit == 0 -> Z eigenvalue +1 ("spin up")
    bit == 1 -> Z eigenvalue -1 ("spin down")
        This matches Qiskit's `Z|0> = +|0>`.

n is capped at 16 (`MAX_N_SPINS`) since a dense `2**16 x 2**16` matrix
is already ~34 GB. Additionally, n > 14 is flagged with a warning
as memory/time-heavy but still allowed up to the hard cap.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

MAX_N_SPINS = 16
WARN_N_SPINS = 14

VALID_BOUNDARIES = ("OBC", "PBC")


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _infer_n_spins(J: np.ndarray, h: np.ndarray, boundary: str, n_spins: Optional[int] = None) -> int:
    """Infer n_spins from array shapes and boundary condition."""
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got {boundary!r}")

    J = np.asarray(J)
    h = np.asarray(h)

    n = n_spins if n_spins is not None else len(h)

    expected_J_len = n if boundary == "PBC" else n - 1
    if len(J) != expected_J_len:
        raise ValueError(
            f"boundary={boundary!r} with n_spins={n} expects len(J) == "
            f"{expected_J_len}, got {len(J)}"
        )
    if len(h) != n:
        raise ValueError(f"expects len(h) == n_spins ({n}), got {len(h)}")

    return n


def _validate_n_spins(n_spins: int) -> None:
    if n_spins < 1:
        raise ValueError(f"n_spins must be >= 1, got {n_spins}")
    if n_spins > MAX_N_SPINS:
        raise ValueError(
            f"n_spins={n_spins} exceeds MAX_N_SPINS={MAX_N_SPINS}: a dense "
            f"2**{n_spins} x 2**{n_spins} matrix is not practical to "
            "diagonalize. Reduce chain length or use a different "
            "sparse/iterative solver."
        )
    if n_spins > WARN_N_SPINS:
        dim = 2 ** n_spins
        approx_gb = (dim ** 2) * 8 / 1e9
        warnings.warn(
            f"n_spins={n_spins} builds a dense {dim} x {dim} matrix "
            f"(~{approx_gb:.1f} GB, float64). This is slow/memory-heavy; "
            f"consider n_spins <= {WARN_N_SPINS} for interactive use.",
            stacklevel=3,
        )


def _boundary_pairs(n_spins: int, boundary: str):
    """(i, j) index pairs for each coupling term, in J-array order.

    OBC has n - 1 bonds, while PBC has n bonds.

    When i = `n_spins - 1`, `(i + 1) % n_spins` wraps back to 0.
    """
    if boundary == "PBC":
        return [(i, (i + 1) % n_spins) for i in range(n_spins)]
    return [(i, i + 1) for i in range(n_spins - 1)] # OBC


# --------------------------------------------------------------------------
# Diagonal energies
# --------------------------------------------------------------------------


def all_state_energies(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> np.ndarray:
    """Computes the classical Ising energy of every one of
    the 2**n basis states in one vectorized shot.

    1. Validate/infer n from the shapes of J/h & boundary condition
    2. Enumerate every basis index k as indices
    3. Extract bits:
        `(indices[:, None] >> qubit_shifts[None, :]) & 1`
            produces a `(dim, n)` matrix `bits[k, i] = bit i` of integer k.
            This is Qiskit's little endian convention. Qubit 0 is the
            least significant bit. So `bits[k, 0]` is qubit 0's value
            in basis state k.
    4. Convert bits to spins:
        `spins = 1 - 2*bits` maps bit 0 -> +1 ("up") and
        bit 1 -> -1 ("down"), matching the convention Z|0⟩ = +|0⟩ .
        Results in a `(dim, n)` array: one row per basis state &
        one column per qubit.
    5. `_boundary_pairs` gives the `(i, j)` bond list.
        `pair_i/pair_j` are the arrays of the first/second indices.
        `spins[:, pair_i] * spins[:, pair_j]` is a `(dim, n_bonds)` array
        of `s_i * s_j` products for every state & every bond (n_bonds is
        n-1 for OBC, n for PBC).
    6. `@ J` contracts each row against the coupling strengths:
        `(spins[:, pair_i] * spins[:, pair_j]) @ J` gives
            `sum_bond J_bond * s_i * s_j`
            which is the total coupling energy of every state.
    7. Field energy is `spins @ h`, the same kind of contraction:
        `sum_i h_i * s_i` per basis state.
    8. Return `coupling_energy + field_energy` of shape `(dim,)`
       where `energies[k]` is the total energy
       `H = ΣJ*ZZ + Σh*Z` of basis state k.
    """
    n = _infer_n_spins(J, h, boundary, n_spins)
    _validate_n_spins(n)

    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)

    dim = 2 ** n
    indices = np.arange(dim, dtype=np.uint64)
    qubit_shifts = np.arange(n, dtype=np.uint64)
    # bits[k, i] = bit i (qubit i) of basis index k
    bits = ((indices[:, None] >> qubit_shifts[None, :]) & 1).astype(np.int64)
    spins = 1 - 2 * bits  # bit 0 -> spin +1, bit 1 -> spin -1 ; shape (dim, n)

    pairs = _boundary_pairs(n, boundary)
    if pairs:
        pair_i = np.array([p[0] for p in pairs])
        pair_j = np.array([p[1] for p in pairs])
        coupling_energy = (spins[:, pair_i] * spins[:, pair_j]) @ J
    else:
        coupling_energy = np.zeros(dim)

    field_energy = spins @ h

    return coupling_energy + field_energy


def bitstring_energy(
    bits, J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> float:
    """Computes the classical Ising energy of a single spin configuration
        (unlike `all_state_energies` which computes all 2**n at once)

    Steps:
    1. Infer/validate `n`
    2. Normalize `bits` into an int array.
        Accepts either a string like "010"
        or a sequence like [0, 1, 0]. `isinstance(bits, str)`
        converts each char into an int, otherwise uses whatever
        was passed via `np.asarray`. Either results in `bit_arr`,
        idexed so `bit_arr[i]` is qubit i's value
        (not the Qiskit reversed-string convention).
    3. Length check:
        `bit_arr` must have exactly `n` entries
    4. Convert bits to spins:
        `spins = 1 - 2*bit_arr` maps bit 0 -> +1 ("up") and
        bit 1 -> -1 ("down"), matching the convention from `all_state_energies`
    5. Coupling energy:
        Non-vectorized, single-state equivalent of the
        `spins[:, pair_i] * spins[:, pair_j] @ J` line in
        `all_state_energies`.
    6. Field energy:
        `np.dot(spins, h)` == Σ h_i * s_i
        The single-state version of `spins @ h`.
    7. Return the sum, cast to float.
    """
    n = _infer_n_spins(J, h, boundary, n_spins)
    if isinstance(bits, str):
        bit_arr = np.array([int(c) for c in bits], dtype=np.int64)
    else:
        bit_arr = np.asarray(bits, dtype=np.int64)
    if len(bit_arr) != n:
        raise ValueError(f"bits must have length n_spins ({n}), got {len(bit_arr)}")

    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    spins = 1 - 2 * bit_arr

    pairs = _boundary_pairs(n, boundary)
    coupling_energy = sum(J[k] * spins[i] * spins[j] for k, (i, j) in enumerate(pairs))
    field_energy = float(np.dot(spins, h))
    return float(coupling_energy) + field_energy


# --------------------------------------------------------------------------
# Cheap path: ground-state energy only (no dense matrix, no eigh)
# --------------------------------------------------------------------------


def ground_state_energy(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> float:
    """Cheap path: minimum energy over all basis states, without building
    the full dense matrix or calling `eigh`. Use this whenever only the
    scalar ground-state energy is needed."""
    energies = all_state_energies(J, h, boundary, n_spins)
    return float(np.min(energies))


# --------------------------------------------------------------------------
# Expensive path: full dense matrix + eigh (needed for eigenvector/overlap)
# --------------------------------------------------------------------------


def build_full_hamiltonian_matrix(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> np.ndarray:
    """Build the full dense `2**n x 2**n` Hamiltonian matrix."""
    n = _infer_n_spins(J, h, boundary, n_spins)
    _validate_n_spins(n)
    # gets the (2**n,) array of every basis state's energy
    energies = all_state_energies(J, h, boundary, n)
    # expands the (2**n,) vector into a (2**n, 2**n) matrix
    # with `energies` on the diagonal
    return np.diag(energies)


def exact_diagonalize(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Full diagonalization via `numpy.linalg.eigh`.

    Returns
    -------
    eigenvalues : np.ndarray, shape (2**n,), ascending order
    eigenvectors : np.ndarray, shape (2**n, 2**n)
        eigenvectors[:, k] is the eigenvector for eigenvalues[k], expressed
        in the same little-endian computational basis as
        `qiskit.quantum_info.Statevector` (qubit 0 = least significant
        bit of the basis index).
    """
    H = build_full_hamiltonian_matrix(J, h, boundary, n_spins)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def ground_state(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """Ground-state energy and ground-state vector, for use in
    comparisons against QAOA-produced statevectors.

    Returns
    -------
    energy : float
    vector : np.ndarray, shape (2**n,)
    """
    eigenvalues, eigenvectors = exact_diagonalize(J, h, boundary, n_spins)
    energy = float(eigenvalues[0])
    vector = eigenvectors[:, 0]
    return energy, vector
