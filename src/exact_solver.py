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
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Size limits
# --------------------------------------------------------------------------
MAX_N_SPINS_DIAGONAL = 30
WARN_N_SPINS_DIAGONAL = 20

MAX_N_SPINS_DENSE = 14
WARN_N_SPINS_DENSE = 12

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


def _validate_n_spins_diagonal(n_spins: int) -> None:
    """Size check for the diagonal path."""
    if n_spins < 1:
        raise ValueError(f"n_spins must be >= 1, got {n_spins}")
    if n_spins > MAX_N_SPINS_DIAGONAL:
        raise ValueError(
            f"n_spins={n_spins} exceeds MAX_N_SPINS_DIAGONAL={MAX_N_SPINS_DIAGONAL} "
            "for the diagonal path."
        )
    if n_spins > WARN_N_SPINS_DIAGONAL:
        dim = 2 ** n_spins
        approx_mb = dim * 8 / 1e6
        warnings.warn(
            f"n_spins={n_spins} allocates length-{dim} float64 array(s) "
            f"(~{approx_mb:.0f} MB each) on the diagonal path. "
            f"Flagging since it exceeds WARN_N_SPINS_DIAGONAL="
            f"{WARN_N_SPINS_DIAGONAL}.",
            stacklevel=3,
        )


def _validate_n_spins_dense(n_spins: int) -> None:
    """Size check for the dense path."""
    if n_spins < 1:
        raise ValueError(f"n_spins must be >= 1, got {n_spins}")
    if n_spins > MAX_N_SPINS_DENSE:
        raise ValueError(
            f"n_spins={n_spins} exceeds MAX_N_SPINS_DENSE={MAX_N_SPINS_DENSE} "
            "for the dense path."
        )
    if n_spins > WARN_N_SPINS_DENSE:
        dim = 2 ** n_spins
        approx_gb = (dim ** 2) * 8 / 1e9
        warnings.warn(
            f"n_spins={n_spins} builds a dense {dim} x {dim} matrix "
            f"(~{approx_gb:.1f} GB, float64). This is slow/memory-heavy; "
            f"consider n_spins <= {WARN_N_SPINS_DENSE} for interactive use.",
            stacklevel=3,
        )


def _boundary_pairs(n_spins: int, boundary: str):
    """(i, j) index pairs for each coupling term, in J-array order."""
    if boundary == "PBC":
        return [(i, (i + 1) % n_spins) for i in range(n_spins)]
    return [(i, i + 1) for i in range(n_spins - 1)] # OBC


# --------------------------------------------------------------------------
# Diagonal energies
# --------------------------------------------------------------------------


def _basis_spins(n_spins: int) -> np.ndarray:
    """Spin values of every computational basis state, shape `(2**n, n)`."""
    dim = 2 ** n_spins
    # bit-shift arithmetic itself is done in int64 (dim can exceed int32
    # range), only the final +-1 spin values are narrowed to int8.
    indices = np.arange(dim, dtype=np.int64)
    qubit_shifts = np.arange(n_spins, dtype=np.int64)
    # bits[k, i] = bit i (qubit i) of basis index k
    bits = ((indices[:, None] >> qubit_shifts[None, :]) & 1).astype(np.int8)
    return np.int8(1) - np.int8(2) * bits  # bit 0 -> +1, bit 1 -> -1 ; shape (dim, n)


def _bond_products(spins: np.ndarray, boundary: str) -> np.ndarray:
    """`(dim, n_bonds)` elementwise spin products `s_i * s_{i+1}` for every bond."""
    consecutive = spins[:, :-1] * spins[:, 1:]  # bonds (0,1), (1,2), ..., (n-2,n-1)
    if boundary == "PBC":
        wraparound = (spins[:, -1] * spins[:, 0])[:, None]  # bond (n-1, 0)
        return np.concatenate([consecutive, wraparound], axis=1)
    return consecutive


def all_state_energies(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> np.ndarray:
    """Computes the classical Ising energy of every one of
    the 2**n basis states in one vectorized shot.
    """
    n = _infer_n_spins(J, h, boundary, n_spins)
    _validate_n_spins_diagonal(n)

    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)

    dim = 2 ** n
    spins = _basis_spins(n)  # shape (dim, n), int8

    bond_products = _bond_products(spins, boundary)  # shape (dim, n_bonds), int8
    if bond_products.shape[1] > 0:
        coupling_energy = bond_products @ J
    else:
        coupling_energy = np.zeros(dim)

    field_energy = spins @ h

    return coupling_energy + field_energy


def bitstring_energy(
    bits, J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> float:
    """Computes the classical Ising energy of a single spin configuration
        (unlike `all_state_energies` which computes all 2**n at once)
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
# Cheap observables (diagonal path -- no dense matrix, no eigh)
# --------------------------------------------------------------------------

DEFAULT_DEGENERACY_TOL = 1e-6


def ground_space_size(
    J: np.ndarray,
    h: np.ndarray,
    boundary: str = "OBC",
    n_spins: Optional[int] = None,
    tol: float = DEFAULT_DEGENERACY_TOL,
) -> int:
    """Number of basis states within `tol` of the minimal energy, i.e. the
    ground-space degeneracy.
    """
    energies = all_state_energies(J, h, boundary, n_spins)
    e_gs = np.min(energies)
    return int(np.sum(energies <= e_gs + tol))


def _probabilities(psi: np.ndarray, input_kind: str) -> np.ndarray:
    """Normalize `psi` into a probability array."""
    psi = np.asarray(psi)
    if input_kind == "statevector":
        return np.abs(psi) ** 2
    elif input_kind == "probabilities":
        return np.asarray(psi, dtype=float)
    raise ValueError(
        f"input_kind must be 'statevector' or 'probabilities', got {input_kind!r}"
    )


def ground_state_success_probability(
    psi: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    boundary: str = "OBC",
    n_spins: Optional[int] = None,
    tol: float = DEFAULT_DEGENERACY_TOL,
    input_kind: str = "statevector",
) -> float:
    """P_gs: total probability mass on the (possibly degenerate) ground
    space of `H = sum_i J_i * Z_i * Z_(i+1) + sum_i h_i * Z_i`.
    """
    n = _infer_n_spins(J, h, boundary, n_spins)

    psi = np.asarray(psi)
    if len(psi) != 2 ** n:
        raise ValueError(f"psi must have length 2**n_spins ({2 ** n}), got {len(psi)}")

    probs = _probabilities(psi, input_kind)
    energies = all_state_energies(J, h, boundary, n)
    e_gs = np.min(energies)
    ground_mask = energies <= e_gs + tol
    return float(np.sum(probs[ground_mask]))


def two_point_correlators(
    psi: np.ndarray,
    n_spins: int,
    r_max: int = 10,
    input_kind: str = "statevector",
) -> np.ndarray:
    """Computes the full pairwise matrix `M[i,j] = ⟨Z_i Z_j⟩`"""
    _validate_n_spins_diagonal(n_spins)
    dim = 2 ** n_spins

    psi = np.asarray(psi)
    if len(psi) != dim:
        raise ValueError(f"psi must have length 2**n_spins ({dim}), got {len(psi)}")

    probs = _probabilities(psi, input_kind)
    spins = _basis_spins(n_spins).astype(np.float64)  # shape (dim, n_spins)

    weighted = spins * probs[:, None]  # shape (dim, n_spins)
    M = spins.T @ weighted  # shape (n_spins, n_spins)

    site_i = np.arange(n_spins)
    correlators = np.empty(r_max, dtype=float)
    for idx, r in enumerate(range(1, r_max + 1)):
        site_j = (site_i + r) % n_spins
        correlators[idx] = np.mean(M[site_i, site_j])
    return correlators


# --------------------------------------------------------------------------
# Expensive path: full dense matrix + eigh (needed for eigenvector/overlap)
# --------------------------------------------------------------------------


def build_full_hamiltonian_matrix(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> np.ndarray:
    """Build the full dense `2**n x 2**n` Hamiltonian matrix."""
    n = _infer_n_spins(J, h, boundary, n_spins)
    _validate_n_spins_dense(n)
    # gets the (2**n,) array of every basis state's energy
    energies = all_state_energies(J, h, boundary, n)
    # expands the (2**n,) vector into a (2**n, 2**n) matrix
    # with `energies` on the diagonal
    return np.diag(energies)


def exact_diagonalize(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Full diagonalization via `numpy.linalg.eigh`."""
    H = build_full_hamiltonian_matrix(J, h, boundary, n_spins)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def ground_state(
    J: np.ndarray, h: np.ndarray, boundary: str = "OBC", n_spins: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """Ground-state energy and ground-state vector, for use in
    comparisons against QAOA-produced statevectors."""
    eigenvalues, eigenvectors = exact_diagonalize(J, h, boundary, n_spins)
    energy = float(eigenvalues[0])
    vector = eigenvectors[:, 0]
    return energy, vector
