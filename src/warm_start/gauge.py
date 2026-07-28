"""
gauge.py
========

Bitflip gauge transformation of the Ising cost Hamiltonian.

Method: Noise-Directed Adaptive Warm-Starting (ND-AWS)
Authors: F. B. Maciejewski, S. Hadfield, O. Wallis, G. Pennington,
S. Brandhofer, S. Woerner, D. J. Egger, D. Venturelli
Reference: arXiv:2607.09368 (2026), Appendix A4a, Eq. (A12)
(gauge concept originates in Noise-Directed Adaptive Remapping (NDAR):
F. B. Maciejewski, J. Biamonte, S. Hadfield, D. Venturelli, "Improving
quantum approximate optimization by noise-directed adaptive remapping,"
Quantum 9, 1906 (2025), arXiv:2404.01412.)

Independent reimplementation written directly from the paper. Not derived
from quapopt, the authors' reference implementation:
https://github.com/usra-riacs/quantum-approximate-optimization

Implements Eq. (A12), NOT main-text Eq. (4)
----------------------------------------------------------------------------
Main-text Eq. (4) only transforms the ZZ couplings:

    H^y = sum_{i,j} (-1)^{y_i + y_j} J_ij Z_i Z_j

Appendix Eq. (A12) additionally covers the local-field h_i term:

    H^y = P_y H P_y
        = sum_i (-1)^{y_i} h_i Z_i  +  sum_{i<j} (-1)^{y_i + y_j} J_ij Z_i Z_j

Eq. (A12) is the one implemented here because this repo's Hamiltonian
(`src/hamiltonian.py`) includes field terms h_i; using main-text Eq. (4)
alone would silently drop the field's gauge transform on every
instance including a field (`ising_model.generate_field_chain`, etc.).

Background
----------------------------------------------------------------------------
The bitflip operator is P_y = tensor_i X_i^{y_i} (Sec. II B / App. A4a).
It is a unitary change-of-basis that flips the |0>/|1> basis states
according to bit y_i: P_y |0...0> = |y_0 ... y_{n-1}>. Conjugating H by P_y
preserves H's eigenvalues while permuting eigenvectors (candidate
solutions) under P_y.

Key identity, Eq. (5): <y| H |y> = <0...0| H^y |0...0>.
The previous iteration's best-found bitstring y becomes the device's
ground state |0...0> at the next iteration once H is
gauge-transformed to H^y.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

VALID_BOUNDARIES = ("OBC", "PBC")

BitsLike = Union[str, Sequence[int], np.ndarray, int]


# --------------------------------------------------------------------------
# Small bit/int helpers (little-endian: bit i = qubit i, index 0 = qubit 0)
# --------------------------------------------------------------------------


def int_to_bits(index: int, n_spins: int) -> np.ndarray:
    """`index` -> bit array of length `n_spins`, little-endian (bit i =
    qubit i, matching `hamiltonian.py`/`exact_solver.py`'s convention)."""
    if index < 0 or index >= 2 ** n_spins:
        raise ValueError(f"index={index} out of range for n_spins={n_spins}")
    return np.array([(index >> i) & 1 for i in range(n_spins)], dtype=np.int64)


def bits_to_int(bits: Sequence[int]) -> int:
    """Bit array (`bits[i]` = qubit i's bit) -> integer basis-state index,
    little-endian, matching `int_to_bits`'s inverse."""
    bits = np.asarray(bits, dtype=np.int64)
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def normalize_bits(y: BitsLike, n_spins: int) -> np.ndarray:
    """Accepts a bitstring as a str ("010"), a 0/1 sequence/array, or an
    integer basis-state index, and returns a length-`n_spins` int64 array
    with `arr[i]` = qubit i's bit (0 or 1)."""
    if isinstance(y, str):
        arr = np.array([int(ch) for ch in y], dtype=np.int64)
    elif isinstance(y, (int, np.integer)):
        arr = int_to_bits(int(y), n_spins)
    else:
        arr = np.asarray(y, dtype=np.int64)
    if len(arr) != n_spins:
        raise ValueError(f"y must have length n_spins ({n_spins}), got {len(arr)}")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"y must be a 0/1 bitstring, got {arr}")
    return arr


def xor_bits(x: BitsLike, y: BitsLike) -> np.ndarray:
    """XOR of two same-length bitstrings; used to undo/apply
    the P_y bitflip permutation in post-processing."""
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same shape, got {x_arr.shape} vs {y_arr.shape}")
    return np.bitwise_xor(x_arr, y_arr)


# --------------------------------------------------------------------------
# Boundary helpers (matching hamiltonian.py / exact_solver.py exactly)
# --------------------------------------------------------------------------


def _normalize_boundary(boundary: str) -> bool:
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got {boundary!r}")
    return boundary == "PBC"


def _boundary_pairs(n_spins: int, periodic: bool):
    if periodic:
        return [(i, (i + 1) % n_spins) for i in range(n_spins)]
    return [(i, i + 1) for i in range(n_spins - 1)]


# --------------------------------------------------------------------------
# Eq. (A12): gauge transform of (J, h)
# --------------------------------------------------------------------------


def gauge_transform(
    J: np.ndarray,
    h: np.ndarray,
    y: BitsLike,
    boundary: str = "OBC",
    n_spins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (A12): H^y = P_y H P_y, applied directly to the `(J, h)`
    coefficient arrays. See `notes/src/warm_start/gauge.md` Sec. 7.

        J_y[k] = (-1)^{y_i + y_j} * J[k]   for bond k = (i, j)
        h_y[i] = (-1)^{y_i} * h[i]
    """
    periodic = _normalize_boundary(boundary)
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    n = n_spins if n_spins is not None else len(h)
    y_bits = normalize_bits(y, n)

    pairs = _boundary_pairs(n, periodic)
    if len(J) != len(pairs):
        raise ValueError(
            f"boundary={boundary!r} with n_spins={n} expects len(J) == {len(pairs)}, got {len(J)}"
        )
    if len(h) != n:
        raise ValueError(f"expects len(h) == n_spins ({n}), got {len(h)}")

    sign_J = np.array([(-1.0) ** int(y_bits[i] + y_bits[j]) for (i, j) in pairs], dtype=float)
    sign_h = np.array([(-1.0) ** int(y_bits[i]) for i in range(n)], dtype=float)

    J_y = J * sign_J
    h_y = h * sign_h
    return J_y, h_y


def undo_gauge(sampled_bits: BitsLike, y: BitsLike) -> np.ndarray:
    """Map bits measured from the gauge-transformed
    circuit back to the PHYSICAL bitstring.

        physical_bits = sampled_bits XOR y
    """
    return xor_bits(sampled_bits, y)


def apply_gauge_to_bits(physical_bits: BitsLike, y: BitsLike) -> np.ndarray:
    """Inverse of `undo_gauge` -- same XOR (P_y is self-inverse), given a
    separate name for readability at the reverse call sites."""
    return xor_bits(physical_bits, y)