"""
ising_model.py
==============

Generates 1D Ising models: nearest-neighbor couplings `J` and
optional local fields `h`, with open (OBC) or periodic (PBC) boundary
conditions.

    H = sum_i J_i * Z_i * Z_(i+1)  +  sum_i h_i * Z_i

Spins/qubits are indexed `0 ... n_spins-1`.

`h` has length `n_spins`.
`J` has length `n_spins - 1` for OBC, or
length `n_spins` for PBC

The extra entry `J[n_spins-1]` is the wraparound bond
between spin `n_spins-1` and spin `0`.

`boundary` is a string `"OBC"` or `"PBC"`

`J_i < 0` favors ferromagnetic (FM),
`J_i > 0` favors antiferromagnetic (AFM)

Returns plain numpy arrays
(wrapped in a `IsingInstance` container for convenience)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

VALID_BOUNDARIES = ("OBC", "PBC")


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IsingInstance:
    """A single 1D Ising problem instance:
    `name`, `n_spins`, `J`, `h`, `boundary`, `description`.
    """

    name: str
    n_spins: int
    J: np.ndarray
    h: np.ndarray
    boundary: str
    description: str = ""

    def __post_init__(self):
        if self.boundary not in VALID_BOUNDARIES:
            raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got {self.boundary!r}")
        if self.n_spins < 1:
            raise ValueError(f"n_spins must be >= 1, got {self.n_spins}")

        expected_J_len = self.n_spins if self.boundary == "PBC" else self.n_spins - 1
        if len(self.J) != expected_J_len:
            raise ValueError(
                f"IsingInstance {self.name!r}: boundary={self.boundary!r} with "
                f"n_spins={self.n_spins} expects len(J) == {expected_J_len}, "
                f"got {len(self.J)}"
            )
        if len(self.h) != self.n_spins:
            raise ValueError(
                f"IsingInstance {self.name!r}: expects len(h) == n_spins "
                f"({self.n_spins}), got {len(self.h)}"
            )

    def as_tuple(self):
        return self.J, self.h, self.boundary


# --------------------------------------------------------------------------
# RNG helper
# --------------------------------------------------------------------------


def _get_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _num_couplings(n_spins: int, boundary: str) -> int:
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got {boundary!r}")
    return n_spins if boundary == "PBC" else n_spins - 1


# --------------------------------------------------------------------------
# Array generators
# --------------------------------------------------------------------------


def generate_couplings(
    n_spins: int,
    boundary: str = "OBC",
    distribution: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Random coupling array `J`, sized for the given boundary condition."""
    rng = _get_rng(rng, seed)
    n_couplings = _num_couplings(n_spins, boundary)
    if distribution == "uniform":
        J = rng.uniform(low, high, size=n_couplings)
    elif distribution == "gaussian":
        J = rng.normal(mean, std, size=n_couplings)
    else:
        raise ValueError(f"Unknown distribution {distribution!r}, expected 'uniform' or 'gaussian'")
    return J.astype(float)


def generate_fields(
    n_spins: int,
    mode: str = "zero",
    value: float = 0.0,
    distribution: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Local field array `h`; `h_i` is the coefficient of `Z_i` on spin `i`."""
    # default: `h_i = 0`
    # coupling only case
    if mode == "zero":
        return np.zeros(n_spins, dtype=float)
    # every site gets same constant field value
    elif mode == "uniform_value":
        return np.full(n_spins, float(value), dtype=float)
    elif mode == "random":
        rng = _get_rng(rng, seed)
        if distribution == "uniform":
            h = rng.uniform(low, high, size=n_spins)
        elif distribution == "gaussian":
            h = rng.normal(mean, std, size=n_spins)
        else:
            raise ValueError(
                f"Unknown distribution {distribution!r}, expected 'uniform' or 'gaussian'"
            )
        return h.astype(float)
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'zero', 'uniform_value', or 'random'")


# --------------------------------------------------------------------------
# Instance-level generators
# --------------------------------------------------------------------------


def generate_uniform_FM_chain(
    n_spins: int,
    boundary: str = "OBC",
    J_value: float = -1.0,
    h_value: float = 0.0,
    name: Optional[str] = None,
) -> IsingInstance:
    """Uniform chain: every coupling and field is identical (no randomness)."""
    n_couplings = _num_couplings(n_spins, boundary)
    J = np.full(n_couplings, float(J_value))
    h = np.full(n_spins, float(h_value))
    return IsingInstance(
        name=name or "uniform_FM",
        n_spins=n_spins,
        J=J,
        h=h,
        boundary=boundary,
        description=(
            f"Uniform J={J_value} chain, {boundary} boundary, h={h_value}. "
            "Ground state: all spins aligned (if J<0) or alternating (if J>0)."
        ),
    )


def generate_random_chain(
    n_spins: int,
    boundary: str = "OBC",
    J_distribution: str = "uniform",
    J_low: float = -1.0,
    J_high: float = 1.0,
    J_mean: float = 0.0,
    J_std: float = 1.0,
    h_mode: str = "zero",
    h_value: float = 0.0,
    h_distribution: str = "uniform",
    h_low: float = -1.0,
    h_high: float = 1.0,
    h_mean: float = 0.0,
    h_std: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    name: Optional[str] = None,
) -> IsingInstance:
    """Random instance generator:
    random couplings plus optional random/uniform/zero field.
    """
    rng = _get_rng(rng, seed)
    J = generate_couplings(
        n_spins,
        boundary=boundary,
        distribution=J_distribution,
        low=J_low,
        high=J_high,
        mean=J_mean,
        std=J_std,
        rng=rng,
    )
    h = generate_fields(
        n_spins,
        mode=h_mode,
        value=h_value,
        distribution=h_distribution,
        low=h_low,
        high=h_high,
        mean=h_mean,
        std=h_std,
        rng=rng,
    )
    return IsingInstance(
        name=name or "random_chain",
        n_spins=n_spins,
        J=J,
        h=h,
        boundary=boundary,
        description=(
            f"Random J~{J_distribution} chain, {boundary} boundary, h_mode={h_mode}."
        ),
    )


def generate_frustrated_chain(
    n_spins: int,
    boundary: str = "OBC",
    distribution: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    name: Optional[str] = None,
) -> IsingInstance:
    """Wrapper around `generate_random_chain`; hardcodes `h_mode="zero"` (no field)."""
    instance = generate_random_chain(
        n_spins,
        boundary=boundary,
        J_distribution=distribution,
        J_low=low,
        J_high=high,
        J_mean=mean,
        J_std=std,
        h_mode="zero",
        seed=seed,
        rng=rng,
        name=name or "frustrated",
    )
    return instance


def generate_field_chain(
    n_spins: int,
    boundary: str = "OBC",
    J_value: float = -1.0,
    h_distribution: str = "uniform",
    h_low: float = -0.5,
    h_high: float = 0.5,
    h_mean: float = 0.0,
    h_std: float = 0.5,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    name: Optional[str] = None,
) -> IsingInstance:
    """Uniform coupling chain with a nonzero random local field on every
    site; default field range is narrower than `J`'s so the field
    perturbs rather than dominates.
    """
    rng = _get_rng(rng, seed)
    n_couplings = _num_couplings(n_spins, boundary)
    J = np.full(n_couplings, float(J_value))
    h = generate_fields(
        n_spins,
        mode="random",
        distribution=h_distribution,
        low=h_low,
        high=h_high,
        mean=h_mean,
        std=h_std,
        rng=rng,
    )
    return IsingInstance(
        name=name or "with_field",
        n_spins=n_spins,
        J=J,
        h=h,
        boundary=boundary,
        description=(
            f"Uniform J={J_value} chain, {boundary} boundary, "
            f"random field h~{h_distribution}(range/scale around 0)."
        ),
    )


def generate_uniform_AFM_ring(
    n_spins: int,
    J_value: float = 1.0,
    h_value: float = 0.0,
    name: Optional[str] = None,
) -> IsingInstance:
    """Uniform AFM ring: `J_i = +J_value` on every bond

    H = J_value * sum_i Z_i * Z_{(i+1) mod n_spins}  +  h_value * sum_i Z_i

    Analytic ground-state energy (for `J_value=1.0`, `h_value=0`):

        E_gs = -n_spins           for even n_spins
        E_gs = -(n_spins - 2)     for odd n_spins
    """
    n_couplings = _num_couplings(n_spins, "PBC")  # == n_spins
    J = np.full(n_couplings, float(J_value))
    h = np.full(n_spins, float(h_value))
    return IsingInstance(
        name=name or "uniform_AFM_ring",
        n_spins=n_spins,
        J=J,
        h=h,
        boundary="PBC",
        description=(
            f"Uniform J={J_value} (AFM) ring, PBC, h={h_value}. "
            "E_gs = -n_spins for even n_spins (2-fold degenerate Neel "
            "order), or -(n_spins - 2) for odd n_spins (2*n_spins-fold "
            "degenerate, one frustrated bond)."
        ),
    )


def generate_test_instances(n_spins: int = 6, seed: int = 42) -> dict:
    """Assembles the depth sweep's four instances into a dict; each
    random instance gets a distinct seed."""
    return {
        "uniform_FM": generate_uniform_FM_chain(
            n_spins, boundary="OBC", J_value=-1.0, h_value=0.0,
            name="uniform_FM",
        ),
        "frustrated": generate_frustrated_chain(
            n_spins, boundary="OBC", distribution="uniform", low=-1.0, high=1.0,
            seed=seed, name="frustrated",
        ),
        "with_field": generate_field_chain(
            n_spins, boundary="OBC", J_value=-1.0,
            h_distribution="uniform", h_low=-0.5, h_high=0.5,
            seed=seed + 1, name="with_field",
        ),
        "frustrated_pbc": generate_frustrated_chain(
            n_spins, boundary="PBC", distribution="uniform", low=-1.0, high=1.0,
            seed=seed + 2, name="frustrated_pbc",
        ),
    }


# --------------------------------------------------------------------------
# Hand-picked small instances with known ground states
# (importable by tests/test_hamiltonian.py)
# --------------------------------------------------------------------------


def two_spin_FM_instance() -> IsingInstance:
    """2-spin OBC, J=[-1.0] (FM), h=[0,0]. E_0=-1.0 (aligned, doubly
    degenerate); anti-aligned gives E=+1.0.

    H = -1 * Z_0 * Z_1
    """
    J = np.array([-1.0])
    h = np.array([0.0, 0.0])
    return IsingInstance(
        name="two_spin_FM",
        n_spins=2,
        J=J,
        h=h,
        boundary="OBC",
        description=(
            "H = -Z_0 Z_1. Ground energy E_0 = -1.0, doubly degenerate "
            "(spins aligned: both +1 or both -1). Excited energy = +1.0 "
            "(spins anti-aligned)."
        ),
    )


def three_spin_AFM_instance() -> IsingInstance:
    """3-spin OBC, J=[1,1] (AFM), h=[0,0,0]. Unfrustrated (OBC AFM chains
    always are); E_0=-2.0, doubly degenerate (alternating spins).

    H = Z_0*Z_1 + Z_1*Z_2
    """
    J = np.array([1.0, 1.0])
    h = np.array([0.0, 0.0, 0.0])
    return IsingInstance(
        name="three_spin_AFM",
        n_spins=3,
        J=J,
        h=h,
        boundary="OBC",
        description=(
            "H = Z_0 Z_1 + Z_1 Z_2. Ground energy E_0 = -2.0, "
            "doubly degenerate (alternating spin configurations)."
        ),
    )


def three_spin_frustrated_ring_instance() -> IsingInstance:
    """3-spin PBC, J=[1,1,1] (AFM), h=[0,0,0] -- the smallest frustrated
    instance; E_0=-1.0, 6-fold degenerate (one bond always unsatisfied).

    H = Z_0*Z_1 + Z_1*Z_2 + Z_2*Z_0
    """
    J = np.array([1.0, 1.0, 1.0])
    h = np.array([0.0, 0.0, 0.0])
    return IsingInstance(
        name="three_spin_frustrated_ring",
        n_spins=3,
        J=J,
        h=h,
        boundary="PBC",
        description=(
            "H = Z_0 Z_1 + Z_1 Z_2 + Z_2 Z_0 (PBC ring). Ground energy E_0 = -1.0, "
            "6-fold degenerate (one bond always unsatisfied)."
        ),
    )


# Pre-built constants for direct import
#   from src.ising_model import TWO_SPIN_FM_INSTANCE
TWO_SPIN_FM_INSTANCE = two_spin_FM_instance()
THREE_SPIN_AFM_INSTANCE = three_spin_AFM_instance()
THREE_SPIN_FRUSTRATED_RING_INSTANCE = three_spin_frustrated_ring_instance()
