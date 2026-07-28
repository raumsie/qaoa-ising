"""
nd_aws.py
=========

Noise-Directed Adaptive Warm-Starting (ND-AWS) -- iterative driver.

F. B. Maciejewski, S. Hadfield, O. Wallis, G. Pennington, S. Brandhofer,
S. Woerner, D. J. Egger, D. Venturelli, "Quantum Approximate Optimization
via Noise-Directed Adaptive Warm-Starting," arXiv:2607.09368 (2026).

Independent reimplementation written directly from the paper. Not derived
from quapopt, the authors' reference implementation:
https://github.com/usra-riacs/quantum-approximate-optimization

Implements the main-text algorithm (Sec. II A-C, Eqs. 2, 5-9), the bias
schedule and greedy accept-if-better/termination rule of Sec. III A / Table
I, and HDQS post-processing (Sec. III A / Appendix D). Deliberate,
documented adaptations to this repo's scale and conventions:

- Instances are this project's existing n=2..14 1D-chain `IsingInstance`s
  (`ising_model.py`), not the paper's n=20 (Sec. II E) / n=100 (Sec. III)
  Erdos-Renyi / 3-regular-graph Hamiltonians.
- Phase separator == cost Hamiltonian always (`H_PS = H_C`, matching the
  paper's own Sec. II E proof-of-concept setting). The paper's Time-Block
  ansatz (App. A6 -- splitting `H_PS` into edge-subset batches to shorten
  hardware circuit depth) is deliberately NOT implemented: a 1D chain has
  only `n-1` (OBC) or `n` (PBC) couplings, nothing meaningful to batch, and
  this project targets GPU/CPU simulation, not QPU circuit depth.
- Variational angles (beta, gamma) are optimized with this repo's
  `optimizer.optimize_qaoa` (COBYLA / L-BFGS-B via `scipy.optimize.minimize`),
  not the paper's COBYQA + basinhopping (App. A7). This is an explicit,
  documented deviation from the paper, per this project's optimizer surface
  (`optimizer.VALID_OPTIMIZER_METHODS`), not an oversight.
- Both `variant="ND"` (gauge-transformed H_C/H_PS, biased towards
  |0...0>) and `variant="standard"` (no gauge; per-qubit bias towards the
  best-found bitstring itself, Sec. III B's "Standard IWS" baseline) are
  implemented. Per Eqs. (A19)-(A20), both give IDENTICAL sampling
  distributions with no noise; they differ only once noise
  (`warm_start.noise.amplitude_damping_noise_model`) is introduced --
  see `tests/test_warm_start.py`.

Bias schedule (Table I / Sec. III A): c=0.5 at iteration r=0 (plain,
unbiased QAOA); c in {0.1, 0.05} for r=1,2,3; c in {0.05, 0.025} for r>3
(both values of c are tried each iteration and the better result is kept).

Termination (Sec. III A): stop once 3 consecutive iterations fail to
improve the best-found solution, `n_iterations_max` is reached, or a
supplied `known_optimum` is sampled.

Greedy accept-if-better (Sec. III A / III B): between iterations, the
running best solution is only replaced if a strictly better one is found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from src.hamiltonian import bitstring_energy, build_ising_hamiltonian
from src.optimizer import VALID_DEVICES, optimize_qaoa
from src.qaoa_circuit import build_qaoa_circuit_from_ising
from src.warm_start.gauge import gauge_transform, xor_bits
from src.warm_start.ws_ansatz import bias_from_bitstring, build_ws_ansatz

VALID_VARIANTS = ("ND", "standard")


def build_default_sampler(device: str = "CPU"):
    """Build a default Sampler-primitive instance for `device`; mirrors
    `optimizer.build_estimator`'s dispatch."""
    if device == "CPU":
        return StatevectorSampler()
    elif device == "GPU":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        backend = AerSimulator(method="statevector", device="GPU")
        return AerSamplerV2.from_backend(backend)
    elif device == "CPU_AER":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        backend = AerSimulator(method="statevector", device="CPU")
        return AerSamplerV2.from_backend(backend)
    else:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")


# --------------------------------------------------------------------------
# Bias schedule (Table I / Sec. III A)
# --------------------------------------------------------------------------


def bias_schedule_for_iteration(r: int) -> List[float]:
    """c=0.5 at r=0; c in {0.1, 0.05} for r=1,2,3; c in {0.05, 0.025} for
    r>3 (both values run each iteration, better result kept per Table I)."""
    if r <= 0:
        return [0.5]
    if r <= 3:
        return [0.1, 0.05]
    return [0.05, 0.025]


# --------------------------------------------------------------------------
# HDQS (Sec. III A / Appendix D)
# --------------------------------------------------------------------------


def hamming_distance_quadratic_search(
    bits, J: np.ndarray, h: np.ndarray, boundary: str
) -> Tuple[Tuple[int, ...], float]:
    """HDQS: from the best sample `x`, generate the `n` 1-flip and
    `n(n-1)/2` 2-flip Hamming-distance neighbors, and keep the
    lowest-energy bitstring among `{x} U neighbors`."""
    bits = np.asarray(bits, dtype=np.int64)
    n = len(bits)
    candidates = [bits.copy()]
    for i in range(n):
        flipped = bits.copy()
        flipped[i] ^= 1
        candidates.append(flipped)
    for i in range(n):
        for j in range(i + 1, n):
            flipped = bits.copy()
            flipped[i] ^= 1
            flipped[j] ^= 1
            candidates.append(flipped)
    energies = [bitstring_energy(cand, J, h, boundary) for cand in candidates]
    best_idx = int(np.argmin(energies))
    return tuple(int(b) for b in candidates[best_idx]), float(energies[best_idx])


# --------------------------------------------------------------------------
# Sampling helper
# --------------------------------------------------------------------------


def _bitstring_key_to_bits(key: str, n: int) -> np.ndarray:
    """Convert a Qiskit `Sampler` counts key (e.g. `"0110"`, qubit n-1
    leftmost) to `bits[i]` = qubit i's bit, matching `hamiltonian.py`'s
    `SparsePauliOp` label convention."""
    key = key.replace(" ", "")
    return np.array([int(key[n - 1 - i]) for i in range(n)], dtype=np.int64)


def sample_bitstrings(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    sampler=None,
    shots: int = 200,
    device: str = "CPU",
) -> Dict[Tuple[int, ...], int]:
    """Bind `params`, measure all qubits, and run through `sampler`.
    Returns `{bits_tuple: count}`."""
    if sampler is None:
        sampler = build_default_sampler(device)

    n = ansatz.num_qubits
    decomposed = ansatz.decompose(reps=6)
    original_names = [str(p) for p in ansatz.parameters]
    decomposed_names = [str(p) for p in decomposed.parameters]
    if decomposed_names != original_names:
        raise RuntimeError(
            "ansatz.decompose(reps=6) changed the parameter order/names "
            f"({original_names} -> {decomposed_names}); refusing to sample "
            "since this would silently corrupt parameter binding."
        )
    bind_dict = dict(zip(decomposed.parameters, np.asarray(params, dtype=float)))
    bound = decomposed.assign_parameters(bind_dict)
    bound.measure_all()

    result = sampler.run([bound], shots=shots).result()
    counts = result[0].data.meas.get_counts()

    out: Dict[Tuple[int, ...], int] = {}
    for key, count in counts.items():
        bits = tuple(int(b) for b in _bitstring_key_to_bits(key, n))
        out[bits] = out.get(bits, 0) + count
    return out


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class NDAWSIterationRecord:
    iteration: int
    variant: str
    c_tried: List[float]
    optimal_params_tried: List[np.ndarray]
    candidate_bits_tried: List[Tuple[int, ...]]
    candidate_energy_tried: List[float]
    chosen_c: float
    chosen_bits: Tuple[int, ...]
    chosen_energy: float
    accepted: bool  # True if this iteration is the best
    hdqs_applied: bool
    gauge_y: Tuple[int, ...]  # gauge bitstring used for H_C/H_PS this iteration


@dataclass
class NDAWSResult:
    instance_name: str
    variant: str
    best_bits: Tuple[int, ...]
    best_energy: float
    n_iterations: int
    converged: bool
    termination_reason: str
    history: List[NDAWSIterationRecord] = field(default_factory=list)


# --------------------------------------------------------------------------
# Main iterative driver
# --------------------------------------------------------------------------


def run_nd_aws(
    instance,
    p: int = 1,
    variant: str = "ND",
    n_iterations_max: int = 20,
    n_no_improve_stop: int = 3,
    shots: int = 200,
    n_angle_restarts: int = 3,
    sampler=None,
    device: str = "CPU",
    use_hdqs: bool = True,
    optimizer_method: str = "COBYLA",
    minimize_options: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    known_optimum: Optional[float] = None,
    verbose: bool = False,
    estimator=None,
) -> NDAWSResult:
    """Run the ND-AWS iteration loop (Sec. II A-C) for one `IsingInstance`,
    variant "ND" or "standard" (Eqs. A19-A20: identical noiseless)."""
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {device!r}")

    resolved_sampler = sampler if sampler is not None else build_default_sampler(device)

    J, h, boundary = instance.as_tuple()
    n = instance.n_spins
    rng = rng if rng is not None else np.random.default_rng(seed)
    zeros: Tuple[int, ...] = tuple(0 for _ in range(n))

    best_bits: Tuple[int, ...] = zeros
    best_energy: Optional[float] = None
    prev_bits: Tuple[int, ...] = zeros

    history: List[NDAWSIterationRecord] = []
    no_improve_count = 0
    termination_reason = "max_iterations_reached"

    for r in range(n_iterations_max):
        c_values = bias_schedule_for_iteration(r)
        iter_candidates = []  # (c, params, bits, energy, gauge_y, hdqs_applied)

        for c in c_values:
            if r == 0:
                # Standard QAOA, c=0.5, no gauge/no per-qubit bias:
                # iteration 0 is identical for both variants
                phase_sep_J, phase_sep_h = J, h
                c_arg = 0.5
                gauge_y: Tuple[int, ...] = zeros
            elif variant == "ND":
                phase_sep_J, phase_sep_h = gauge_transform(J, h, prev_bits, boundary=boundary, n_spins=n)
                c_arg = c  # scalar: same bias for every qubit, towards |0...0>
                gauge_y = prev_bits
            else:  # variant == "standard"
                phase_sep_J, phase_sep_h = J, h  # never gauge-transformed
                c_arg = bias_from_bitstring(np.asarray(prev_bits), c)  # per-qubit
                gauge_y = zeros

            H_phase_sep = build_ising_hamiltonian(phase_sep_J, phase_sep_h, boundary)
            ansatz = build_ws_ansatz(H_phase_sep, c_arg, p=p, n_spins=n)

            best_opt = None
            for _ in range(max(1, n_angle_restarts)):
                # TODO: change call so that ND-AWS can use GPU
                res = optimize_qaoa(
                    ansatz,
                    H_phase_sep,
                    optimizer_method=optimizer_method,
                    device=device,
                    minimize_options=minimize_options,
                    rng=rng,
                    estimator=estimator,
                )
                if best_opt is None or res.optimal_energy < best_opt.optimal_energy:
                    best_opt = res

            counts = sample_bitstrings(
                ansatz, best_opt.optimal_params, sampler=resolved_sampler, shots=shots
            )

            sampled = []
            for bits_in_frame in counts:
                if gauge_y != zeros:
                    physical_bits = xor_bits(np.asarray(bits_in_frame), np.asarray(gauge_y))
                else:
                    physical_bits = np.asarray(bits_in_frame)
                e = bitstring_energy(physical_bits, J, h, boundary)
                sampled.append((tuple(int(b) for b in physical_bits), e))
            candidate_bits, candidate_energy = min(sampled, key=lambda t: t[1])

            hdqs_applied = False
            if use_hdqs:
                hdqs_bits, hdqs_energy = hamming_distance_quadratic_search(candidate_bits, J, h, boundary)
                if hdqs_energy < candidate_energy:
                    candidate_bits, candidate_energy = hdqs_bits, hdqs_energy
                    hdqs_applied = True

            iter_candidates.append(
                (c, best_opt.optimal_params, candidate_bits, candidate_energy, gauge_y, hdqs_applied)
            )

        chosen = min(iter_candidates, key=lambda t: t[3])
        _, _, chosen_bits, chosen_energy, chosen_gauge_y, chosen_hdqs = chosen
        chosen_c = chosen[0]

        accepted = best_energy is None or chosen_energy < best_energy
        if accepted:
            best_energy = chosen_energy
            best_bits = chosen_bits
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(
            NDAWSIterationRecord(
                iteration=r,
                variant=variant,
                c_tried=c_values,
                optimal_params_tried=[cand[1] for cand in iter_candidates],
                candidate_bits_tried=[cand[2] for cand in iter_candidates],
                candidate_energy_tried=[cand[3] for cand in iter_candidates],
                chosen_c=chosen_c,
                chosen_bits=chosen_bits,
                chosen_energy=chosen_energy,
                accepted=accepted,
                hdqs_applied=chosen_hdqs,
                gauge_y=chosen_gauge_y,
            )
        )

        if verbose:
            print(
                f"[iter {r}] variant={variant} c_tried={c_values} "
                f"chosen_energy={chosen_energy:.6f} best_energy={best_energy:.6f} "
                f"accepted={accepted}"
            )

        prev_bits = best_bits  # greedy: always bias/gauge towards the current best

        if known_optimum is not None and np.isclose(best_energy, known_optimum, atol=1e-9):
            termination_reason = "known_optimum_reached"
            break
        if no_improve_count >= n_no_improve_stop:
            termination_reason = "no_improvement"
            break
    else:
        termination_reason = "max_iterations_reached"

    return NDAWSResult(
        instance_name=getattr(instance, "name", "unknown"),
        variant=variant,
        best_bits=best_bits,
        best_energy=float(best_energy) if best_energy is not None else float("nan"),
        n_iterations=len(history),
        converged=(termination_reason != "max_iterations_reached"),
        termination_reason=termination_reason,
        history=history,
    )


# --------------------------------------------------------------------------
# warm_start_fn adapter for optimizer.py's initial_params / experiment.py's
# warm_start_fn extension points
# --------------------------------------------------------------------------


def make_nd_aws_warm_start_fn(
    variant: str = "ND",
    n_iterations_max: int = 10,
    shots: int = 200,
    n_angle_restarts: int = 2,
    use_hdqs: bool = True,
    optimizer_method: str = "COBYLA",
    device: str = "CPU",
    minimize_options: Optional[dict] = None,
    sampler=None,
    seed: Optional[int] = None,
) -> Callable:
    """Approximate `warm_start_fn(instance, p)` adapter for ND-AWS: keeps
    only `gamma` from the real result and re-optimizes `beta` from scratch,
    since the default ansatz this hook targets has neither ND-AWS's custom
    mixer nor its gauge-transformed Hamiltonian. NOT the actual ansatz;
    `run_nd_aws()` is used for that."""

    def warm_start_fn(instance, p: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed)
        result = run_nd_aws(
            instance,
            p=p,
            variant=variant,
            n_iterations_max=n_iterations_max,
            shots=shots,
            n_angle_restarts=n_angle_restarts,
            use_hdqs=use_hdqs,
            optimizer_method=optimizer_method,
            device=device,
            minimize_options=minimize_options,
            sampler=sampler,
            rng=local_rng,
        )

        if not result.history:
            # n_iterations_max <= 0 -> no ND-AWS iteration ran, so
            # `result.history[-1]` below would raise IndexError, and there's
            # no optimized gamma to seed from either. Not a real warm start
            # -- just a safe, valid (beta, gamma) vector so the caller still
            # gets something usable instead of a crash.
            return local_rng.uniform(0.0, 0.1, size=2 * p)

        last = result.history[-1]
        best_local_idx = int(np.argmin(last.candidate_energy_tried))
        gamma_seed_params = np.asarray(last.optimal_params_tried[best_local_idx], dtype=float)
        # WS ansatz parameter order is the same alphabetical beta-then-gamma
        # order as the default ansatz (both built via qaoa_circuit.build_qaoa_circuit).
        gamma_values = gamma_seed_params[p:]

        J, h, boundary = instance.as_tuple()
        H = build_ising_hamiltonian(J, h, boundary)
        default_ansatz, _ = build_qaoa_circuit_from_ising(J, h, boundary=boundary, p=p)
        beta_seed = local_rng.uniform(0.0, 0.1, size=p)  # small, non-zero
        x0 = np.concatenate([beta_seed, gamma_values])

        refine = optimize_qaoa(
            default_ansatz,
            H,
            optimizer_method=optimizer_method,
            device=device,
            initial_params=x0,
            minimize_options={"maxiter": 30},
        )
        return refine.optimal_params

    return warm_start_fn
