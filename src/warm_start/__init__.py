"""
src.warm_start
===============

Two independent warm-starting methods for QAOA, built on top of
`ising_model.py` / `hamiltonian.py` / `qaoa_circuit.py` / `optimizer.py`:

- Noise-Directed Adaptive Warm-Starting (ND-AWS): F. B. Maciejewski,
  S. Hadfield, O. Wallis, G. Pennington, S. Brandhofer, S. Woerner,
  D. J. Egger, D. Venturelli, "Quantum Approximate Optimization via
  Noise-Directed Adaptive Warm-Starting," arXiv:2607.09368 (2026).
- Warm-Start QAOA (WS-QAOA): D. J. Egger, J. Marecek, S. Woerner,
  "Warm-starting quantum optimization," Quantum 5, 479 (2021),
  arXiv:2009.10095.

Both are independent reimplementations written directly from their papers.
Not derived from quapopt, the ND-AWS authors' reference implementation:
https://github.com/usra-riacs/quantum-approximate-optimization

Modules
-------
`gauge`      -- Eq. (A12) bitflip gauge transform of (J, h); little-endian
                bit helpers.
`ws_ansatz`  -- Eq. (6)-(7)/(A5-A11): WS initial state, WS mixer (as an
                exact QuantumCircuit).
`nd_aws`     -- the ND-AWS driver (`run_nd_aws`),
                the `warm_start_fn(instance, p) -> np.ndarray` adapter
                for optimizer.py/experiment.py's existing
                extension points (`make_nd_aws_warm_start_fn`).
`egger_ws`   -- the original WS-QAOA (Egger et al.), continuous and rounded
                variants (`run_egger_ws`), its mean-field-relaxation
                classical front end, and the uniform
                `WarmStartComparisonResult` interface
                (`run_standard_qaoa`/`run_egger_ws`/
                `nd_aws_to_comparison_result`) used to compare standard
                QAOA, Egger-WS, and ND-AWS head-to-head.
`noise`      -- Eq. (10) amplitude-damping noise model, used to
                demonstrate the ND-vs-Standard noise-adaptivity mechanism.
"""

from src.warm_start.egger_ws import (
    WarmStartComparisonResult,
    magnetization_to_bias,
    make_egger_ws_warm_start_fn,
    mean_field_relaxation,
    nd_aws_to_comparison_result,
    round_to_bitstring,
    run_egger_ws,
    run_standard_qaoa,
)
from src.warm_start.gauge import (
    apply_gauge_to_bits,
    bits_to_int,
    gauge_transform,
    int_to_bits,
    normalize_bits,
    undo_gauge,
    xor_bits,
)
from src.warm_start.nd_aws import (
    NDAWSIterationRecord,
    NDAWSResult,
    bias_schedule_for_iteration,
    hamming_distance_quadratic_search,
    make_nd_aws_warm_start_fn,
    run_nd_aws,
    sample_bitstrings,
)
from src.warm_start.noise import (
    amplitude_damping_noise_model,
    build_noisy_backend,
    build_noisy_estimator,
    build_noisy_sampler,
    noisy_expectation,
    optimize_qaoa_noisy,
    prepare_ansatz_for_aer,
)
from src.warm_start.ws_ansatz import (
    bias_from_bitstring,
    build_ws_ansatz,
    ws_bias_state,
    ws_mixer_circuit,
    ws_mixer_hamiltonian,
)

__all__ = [
    # gauge
    "gauge_transform",
    "undo_gauge",
    "apply_gauge_to_bits",
    "xor_bits",
    "bits_to_int",
    "int_to_bits",
    "normalize_bits",
    # ws_ansatz
    "ws_bias_state",
    "ws_mixer_hamiltonian",
    "ws_mixer_circuit",
    "bias_from_bitstring",
    "build_ws_ansatz",
    # nd_aws
    "run_nd_aws",
    "make_nd_aws_warm_start_fn",
    "hamming_distance_quadratic_search",
    "sample_bitstrings",
    "bias_schedule_for_iteration",
    "NDAWSResult",
    "NDAWSIterationRecord",
    # egger_ws
    "run_standard_qaoa",
    "run_egger_ws",
    "nd_aws_to_comparison_result",
    "make_egger_ws_warm_start_fn",
    "mean_field_relaxation",
    "magnetization_to_bias",
    "round_to_bitstring",
    "WarmStartComparisonResult",
    # noise
    "amplitude_damping_noise_model",
    "build_noisy_backend",
    "build_noisy_estimator",
    "build_noisy_sampler",
    "prepare_ansatz_for_aer",
    "noisy_expectation",
    "optimize_qaoa_noisy",
]
