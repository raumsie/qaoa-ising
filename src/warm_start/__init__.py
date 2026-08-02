"""
src.warm_start
===============

Warm-starting for QAOA, built on top of `ising_model.py` /
`hamiltonian.py` / `qaoa_circuit.py` / `optimizer.py`:

- Warm-Start QAOA (WS-QAOA): D. J. Egger, J. Marecek, S. Woerner,
  "Warm-starting quantum optimization," Quantum 5, 479 (2021),
  arXiv:2009.10095.

An independent reimplementation written directly from the paper.

Modules
-------
`ws_ansatz`  -- Eqs. (6)-(7): WS initial state, WS mixer (as an exact
                QuantumCircuit), and the bitstring-derived bias
                generalization.
`egger_ws`   -- WS-QAOA (Egger et al.), continuous and rounded variants
                (`run_egger_ws`), its mean-field-relaxation classical
                front end, and the uniform `WarmStartComparisonResult`
                interface (`run_standard_qaoa`/`run_egger_ws`) used to
                compare standard QAOA and Egger-WS head-to-head.
`noise`      -- amplitude-damping noise model and `optimize_qaoa_noisy`,
                a thin wrapper over optimizer.py's `estimator` hook.
"""

from src.warm_start.egger_ws import (
    WarmStartComparisonResult,
    magnetization_to_bias,
    make_egger_ws_warm_start_fn,
    mean_field_relaxation,
    round_to_bitstring,
    run_egger_ws,
    run_standard_qaoa,
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
    # ws_ansatz
    "ws_bias_state",
    "ws_mixer_hamiltonian",
    "ws_mixer_circuit",
    "bias_from_bitstring",
    "build_ws_ansatz",
    # egger_ws
    "run_standard_qaoa",
    "run_egger_ws",
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
