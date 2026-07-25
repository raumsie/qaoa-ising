# QAOA for Ising Models

QAOA-based ground-state search for a general 1D Ising chain validated against exact
diagonalization. Optimizer choice (COBYLA / L-BFGS-B) and device (CPU / GPU)
are both runtime-selectable, and results are benchmarked with a depth sweep 
across several coupling regimes.

## The model

```
H = Σ_i J_i * Z_i * Z_(i+1)   +   Σ_i h_i * Z_i
```

- `J_i`: nearest-neighbor coupling between site `i` and `i+1` — random,
  uniform, ferromagnetic (`J < 0`) or antiferromagnetic (`J > 0`)
- `h_i`: optional local longitudinal field at site `i` (zero recovers a
  pure-coupling model)
- **OBC** (open boundary): chain, no wraparound term
- **PBC** (periodic boundary): adds a wraparound bond `J_N * Z_N * Z_1`,
  forming a ring

`H` is diagonal in the computational basis, so the QAOA cost unitary is
built from `RZZ` gates (one per coupling) and `RZ` gates (one per field
term), with an `RX` mixer.

## Module breakdown (`src/`)

| Module | Responsibility                                                                                                                                                                                  |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ising_model.py` | Generates 1D Ising instances: random/uniform couplings, optional fields, OBC/PBC, reproducible via seeds.                                                                                       |
| `hamiltonian.py` | Converts `(J, h, boundary)` into a `SparsePauliOp` cost Hamiltonian, and provides an independent classical bitstring-energy function for cross-checking.                                        |
| `qaoa_circuit.py` | Builds the parameterized depth-`p` QAOA ansatz (`RZZ`/`RZ` cost layer + `RX` mixer) via `QAOAAnsatz`.                                                                                           |
| `optimizer.py` | Runs the QAOA optimization loop with a selectable optimizer (COBYLA or L-BFGS-B) and device: `"CPU"` (qiskit's `StatevectorEstimator`), `"GPU"` (Aer/CUDA), or `"CPU_AER"` (Aer's CPU backend). |
| `exact_solver.py` | Independent exact diagonalization of the full Ising Hamiltonian (`numpy.linalg.eigh`) for small chains, used as the baseline QAOA results are compared against.                                 |
| `experiment.py` | Runs the depth sweep: for each instance, `p`, and optimizer, runs multiple restarts and records the best energy, relative error `ε(p) = (E_qaoa(p) - E_0)/                                      |E_0|`, and run metadata into `SweepRecord`s ready for plotting or `json.dump`. |

## Running it

There are two notebooks with distinct roles:

- **`notebooks/run_on_colab.ipynb`** — driver that installs
  dependencies, imports directly from `src/`, runs a depth sweep via
  `run_depth_sweep`, and saves results to `results/depth_sweep_results.json`.
  A `COLAB_DEVICE` flag near the top of the setup cell selects `"CPU"` or
  `"GPU"` for the whole session.
- **`notebooks/analysis.ipynb`** — loads `results/depth_sweep_results.json`
  (and `results/gpu_timing_results.json` if present) and produces the
  `ε(p)` vs. `p` plots (saved to `results/epsilon_vs_p.png`, embedded
  below), the uniform-vs-frustrated comparison, restart-energy spread and
  cost plots, and a CPU-vs-GPU wall-clock chart. It
  does not recompute anything and does not require Colab, so it can be run
  entirely offline against previously saved results.

### Local setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Then run `notebooks/run_on_colab.ipynb` (locally or on Colab) to generate
results, followed by `notebooks/analysis.ipynb` to plot them.

## Current results (confirmation run)

![Relative energy error vs. QAOA depth, per instance and optimizer](results/epsilon_vs_p.png)


## Warm-starting

`optimizer.py` and `experiment.py` accept an `initial_params` /
`warm_start_fn` argument for a yet to be implemented warm-starting
method.
