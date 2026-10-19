# financial-contagion-sim

Co-evolutionary financial contagion & systemic risk simulator: signed
financial networks, SAIS contagion dynamics, structural-balance energy and
co-evolutionary Monte Carlo.

## Status

**Version 0.1.0, a review candidate (Phase 5: integration, validation and
release preparation).** It is not released and not distributable: no license
or author metadata has been supplied yet. See
[docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) and
[docs/TRACEABILITY.md](docs/TRACEABILITY.md).

`graph_builder.py` (Phase 1)
builds and validates signed networks. `sais_engine.py` and
`energy_calculator.py` (Phase 2) implement the SAIS dynamics and the network
energies. `simulator.py` (Phase 3) runs contagion and edge-sign flips as one
continuous-time process. Phase 4 adds `visualizer.py` (2D time series and 3D
surfaces), `sweep.py` (parameter sweeps and steady-state estimation),
`artifacts.py` (run directories and files) and the `run` / `sweep` commands in
`cli.py`. Each module docstring documents its conventions.

**Everything this tool produces is synthetic simulation output, not empirical
market data.**

| Feature (spec section) | Module | Status |
|---|---|---|
| Barabási–Albert and Erdős–Rényi signed graphs with positive link ratio r_0 (§5) | `graph_builder` | Implemented (Phase 1) |
| Signed graphs from stock-return correlation matrices (§5) | `graph_builder` | Implemented (Phase 1): one matrix per call |
| Signed-graph validation | `graph_builder` | Implemented (Phase 1) |
| Correlation sequences or rolling windows; MST / k-NN filtering | `graph_builder` | Not implemented (out of scope; see DECISIONS.md) |
| CTMC SAIS transition rates (§3.1) | `sais_engine` | Implemented (Phase 2) |
| Gillespie state-jump logic (§4) | `sais_engine` | Implemented (Phase 2): direct method on a fixed graph |
| Exact O(n^3) triad energy E_Δ (§3.2) | `energy_calculator` | Implemented (Phase 2), plus a sparse engine |
| Pairwise spreading energy E_p and total energy E(G_t) (§3.2) | `energy_calculator` | Implemented (Phase 2): E_p normalized by C(n,2) |
| Local energy difference for an edge-sign flip (§3.3) | `energy_calculator` | Implemented (Phase 2): O(min degree), not O(1) |
| Flip acceptance rule (§3.3) | `energy_calculator` | Implemented (Phase 2): not yet used by any loop |
| Co-evolutionary Monte Carlo loop tracking S(t), A(t), I(t), r(t), E(G_t) (§5) | `simulator` | Implemented (Phase 3): joint CTMC, per-edge flip rate γ |
| Steady-state estimation and parameter sweeps (§5) | `sweep` | Implemented (Phase 4): window-halves test, see below |
| 2D state-trajectory plots (§5) | `visualizer` | Implemented (Phase 4): S/A/I, r(t), E(G_t) as PNG |
| 3D Plotly surface plots exported as `.html` (§5) | `visualizer` | Implemented (Phase 4): ρ∞ and E over (α, r0), one surface per ρ0 |
| "Natural immunization" and "clustering" metrics (§5) | — | Not implemented: the spec defines no metric (Open) |
| CLI simulation commands (§4) | `cli` / `main.py` | Implemented (Phase 4): `run`, `sweep` |
| Correlation-matrix input from the CLI (§5) | — | Not implemented (Open); use `simulate()` from Python |

## Mathematical model (spec §3)

Summarized from the project specification
([docs/spec/specification.pdf](docs/spec/specification.pdf)). Each equation is
given in LaTeX with a plain-text line underneath. Points the spec leaves open,
and how they were settled, are in [DECISIONS.md](DECISIONS.md).

### States and transitions (§3.1)

Each node $i \in \{1, \dots, n\}$ has a state $X_i(t) \in \{1, 0, -1\}$: S
(solvent, unhedged) = 1, A (capital-buffered / hedged) = 0, I (distressed /
defaulted) = −1. $N_i^{I,+}(t)$ is the number of infected neighbours of $i$
joined to it by a **positive** edge; negative edges transmit nothing.

| Transition | Meaning | Rate (1/time) |
|---|---|---|
| S → I | Default cascade | $\beta \, N_i^{I,+}(t)$ |
| A → I | Attenuated default | $\beta_a \, N_i^{I,+}(t)$, with $0 \le \beta_a < \beta$ |
| S → A | Precautionary hedging | $\kappa \, N_i^{I,+}(t)$ |
| I → S | Recapitalization / bailout | $\delta$ |

Plain text: S→I at beta·N_i, A→I at beta_a·N_i (0 ≤ beta_a < beta), S→A at
kappa·N_i, I→S at delta.

### Energy (§3.2)

$$E(G_t) = \alpha \, E_\Delta(G_t) + (1 - \alpha) \, E_p(G_t), \qquad \alpha \in [0, 1]$$

Plain text: E = alpha·E_delta + (1 − alpha)·E_p, alpha in [0, 1].

$$E_\Delta(G_t) = -\frac{1}{\binom{n}{3}} \sum_{i<j<k} A_{ij}(t)\, A_{jk}(t)\, A_{ki}(t)$$

Plain text: E_delta = −(1 / C(n,3)) · Σ over i<j<k of A_ij·A_jk·A_ki, where
A_ij is the edge sign (+1 or −1) and 0 for a missing edge. A triad is balanced
when the product is +1 (+++, +−−) and unbalanced when it is −1 (++−, −−−).

Pair term $E_{ij}^p$ for an edge with sign $A_{ij}$: +1 for S+I and A+I, −1 for
S−I and A−I, and 0 for S±S, I±I, A±A and S±A. The spec does not say how the pair
terms are combined; the approved rule is

$$E_p(G_t) = \frac{1}{\binom{n}{2}} \sum_{(i,j) \in \text{edges}} E_{ij}^p$$

Plain text: E_p = (Σ over edges of E_ij^p) / C(n,2). Both energies are 0 when
there are too few nodes (n < 3 for E_delta, n < 2 for E_p).

### Local energy difference and acceptance (§3.3)

Flipping an existing edge, $A_{ij} \to -A_{ij}$, changes only the triads that
contain it:

$$\Delta E_\Delta(i, j) = \frac{2 A_{ij}^{\text{old}}}{\binom{n}{3}} \sum_{k} A_{ik} A_{jk} = -\frac{2 A_{ij}(t'')}{\binom{n}{3}} \sum_{k} A_{ik}(t') A_{jk}(t')$$

Plain text: dE_delta = +2·A_ij_old / C(n,3) · Σ_k A_ik·A_jk. This is the
spec's −2·A_ij(t'')/C(n,3)·Σ A_ik(t')·A_jk(t') with t'' = after the flip and
t' = before it. With node states fixed, $\Delta E_p = -2\, c_{ij} A_{ij}^{\text{old}} / \binom{n}{2}$,
where $c_{ij} = 1$ if exactly one endpoint is infected (plain text:
dE_p = −2·c_ij·A_ij_old / C(n,2)).

A flip is accepted if the change in the **full** energy E is negative, with
probability 1/2 if it is exactly 0, and never if it is positive. Ties are
detected with exact rational arithmetic.

### Where the spec's wording and the implementation differ

These are documented interpretations, not hidden changes:

1. **Rates, not probabilities (§3.1).** The spec speaks of "transition
   probabilities over discrete time intervals" but gives rates. The code is a
   continuous-time Markov chain driven by those rates (Gillespie algorithm,
   no fixed time step).
2. **Not O(1) (§3.3).** The section is titled "Fast O(1)", but the local update
   sums over the common neighbours of i and j. Its cost is
   O(|N(i) ∩ N(j)|) ≤ O(min(deg i, deg j)), which is O(n) on dense graphs. ΔE_p is O(1).
3. **Balanced = −1 only for complete graphs (§3.2).** On sparse graphs most
   triples are not triangles, so |E_delta| ≪ 1 (about 1e-4 on the example
   graphs).
4. **One symbol, two meanings.** The spec's ρ means both correlation (ρ_ij) and
   infection density (ρ_0, ρ∞). The code uses `corr` for correlations and
   `initial_infected_fraction` / `rho_inf` for densities.
5. **No metric for two named behaviours (§5).** "Natural immunization" and
   "clustering" are not defined in the spec, so no output is presented as
   demonstrating them.

### Approved deviations from the PDF and decisions on spec gaps

Every row below is a "Made" entry in [DECISIONS.md](DECISIONS.md), referenced by
phase and item.

| Topic | What the code does | Decision |
|---|---|---|
| Package layout | Package at `src/financial_contagion_sim/` instead of a bare `src/__init__.py`; extra files (`DECISIONS.md`, `docs/`, `cli.py`, `sweep.py`, `artifacts.py`, extra tests and scripts) and extra dev tools (ruff, mypy, build) | Phase 0 (2026-10-14 rows); Phase 4 E1 |
| Signed adjacency | A missing edge contributes A_ij = 0 | Phase 1 A8; Phase 2 B2 |
| r0 and r(t) | r = positive edges / m, assigned exactly as `round(r0 · m)` on the float product; r(t) is NaN when m = 0 | Phase 1 B1–B2 (rounding resolved 2026-10-14); Phase 3 C1 |
| Correlation graphs | Edge iff corr ≠ 0 and \|corr\| ≥ τ, sign = sign(corr), d_ij stored but not used for selection; one matrix per call | Phase 1 C1, C2, C6 |
| Transitions | Exactly the four §3.1 transitions; the diagram's "Recapitalized" arrow is I → S | Phase 2 A3 |
| Initial states | ρ0 is the initial infected fraction, `round(ρ0 · n)` nodes; no node starts in A (the initial A fraction is unspecified; still Open) | Phase 2 A7; Phase 3 A4 |
| E_p aggregation | Sum over edges divided by C(n,2); E_p = 0 for n < 2 and E_Δ = 0 for n < 3 | Phase 2 C2, B3 |
| §3.3 sign convention | t'' read as after the flip | Phase 2 D2 |
| Acceptance | Uses the full E; dE > 0 is always rejected (no temperature); ties detected exactly with `Fraction`, no tolerance | Phase 2 D4–D6 |
| Topology dynamics | Only signs of existing edges flip; no rewiring | Phase 3 B1 |
| Edge-update schedule | One joint CTMC; every edge proposes a flip at rate γ, so R_net = γ·m | Phase 3 B2 |
| Recording | Exact state on a time grid, or after every event | Phase 3 C3–C4 |
| Stopping | `stop_on_sais_absorbed` defaults to False; no convergence stopping | Phase 3 D1–D3 |
| Randomness | `rng` is a `Generator` or int seed, with no global seeding; the simulator uses five fixed child streams from `SeedSequence(seed)` | Phase 1 D2; Phase 3 E1–E3 |
| CLI (§4 names only "argparse") | `run` and `sweep` subcommands, required physical flags, exit codes 0/1/2/3 | Phase 4 C1–C8 |
| Output files (spec names only .html surfaces) | `config.json`, `metadata.json`, `metrics.csv`, PNG and sweep files; never overwritten | Phase 4 D1–D5 |
| Steady state | Last 20% of t_max, halves test, absolute tol 0.02 for ρ∞ and relative tol 0.05 for E | Phase 4 B3 |
| Sweep and surfaces | x = α, y = r0, one surface per ρ0; default 11 × 11 × 3 grid with 5 repetitions | Phase 4 B1–B2 |
| Data | Synthetic simulation output only; no downloads, no empirical data | Phase 1 C7; Phase 4 A5, D1 |
| Not implemented | "Natural immunization" / "clustering" metrics; CLI correlation input (`--corr-file`) | Phase 4 A9, C9 (Open) |

### References

As cited in the specification, with no further bibliographic detail:

- Kalimzhanov et al., *IEEE TKDE* (spec §1).
- Heider's Structural Balance Theory (spec §3.2).

## Layout

```
pyproject.toml              metadata, dependencies, tool configuration
requirements.txt            spec §4 package list (pyproject.toml is authoritative)
main.py                     thin wrapper around financial_contagion_sim.cli
src/financial_contagion_sim/
    __init__.py             __version__
    cli.py                  `run` and `sweep` commands (argparse)
    artifacts.py            run directories, config/metadata/metrics files
    sweep.py                parameter sweeps and steady-state estimation
    graph_builder.py        signed financial network construction
    sais_engine.py          SAIS rates and Gillespie simulation
    energy_calculator.py    structural, spreading and total energy; flip deltas
    simulator.py            co-evolutionary simulation (SAIS + edge-sign flips)
    visualizer.py           2D time series (Matplotlib), 3D surfaces (Plotly)
tests/
    test_smoke.py           import, version, CLI --help
    test_graph_builder.py   signed network loaders and validation
    test_energy.py          energies, local deltas vs exact recomputation
    test_sais.py            SAIS rates, conservation, Gillespie statistics
    test_simulator.py       configuration, invariants, co-evolution, streams
    test_visualizer.py      chart and surface data, HTML self-containment
    test_sweep.py           steady state, grid, seeding, resume, aggregation
    test_artifacts.py       files, run ids, overwrite and failure safety
    test_cli.py             commands, exit codes, end-to-end run and sweep
    test_integration.py     full pipelines; saved and plotted data vs memory
    test_invariants_system.py  consequence checks, recomputation, reproducibility
    test_reliability.py     boundaries, degenerate inputs, failure safety
    test_readme_examples.py runs the README blocks marked <!-- verify -->
    conftest.py             shared SimulationResult factory
scripts/
    benchmark_simulator.py  measured events per second (not a test)
    benchmark_suite.py      Phase 5 timing, memory and profiles (not a test)
DECISIONS.md                decision log
docs/spec/specification.pdf project specification
docs/TRACEABILITY.md        spec item -> code -> tests, mutation results
docs/RELEASE_CHECKLIST.md   review and production checklists with status
```

## Installation

Requires Python 3.11 or newer. It has been tested only with Python 3.14.6 on
Windows 11 (see "Tested environment" below).

To install the package and its runtime dependencies from a checkout:

```bash
python -m pip install .
```

This installs the `financial-contagion-sim` command. Installing the built
wheel into a fresh virtual environment outside the repository, then running
the commands under Usage from another directory, was verified in Phase 5.

## Development setup

For development (tests, linters, build tools), use an editable install with
the `dev` extra.

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies.
`requirements.txt` lists the six packages named in the spec (§4) for reference;
installing from it does not install this package. A smoke test fails if it
names a package that `pyproject.toml` does not declare.

## Checks

Run all of these before committing:

```bash
pytest -q
ruff check .
ruff format --check .
mypy src
python -m build
```

## Usage

The installed `financial-contagion-sim` command and `python main.py` (from a
checkout) are the same program. Every physical parameter is required; there
are no built-in model values. Blocks marked `<!-- verify -->` in this file are
executed by `tests/test_readme_examples.py`.

One simulation (writes `outputs/<run_id>/`):

<!-- verify -->
```bash
financial-contagion-sim run --model erdos_renyi --n 200 --er-p 0.05 --beta 1.0 --beta-a 0.2 --kappa 0.1 --delta 1.0 --gamma 1.0 --alpha 0.5 --r0 0.6 --rho0 0.1 --t-max 50 --seed 1
```

A parameter sweep over α × r0 × ρ0 (writes `outputs/<run_id>/` with a
`sweep/` folder). This example is deliberately tiny (2 × 2 × 1 grid, 2
repetitions, 8 runs). The default grid is 11 × 11 × 3 with 5 repetitions
(1,815 runs); set it with `--alpha-range`, `--r0-range`, `--rho0-values` and
`--reps`.

<!-- verify -->
```bash
financial-contagion-sim sweep --model erdos_renyi --n 50 --er-p 0.1 --beta 1.0 --beta-a 0.2 --kappa 0.1 --delta 1.0 --gamma 1.0 --t-max 10 --seed 1 --alpha-range 0 1 2 --r0-range 0 1 2 --rho0-values 0.1 --reps 2
```

Add `--dry-run` to either command to validate everything and print the plan
(for a sweep, the run count and a runtime estimate from one pilot run) without
writing anything.

| Option | Commands | Meaning |
|---|---|---|
| `--model {barabasi_albert,erdos_renyi}` | both | network topology (required) |
| `--n`, `--ba-m` / `--er-p` | both | nodes; BA edges per new node or ER edge probability |
| `--beta`, `--beta-a`, `--kappa`, `--delta` | both | SAIS rates (§3.1), 1/time; 0 ≤ β_a < β |
| `--gamma` | both | edge-flip proposal rate per edge, 1/time |
| `--alpha`, `--r0`, `--rho0` | `run` | energy weight, positive-link ratio, initial infected fraction |
| `--t-max`, `--seed` | both | time horizon; random seed (int ≥ 0) |
| `--max-events`, `--num-points` | both | control options; defaults come from `SimulationConfig` |
| `--stop-on-sais-absorbed` | `run` | stop when no SAIS jump is possible |
| `--alpha-range START STOP COUNT`, `--r0-range START STOP COUNT` | `sweep` | surface axes, endpoints included (default 0 1 11) |
| `--rho0-values V [V ...]` | `sweep` | one surface per value (default 0.05 0.1 0.2) |
| `--reps` | `sweep` | repetitions per grid point (default 5) |
| `--tol-rho`, `--tol-e` | `sweep` | steady-state tolerances (defaults 0.02 absolute, 0.05 relative) |
| `--max-runs` | `sweep` | refuse larger sweeps (default 2000) |
| `--resume` | `sweep` | continue an interrupted sweep |
| `--config FILE.json` | both | load a saved `config.json`; explicit flags override it |
| `--output-dir`, `--run-name` | both | parent directory (default `outputs/`); directory name |
| `--dry-run`, `--no-plot` / `--no-html` | both | write nothing; skip the PNG / the HTML surfaces |

Exit codes: **0** success; **1** runtime failure; **2** argument or
configuration error (the message names the option); **3** refused to
overwrite an existing run directory, or to resume an unfinished sweep without
`--resume` or with a different configuration.

Output layout (`outputs/` is git-ignored; nothing is ever overwritten):

```
outputs/
  run_<hash8>_seed<seed>/          (or the --run-name)
    config.json                    resolved configuration and seed
    metadata.json                  versions, git commit, UTC time, argv, counts
    metrics.csv                    time,S,A,I,r,E,E_delta,E_p,infected_fraction
    timeseries.png
  sweep_<hash8>_seed<seed>/
    config.json                    sweep definition (base, grid, repetitions, steady state)
    metadata.json
    sweep/
      sweep_results.csv            one row per (grid point, repetition)
      sweep_summary.json           grid, settings, non-converged counts, runtime
      surface_rho_infinity.html    self-contained (plotly.js embedded, ~4.8 MB)
      surface_energy.html
```

A run is built in a temporary `<run_id>.tmp-*` directory and renamed into
place only when complete. A sweep is built in `<run_id>.incomplete/`, which is
kept if the sweep is interrupted so that `--resume` can continue it.

From Python (every physical parameter must be given; there are no defaults):

<!-- verify -->
```python
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation

config = SimulationConfig(
    n=200,
    topology="erdos_renyi",
    er_p=0.05,
    sais_params=SAISParams(beta=0.5, beta_a=0.1, kappa=0.3, delta=1.0),
    alpha=0.5,
    r0=0.6,
    initial_infected_fraction=0.1,
    edge_update_rate=1.0,
    t_max=50.0,
    seed=0,
)
result = run_simulation(config)  # result.times, result.I, result.r, result.E, ...
```

`simulate(graph, initial_states, ...)` runs the same engine on a graph you
build yourself, such as one from `graph_from_correlation`.

## Co-evolution mechanics

The spec does not say how often edges are updated, so the approved model
(DECISIONS.md, Phase 3) is one continuous-time Markov chain with two channels:

- **SAIS jumps** at the §3.1 rates (total `R_sais`).
- **Edge-flip proposals**: every edge proposes a sign flip at rate γ
  (`edge_update_rate`), so `R_net = γ·m`. A proposal is accepted if it lowers
  the total energy E, with probability 1/2 if E is unchanged, and rejected if
  it raises E (§3.3). Only signs change; edges are never added or removed.

The waiting time to the next event is exponential with rate
`R_sais + R_net`, and the channel is chosen in proportion to its rate. There
is no fixed time step.

**Modelling note.** On sparse graphs most edges have no common neighbours and
no infected endpoint, so flipping them leaves E exactly unchanged. Those flips
are accepted half the time, and the signs of such edges random-walk towards a
friendly-link density of r = 0.5. This follows from the §3.3 rule; it is not a
bug.

## Steady state and sweeps

The spec does not define a steady state, so the approved rule (DECISIONS.md,
Phase 4 B3) is:

- Each run is recorded on its `num_points` time grid. The window is every grid
  time with t ≥ 0.8·t_max, split into two halves at 0.9·t_max.
- **ρ∞** (`rho_inf`) is the mean infected fraction over the window. It counts
  as converged if the two half-means differ by at most `--tol-rho`
  (absolute, default 0.02).
- **E_ss** is the mean energy E over the window. It counts as converged if the
  half-means differ by at most `--tol-e` × |window mean| (relative, default
  0.05). The tolerance is relative because E is of order 1e-4 on sparse
  graphs.
- An absorbed run is constant, so it counts as converged. A run that stops
  before t_max is not converged. A metric that does not converge is NaN, not
  its last value.
- A grid point's value is the mean over converged repetitions, provided at
  least half of them converged; otherwise it is NaN, shown as a hole in the
  surface. Hover text shows the standard deviation, the standard error and the
  converged count.
- Repetition `rep` of grid point `index` is seeded with
  `SeedSequence(entropy=seed, spawn_key=(index, rep))`, so results do not
  depend on execution order or on resuming.

**Reading the outputs.** These are properties of the spec's rules, not
findings:

- **Neutral flips:** flips that leave E unchanged are accepted half the time
  (§3.3). So edges with no triangles and no infected endpoint random-walk
  towards r = 0.5, regardless of contagion.
- **α = 1:** acceptance ignores node states, so topology evolves independently
  of infection.
- **α = 0:** only the spreading energy counts.

The spec's "natural immunization" and "clustering" behaviours have no defined
metric, so the charts make no claim about them.

## Tested environment

Everything in this repository was run and checked on one machine:

| Item | Version |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 |
| CPU | Intel Core Ultra 7 258V, 8 logical CPUs |
| Python | 3.14.6 (the only version tested; `requires-python >=3.11`, so 3.11–3.13 are untested) |
| numpy, scipy, networkx | 2.5.4, 1.18.1, 3.7 |
| matplotlib, plotly | 3.11.2, 7.1.0 |

The declared dependency lower bounds in `pyproject.toml` (for example
numpy ≥ 1.26, plotly ≥ 5.18) have not been tested. Results are reproducible
for the same configuration and seed on the same machine and package versions;
reproducibility across versions or platforms is not claimed.

## Performance

No performance targets exist in the spec, so performance is only measured,
never pass/fail. `scripts/benchmark_suite.py` runs the approved tiers (n from
100 to 5,000, Barabási–Albert and Erdős–Rényi, plus a reduced sweep), measures
memory in separate runs, and profiles the main code paths. It refuses to run
on battery power, because throttling changes the results several-fold.
**Measured numbers will be added here after a run on AC power; none are
quoted until then.**

Complexity, as implemented:
- **SAIS events:** a jump costs O(n) to rebuild the cached rates plus O(deg v)
  for the updates.
- **Flip proposals:** a proposal costs O(min(deg i, deg j)), and the flip
  itself is O(1) to apply.
- **Exact energy:** the dense reference engine `exact_structural_energy` is
  O(n³) time and O(n²) memory. It is used only as a test oracle; the
  simulator uses the sparse integer sums.

## Known limitations

- Synthetic simulation output only: there is no empirical data, and no
  validation against real financial markets or crises.
- "Natural immunization" and "clustering" (§5) have no metric, so they are
  not computed and not claimed.
- The CLI cannot read correlation matrices; use `graph_from_correlation` and
  `simulate()` from Python.
- The initial alert fraction is unspecified in the spec; all runs start with
  no node in state A.
- The sdist does not include `tests/conftest.py`, so its test suite cannot
  run as shipped (open packaging decision).
- Sweeps run serially. The self-contained HTML surfaces are about 4.8 MB each.
- Rates whose product with N_i overflows float64 (around 1.8e308) are
  rejected with a clear error.

## Workflow

- One branch per phase (`phase-N/<topic>`), merged to `main` by pull request.
- [Conventional Commits](https://www.conventionalcommits.org/); no amending,
  rebasing or force-pushing.
- Decisions are recorded in [DECISIONS.md](DECISIONS.md).
