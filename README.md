# financial-contagion-sim

Co-evolutionary financial contagion & systemic risk simulator: signed
financial networks, SAIS contagion dynamics, structural-balance energy and
co-evolutionary Monte Carlo.

## Status

**Phase 4 (visualization, sweeps and CLI).** `graph_builder.py` (Phase 1)
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

## Mathematical summary (spec §3)

Summarized from the project specification
([docs/spec/specification.pdf](docs/spec/specification.pdf)). Points the spec
leaves ambiguous are tracked under "Open" in [DECISIONS.md](DECISIONS.md).

### States and transitions (§3.1)

Each node i ∈ {1, ..., n} has a discrete state X_i(t) ∈ {1 (S), 0 (A), −1 (I)}:
S is solvent and unhedged, A is capital-buffered / hedged, I is distressed /
defaulted. Transitions follow a continuous-time Markov chain, where
N_i^{I,+}(t) is the number of defaulted friendly (+1) counterparties of node i:

| Transition | Meaning | Rate |
|---|---|---|
| S → I | Default cascade | β · N_i^{I,+}(t) |
| A → I | Attenuated default | β_a · N_i^{I,+}(t), with 0 ≤ β_a < β |
| S → A | Precautionary hedging / awareness | κ · N_i^{I,+}(t) |
| I → S | Recapitalization / bailout | δ |

### Energy (§3.2)

Total network energy, with α ∈ [0, 1] trading structural balance against
epidemic control:

```
E(G_t) = α · E_Δ(G_t) + (1 − α) · E_p(G_t)
```

Pairwise spreading energy E_ij^p for a pair (i, j) with edge sign A_ij ∈ {−1, +1}:

| Link | E_ij^p |
|---|---|
| Unbalanced: S+I, A+I (active shock transmission) | +1 |
| Balanced: S−I, A−I (protective hedge) | −1 |
| Neutral: S±S, I±I, A±A, S±A | 0 |

Triad structural-balance energy (Heider). A triad (i, j, k) is balanced when
A_ij · A_jk · A_ki = +1:

```
E_Δ(G_t) = −1 / C(n, 3) · Σ A_ij(t) · A_jk(t) · A_ki(t)
```

A perfectly balanced network gives E_Δ = −1; a completely frustrated network
gives E_Δ = +1.

### Local energy difference (§3.3)

Recomputing E_Δ over all C(n, 3) triads costs O(n^3). When edge (i, j) flips
sign (A_ij → −A_ij), the local difference is used instead:

```
ΔE_Δ(i, j) = −2 A_ij(t'') / C(n, 3) · Σ A_ik(t') · A_jk(t')
```

A flip is accepted if ΔE(i, j) < 0; if ΔE(i, j) = 0 it is accepted with
probability 0.5 to avoid freezing in local energy minima.

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
    conftest.py             shared SimulationResult factory
scripts/
    benchmark_simulator.py  measured events per second (not a test)
DECISIONS.md                decision log
docs/spec/specification.pdf project specification
```

## Development setup

Requires Python 3.11 or newer.

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

`python main.py` and the installed `financial-contagion-sim` command are the
same program. Every physical parameter is required; there are no built-in
model values.

One simulation (writes `outputs/<run_id>/`):

```bash
python main.py run --model erdos_renyi --n 200 --er-p 0.05 --beta 1.0 --beta-a 0.2 --kappa 0.1 --delta 1.0 --gamma 1.0 --alpha 0.5 --r0 0.6 --rho0 0.1 --t-max 50 --seed 1
```

A parameter sweep over α × r0 × ρ0 (writes `outputs/<run_id>/` with a
`sweep/` folder). The default grid is 11 × 11 × 3 with 5 repetitions
(1,815 runs); this small example has 36:

```bash
python main.py sweep --model erdos_renyi --n 100 --er-p 0.1 --beta 1.0 --beta-a 0.2 --kappa 0.1 --delta 1.0 --gamma 1.0 --t-max 30 --seed 1 --alpha-range 0 1 3 --r0-range 0 1 3 --rho0-values 0.05 0.2 --reps 2
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

## Workflow

- One branch per phase (`phase-N/<topic>`), merged to `main` by pull request.
- [Conventional Commits](https://www.conventionalcommits.org/); no amending,
  rebasing or force-pushing.
- Decisions are recorded in [DECISIONS.md](DECISIONS.md).
