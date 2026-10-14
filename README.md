# financial-contagion-sim

Co-evolutionary financial contagion & systemic risk simulator: signed
financial networks, SAIS contagion dynamics, structural-balance energy and
co-evolutionary Monte Carlo.

## Status

**Phase 2 (SAIS dynamics and network energy).** `graph_builder.py` (Phase 1)
builds and validates signed networks. `sais_engine.py` implements the SAIS
transition rates and Gillespie simulation on a fixed graph, and
`energy_calculator.py` implements the structural, spreading and total energies
and their local edge-flip differences. Each module docstring documents its
conventions. The co-evolutionary loop (`simulator.py`) and plotting
(`visualizer.py`) are still docstrings only, and the CLI offers `--help` and
`--version` only.

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
| Co-evolutionary Monte Carlo loop tracking S(t), A(t), I(t), r(t), E(G_t) (§5) | `simulator` | Not implemented |
| 2D state-trajectory plots (§5) | `visualizer` | Not implemented |
| 3D Plotly surface plots exported as `.html` (§5) | `visualizer` | Not implemented |
| CLI simulation commands (§4) | `cli` / `main.py` | Not implemented (`--help` and `--version` only) |

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
    cli.py                  argparse stub (--help, --version)
    graph_builder.py        signed financial network construction
    sais_engine.py          SAIS rates and Gillespie simulation
    energy_calculator.py    structural, spreading and total energy; flip deltas
    simulator.py            co-evolutionary Monte Carlo orchestration
    visualizer.py           matplotlib / plotly figures
tests/
    test_smoke.py           import, version, CLI --help
    test_graph_builder.py   signed network loaders and validation
    test_energy.py          energies, local deltas vs exact recomputation
    test_sais.py            SAIS rates, conservation, Gillespie statistics
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

```bash
python main.py --help
financial-contagion-sim --version
```

## Workflow

- One branch per phase (`phase-N/<topic>`), merged to `main` by pull request.
- [Conventional Commits](https://www.conventionalcommits.org/); no amending,
  rebasing or force-pushing.
- Decisions are recorded in [DECISIONS.md](DECISIONS.md).
