# financial-contagion-sim

Co-evolutionary financial contagion & systemic risk simulator: signed
financial networks, SAIS contagion dynamics, structural-balance energy and
co-evolutionary Monte Carlo.

## Status

**Phase 0 (scaffolding) only.** The repository, packaging, environment and
baseline checks are in place. No simulator mathematics is implemented: every
module under `src/financial_contagion_sim/` except `cli.py` is a docstring
describing its future responsibility, and the CLI offers `--help` and
`--version` only.

## Layout

```
pyproject.toml              metadata, dependencies, tool configuration
requirements.txt            spec §4 package list (pyproject.toml is authoritative)
main.py                     thin wrapper around financial_contagion_sim.cli
src/financial_contagion_sim/
    __init__.py             __version__
    cli.py                  argparse stub (--help, --version)
    graph_builder.py        signed financial network construction
    sais_engine.py          SAIS contagion dynamics
    energy_calculator.py    structural-balance energy
    simulator.py            co-evolutionary Monte Carlo orchestration
    visualizer.py           matplotlib / plotly figures
tests/
    test_smoke.py           import, version, CLI --help
    test_energy.py          skipped until Phase 1
    test_sais.py            skipped until Phase 1
DECISIONS.md                decision log
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
