# Decision log

## Made

| Date       | Decision | Rationale |
|------------|----------|-----------|
| 2026-10-11 | GitHub repository `vign-xsh/financial-contagion-sim`, private, default branch `main`. | Pre-authorized in the Phase 0 brief. |
| 2026-10-11 | Recreated the repository from scratch. The previous repository of the same name (Phase 0 + Phase 1 history) was deleted by the owner and the local folder emptied first. | Owner's instruction. |
| 2026-10-11 | Python `>=3.11`; development interpreter is CPython 3.14. | Brief: newest 3.x available locally. |
| 2026-10-11 | Environment is `.venv` created with `python -m venv`; pip + `pyproject.toml` (PEP 621, setuptools backend). | Pre-authorized. |
| 2026-10-11 | `src` layout, package `financial_contagion_sim`. | Pre-authorized. |
| 2026-10-11 | Runtime dependencies: networkx, numpy, scipy, matplotlib, plotly. Dev extra: pytest, ruff, mypy, build. | Pre-authorized. |
| 2026-10-11 | `pyproject.toml` is the single source of truth; `requirements.txt` contains only `-e .[dev]`. | Pre-authorized. |
| 2026-10-11 | Version is defined once, in `financial_contagion_sim.__version__`, and read by setuptools (`dynamic = ["version"]`). | Avoids two copies drifting apart. |
| 2026-10-11 | The argparse stub lives in `src/financial_contagion_sim/cli.py`; root `main.py` is a wrapper around it. This adds one file to the layout in the brief. | `[project.scripts]` needs an importable entry point inside the installed package; a root-level `main.py` is not installable under a `src` layout. |
| 2026-10-11 | mypy runs in `strict` mode on `src`; ruff lint rules `E, F, W, I, B, UP, SIM`, line length 88. | Start strict while there is no code to retrofit. |
| 2026-10-11 | Conventional Commits; never amend, rebase or force-push. One branch per phase, pull request to `main`. | Pre-authorized. |

## Open

| Raised     | Question | Notes |
|------------|----------|-------|
| 2026-10-11 | Which license, if any? | None declared; the repository is private. Needed before any public release. |
| 2026-10-11 | Should CI (GitHub Actions) run the checks on pull requests? | Workflows need explicit approval; not part of Phase 0. |
| 2026-10-11 | `python` and `pip` are not on PATH on the development machine (only the `py` launcher). | Harmless inside an activated `.venv`; bootstrap with `py -3 -m venv .venv`. |
