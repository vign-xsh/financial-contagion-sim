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
| 2026-10-14 | Spec §4 shows `src/*.py` with a bare `src/__init__.py`. Deliberate deviation: the package lives at `src/financial_contagion_sim/`. | A bare `src/__init__.py` is not an importable package. |
| 2026-10-14 | Extra files not in spec §4: `DECISIONS.md`, `tests/test_smoke.py`, `docs/spec/`, and `src/financial_contagion_sim/cli.py` (see the 2026-10-11 entry). | Decision log, Phase 0 smoke checks, a versioned copy of the spec, and the installable CLI entry point. |
| 2026-10-14 | Extra dev tools not in the spec: ruff, mypy, build. | Linting, formatting, static typing and packaging checks. |
| 2026-10-14 | Naming: the spec uses ρ for correlation (`ρ_ij`) and for infection density (`ρ_∞`, `ρ_0`). Code uses `corr` for correlations and `infected_fraction` / `initial_infected_fraction` for densities. | Avoids one symbol meaning two things in code. |
| 2026-10-14 | Supersedes the 2026-10-11 `requirements.txt` entry: it now lists the six spec §4 packages with lower bounds matching `pyproject.toml`, which stays authoritative; development installs use `pip install -e ".[dev]"`. A smoke test guards against drift. | Spec §4 alignment. |

## Open

| Raised     | Question | Notes |
|------------|----------|-------|
| 2026-10-11 | Which license, if any? | None declared; the repository is private. Needed before any public release. |
| 2026-10-11 | Should CI (GitHub Actions) run the checks on pull requests? | Workflows need explicit approval; not part of Phase 0. |
| 2026-10-11 | `python` and `pip` are not on PATH on the development machine (only the `py` launcher). | Harmless inside an activated `.venv`; bootstrap with `py -3 -m venv .venv`. |
| 2026-10-14 | Edge-flip rate and scheduling in the co-evolutionary loop are unspecified (§5). | No default proposed. |
| 2026-10-14 | Is r(t) "friendly link density" positive edges / m or positive edges / C(n,2)? | Proposed default: positive / m. |
| 2026-10-14 | Initial fraction of nodes in state A is unspecified. | Proposed default: A = 0 initially, S = 1 − ρ_0. |
| 2026-10-14 | Meaning of "dynamic" stock return correlation matrices in §5. | Proposed default: one matrix per call. |
| 2026-10-14 | Aggregation and normalization of E_p are unspecified (§3.2). | Proposed default: sum of E_ij^p over existing edges, divided by C(n,2). |
| 2026-10-14 | The §3.1 diagram shows a "Recapitalized" arrow near β_a, but the text lists only four transitions. | Proposed default: exactly the four listed transitions; no A → S. |
| 2026-10-14 | Sign convention of the §3.3 local formula: A_ij(t'') vs A_ij(t'). | No default proposed. |
| 2026-10-14 | Steady-state definition, the "clustering" and "natural immunization" metrics, and which two of (α, r_0, ρ_0) are the 3D surface axes (§5). | No default proposed. |
