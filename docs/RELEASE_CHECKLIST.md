# Release checklist

Status vocabulary: **passed**, **failed**, **skipped** (with reason), **not
configured**, **not executed**. A check is *passed* only if it ran in the
Phase 5 session (2026-10-20, Windows 11, Python 3.14.6) with the command and
counts shown. Nothing has been tagged, released or published.

## Level 1: ready for review

| Item | Status | Evidence |
|---|---|---|
| Test suite, default hash seed (twice) | passed | `pytest -q`: 936 passed at the Phase 5 baseline; 1077 passed after the Phase 5 tests were added |
| Test suite, `PYTHONHASHSEED=0` and `12345` | passed | 936 passed each at the baseline; no outcome changed between runs (only 4 test ids differed, fixed as D-1) |
| Test suite with `-W error` | passed | `python -W error -m pytest -q -W error`: 1077 passed, no warnings |
| `ruff check .` | passed | All checks passed |
| `ruff format --check .` | passed | all files already formatted |
| `mypy src` | passed | no issues in 9 source files (strict; plotly imports exempt by a scoped override) |
| `python -m build` | passed | wheel and sdist built into a temporary outdir |
| Wheel contents | passed | package modules and dist-info only; no outputs, `.venv`, benchmarks, secrets or `__pycache__`; root `main.py` is not shipped (the console script is) |
| sdist contents | **failed** | contains source, tests, README and pyproject, but **omits `tests/conftest.py`** (defect D-3, class b), so the sdist's test suite cannot run; fixing it needs a packaging edit (MANIFEST.in or setuptools config) that has not been approved |
| Clean-venv install of the wheel | passed | fresh venv outside the repo, `pip install <wheel>` (dependencies from PyPI), `pip check` clean |
| Console script from another directory | passed | with the final 0.1.0 wheel: `financial-contagion-sim --version` prints `financial-contagion-sim 0.1.0`; a tiny `run` and `sweep` produced their artifact trees in that directory; `git_commit` recorded as null outside a repository |
| README examples | passed | `tests/test_readme_examples.py` (3 marked blocks) in the dev environment, and the same blocks with the 0.1.0 wheel in the clean venv (all exit 0) |
| Traceability | passed | `docs/TRACEABILITY.md`: 24 rows; all 89 referenced test ids exist; the 2 `uncovered` and 4 `deviates` rows each have an approved decision-log entry |
| Mutation (sensitivity) check | passed | 12 of 12 perturbations caught; none UNCAUGHT |
| Scientific consequence checks (B2) | passed | 9 checks in `tests/test_invariants_system.py` |
| Reproducibility (B5) | passed | identical results in one process, across two processes with different `PYTHONHASHSEED`, library vs CLI (byte-identical `metrics.csv`), and Phase 4 vs Phase 5 code for the same config and seed |
| Performance measurement (A2–A4) | **not executed** | `scripts/benchmark_suite.py` refuses to run on battery power (approved rule A3); the machine was on battery throughout the session |
| Deviations from the PDF documented | passed | README "Approved deviations" table cross-checked against DECISIONS.md |

**Level 1 is not yet reached**: the performance measurement has not been
executed and D-3 (sdist omits `tests/conftest.py`) is unresolved.

## Level 2: production ready

Everything in Level 1, plus:

| Item | Status | Notes |
|---|---|---|
| License (LICENSE file and SPDX id) | not executed | not supplied; the package is **not distributable** |
| Authors | not executed | not supplied |
| Project URLs | not executed | not supplied |
| Classifiers and keywords | not executed | not supplied |
| Version agreed | passed | 0.1.0 (review candidate), single source `financial_contagion_sim.__version__`, test `test_version_has_one_source_of_truth` |
| CI decision | passed | decided: no CI configuration (Phase 5 E6) |
| No Open scientific decisions | **failed** | Open: initial A fraction (default A = 0), "natural immunization" / "clustering" metrics |
| Supported Python versions stated only as tested | passed | tested: 3.14.6 only; 3.11–3.13 untested although `requires-python >=3.11` |
| Supported platforms stated only as tested | passed | tested: Windows 11 (10.0.26200) only |
| Dependency lower bounds | not executed | declared bounds (e.g. numpy >= 1.26, plotly >= 5.18) were never tested; only the installed versions were (N1) |

**Level 2 is not reached.** The words "release ready" and "production ready"
do not apply to this version.
