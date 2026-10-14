"""Phase 0 smoke checks: the package imports, is versioned, and the CLI starts."""

import importlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import financial_contagion_sim
from financial_contagion_sim import cli

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

MODULES = [
    "cli",
    "energy_calculator",
    "graph_builder",
    "sais_engine",
    "simulator",
    "visualizer",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    module = importlib.import_module(f"financial_contagion_sim.{name}")
    assert module.__doc__


def test_version_is_pep440_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", financial_contagion_sim.__version__)


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    assert "financial-contagion-sim" in capsys.readouterr().out


def test_main_py_help_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    assert match, f"unparseable requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_requirements_txt_is_subset_of_pyproject() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = tomllib.loads(pyproject)["project"]
    declared = {_requirement_name(r) for r in project["dependencies"]}
    for extra in project["optional-dependencies"].values():
        declared |= {_requirement_name(r) for r in extra}

    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    listed = {
        _requirement_name(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert listed, "requirements.txt lists no packages"
    assert listed <= declared, (
        f"in requirements.txt but not pyproject.toml: {sorted(listed - declared)}"
    )
