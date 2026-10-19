"""Execute the README examples marked ``<!-- verify -->`` (Phase 5 F3).

Each marked fenced block runs in a fresh temporary directory: ``bash`` blocks
through the installed ``financial-contagion-sim`` console script, ``python``
blocks with the current interpreter. The test checks the exit code and the
artifacts the example promises.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
BLOCK = re.compile(r"<!-- verify -->\s*\n```(\w+)\n(.*?)\n```", re.S)
COMMAND = "financial-contagion-sim"


def _blocks() -> list[tuple[str, str]]:
    return BLOCK.findall(README.read_text(encoding="utf-8"))


def _console_script() -> str:
    found = shutil.which(COMMAND, path=str(Path(sys.executable).parent))
    if found is None:
        found = shutil.which(COMMAND)
    assert found is not None, f"{COMMAND} is not installed in this environment"
    return found


def test_readme_has_marked_examples() -> None:
    languages = [language for language, _ in _blocks()]
    assert languages.count("bash") >= 2 and "python" in languages


@pytest.mark.parametrize("index", range(len(_blocks())))
def test_readme_example_runs(index: int, tmp_path: Path) -> None:
    language, code = _blocks()[index]
    if language == "bash":
        argv = code.split()
        assert argv[0] == COMMAND, "verified shell examples use the console script"
        proc = subprocess.run(
            [_console_script(), *argv[1:]],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        assert language == "python"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    if language == "bash":
        (run_dir,) = (tmp_path / "outputs").iterdir()
        if argv[1] == "run":
            expected = {"config.json", "metadata.json", "metrics.csv", "timeseries.png"}
            assert {p.name for p in run_dir.iterdir()} == expected
        else:
            files = {p.name for p in (run_dir / "sweep").iterdir()}
            assert files == {
                "sweep_results.csv",
                "sweep_summary.json",
                "surface_rho_infinity.html",
                "surface_energy.html",
            }
