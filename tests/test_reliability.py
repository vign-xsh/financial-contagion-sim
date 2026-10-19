"""Reliability checks: boundaries, degenerate inputs, failure handling (Phase 5).

Every test here can fail if the behaviour it describes were broken.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_collected_test_ids_are_deterministic() -> None:
    """No test id may embed a memory address (``at 0x...``): such ids change on
    every run, so they cannot be selected, deselected or cached (defect D-1)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(REPO_ROOT / "tests"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ids = [line for line in proc.stdout.splitlines() if "::" in line]
    assert len(ids) > 900
    unstable = [i for i in ids if " at 0x" in i]
    assert unstable == []
