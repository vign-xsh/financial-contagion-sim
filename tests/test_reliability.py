"""Reliability checks: boundaries, degenerate inputs, failure handling (Phase 5).

Every test here can fail if the behaviour it describes were broken.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.sais_engine import (
    SAISParams,
    compute_rates,
    infected_positive_neighbors,
)
from financial_contagion_sim.simulator import simulate

REPO_ROOT = Path(__file__).resolve().parent.parent


def _graph(n: int, edges: list[tuple[int, int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def test_rate_overflow_is_reported_clearly() -> None:
    """beta is finite, but beta * N_i overflows float64 (defect D-2)."""
    g = _graph(3, [(0, 1, 1), (0, 2, 1)])
    states = np.array([1, -1, -1], dtype=np.int8)
    params = SAISParams(beta=1.7e308, beta_a=0.0, kappa=0.0, delta=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning may leak out either
        with pytest.raises(ValueError, match="overflow"):
            compute_rates(states, infected_positive_neighbors(g, states), params)
        with pytest.raises(ValueError, match="overflow"):
            simulate(g, states, params, 0.5, 0.0, 1.0, seed=0)
    # Large but representable rates still work.
    big = SAISParams(beta=1e300, beta_a=0.0, kappa=0.0, delta=1.0)
    res = simulate(g, states, big, 0.5, 0.0, 1.0, seed=0, max_events=3)
    assert res.n_events == 3


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
