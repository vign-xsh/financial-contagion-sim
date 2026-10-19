"""Reliability checks: boundaries, degenerate inputs, failure handling (Phase 5).

Every test here can fail if the behaviour it describes were broken.
"""

import dataclasses
import math
import subprocess
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim import artifacts as artifacts_module
from financial_contagion_sim import cli
from financial_contagion_sim.graph_builder import (
    generate_erdos_renyi,
    graph_from_correlation,
)
from financial_contagion_sim.sais_engine import (
    SAISParams,
    compute_rates,
    infected_positive_neighbors,
    make_initial_state,
)
from financial_contagion_sim.simulator import SimulationConfig, run_simulation, simulate
from financial_contagion_sim.sweep import SteadyStateSettings
from financial_contagion_sim.visualizer import plot_timeseries

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


# --- configuration boundaries (at and just beyond) ---------------------------


PARAMS = SAISParams(beta=1.0, beta_a=0.2, kappa=0.3, delta=0.7)
BASE = SimulationConfig(
    n=10,
    topology="erdos_renyi",
    er_p=0.3,
    sais_params=PARAMS,
    alpha=0.5,
    r0=0.5,
    initial_infected_fraction=0.2,
    edge_update_rate=1.0,
    t_max=1.0,
    seed=0,
)
up, down = (
    (lambda x: math.nextafter(x, math.inf)),
    (lambda x: math.nextafter(x, -math.inf)),
)

CONFIG_BOUNDARIES = [
    # field, valid boundary values, invalid values just beyond
    ("n", [1], [0]),
    ("er_p", [0.0, 1.0], [down(0.0), up(1.0)]),
    ("alpha", [0.0, 1.0], [down(0.0), up(1.0)]),
    ("r0", [0.0, 1.0], [down(0.0), up(1.0)]),
    ("initial_infected_fraction", [0.0, 1.0], [down(0.0), up(1.0)]),
    ("edge_update_rate", [0.0], [down(0.0), math.inf]),
    ("t_max", [0.0], [down(0.0), math.inf]),
    ("seed", [0], [-1]),
    ("max_events", [0, None], [-1]),
    ("num_points", [2], [1]),
    ("verify_every", [1, None], [0]),
]


@pytest.mark.parametrize(
    ("field", "valid", "invalid"),
    CONFIG_BOUNDARIES,
    ids=[c[0] for c in CONFIG_BOUNDARIES],
)
def test_config_boundaries(
    field: str, valid: list[object], invalid: list[object]
) -> None:
    for value in valid:
        dataclasses.replace(BASE, **{field: value})
    for value in invalid:
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(BASE, **{field: value})


def test_ba_m_boundaries() -> None:
    ba = dataclasses.replace(BASE, topology="barabasi_albert", er_p=None, ba_m=9)
    assert ba.ba_m == 9  # n - 1 is the largest valid value
    for bad in (10, 0):
        with pytest.raises(ValueError, match="ba_m"):
            dataclasses.replace(ba, ba_m=bad)


SAIS_BOUNDARIES = [
    ({"beta": up(0.0), "beta_a": 0.0}, None),
    ({"beta": 0.0, "beta_a": 0.0}, "beta must be > 0"),
    ({"beta": 1.0, "beta_a": down(1.0)}, None),
    ({"beta": 1.0, "beta_a": 1.0}, "beta_a"),
    ({"beta": 1.0, "beta_a": down(0.0)}, "beta_a"),
    ({"kappa": 0.0}, None),
    ({"kappa": down(0.0)}, "kappa"),
    ({"delta": 0.0}, None),
    ({"delta": down(0.0)}, "delta"),
]


@pytest.mark.parametrize(("change", "error"), SAIS_BOUNDARIES)
def test_sais_parameter_boundaries(change: dict[str, float], error: str | None) -> None:
    values = {"beta": 1.0, "beta_a": 0.2, "kappa": 0.3, "delta": 0.7, **change}
    if error is None:
        SAISParams(**values)
    else:
        with pytest.raises(ValueError, match=error):
            SAISParams(**values)


@pytest.mark.parametrize(
    ("kwargs", "ok"),
    [
        ({"window_start": up(0.0)}, True),
        ({"window_start": down(1.0)}, True),
        ({"window_start": 0.0}, False),
        ({"window_start": 1.0}, False),
        ({"tol_rho": 0.0, "tol_e_rel": 0.0}, True),
        ({"tol_rho": down(0.0)}, False),
        ({"min_converged_fraction": 1.0}, True),
        ({"min_converged_fraction": 0.0}, False),
    ],
)
def test_steady_state_setting_boundaries(kwargs: dict[str, float], ok: bool) -> None:
    if ok:
        SteadyStateSettings(**kwargs)
    else:
        with pytest.raises(ValueError):
            SteadyStateSettings(**kwargs)


# --- degenerate inputs -------------------------------------------------------


def _check(res: object, n: int) -> None:
    assert np.all(n == res.S + res.A + res.I)
    assert res.times[0] == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 1},
        {"n": 2},
        {"n": 3},
        {"er_p": 0.0},
        {"initial_infected_fraction": 0.0},
        {"initial_infected_fraction": 1.0},
        {"t_max": 0.0},
        {"max_events": 0},
        {"max_events": 1},
        {"r0": 0.0},
        {"r0": 1.0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"edge_update_rate": 0.0, "initial_infected_fraction": 0.0},
    ],
    ids=[
        "n1",
        "n2",
        "n3",
        "m0",
        "no-infected",
        "all-infected",
        "t_max0",
        "max_events0",
        "max_events1",
        "r0=0",
        "r0=1",
        "alpha0",
        "alpha1",
        "absorbing",
    ],
)
@pytest.mark.parametrize("record", ["grid", "events"])
def test_degenerate_runs_are_valid(overrides: dict[str, object], record: str) -> None:
    res = run_simulation(
        dataclasses.replace(BASE, verify_every=1, record=record, **overrides)
    )
    _check(res, res.config.n)
    if res.config.er_p == 0.0 or res.config.n == 1:
        assert len(res.edges) == 0 and np.all(np.isnan(res.r))


def test_barabasi_albert_smallest_graph() -> None:
    res = run_simulation(
        dataclasses.replace(
            BASE, n=2, topology="barabasi_albert", er_p=None, ba_m=1, verify_every=1
        )
    )
    assert len(res.edges) == 1
    _check(res, 2)


@pytest.mark.parametrize(
    "params",
    [
        SAISParams(1.0, down(1.0), 0.3, 0.7),
        SAISParams(1e-12, 0.0, 1e-12, 1e-12),
        SAISParams(1e12, 1e11, 1e12, 1e12),
    ],
    ids=["beta_a-just-below-beta", "tiny-rates", "huge-rates"],
)
def test_numerical_edge_rates(params: SAISParams) -> None:
    res = run_simulation(
        dataclasses.replace(BASE, sais_params=params, max_events=500, verify_every=1)
    )
    _check(res, 10)
    assert res.n_events <= 500


# --- failures propagate, nothing partial is left, inputs are untouched -------


def test_component_error_propagates_with_its_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import financial_contagion_sim.simulator as simulator_module

    def broken(*args: object) -> tuple[int, int]:
        raise ArithmeticError("energy component failed")

    monkeypatch.setattr(simulator_module, "flip_energy_numerators", broken)
    with pytest.raises(ArithmeticError, match="energy component failed"):
        run_simulation(BASE)


def test_cli_failure_mid_write_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path, result: object) -> None:
        path.write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(artifacts_module, "write_metrics_csv", broken)
    argv = [
        "run",
        "--model",
        "erdos_renyi",
        "--n",
        "10",
        "--er-p",
        "0.3",
        "--beta",
        "1",
        "--beta-a",
        "0.2",
        "--kappa",
        "0.3",
        "--delta",
        "0.7",
        "--gamma",
        "1",
        "--alpha",
        "0.5",
        "--r0",
        "0.5",
        "--rho0",
        "0.2",
        "--t-max",
        "1",
        "--seed",
        "0",
        "--output-dir",
        str(tmp_path),
    ]
    with pytest.raises(OSError, match="disk full"):
        cli.main(argv)
    assert list(tmp_path.iterdir()) == []


def test_pipeline_never_mutates_caller_data() -> None:
    corr = np.array([[1.0, 0.7, -0.6], [0.7, 1.0, 0.0], [-0.6, 0.0, 1.0]])
    corr_copy = corr.copy()
    g = graph_from_correlation(corr, tau=0.5)
    g2 = generate_erdos_renyi(15, 0.3, 0.5, rng=1)
    states = make_initial_state(15, 0.3, rng=1).states
    snapshot = (sorted(g2.edges(data="sign")), states.copy())
    res = simulate(g2, states, PARAMS, 0.5, 2.0, 3.0, seed=0)
    arrays = {k: getattr(res, k).copy() for k in ("times", "S", "I", "r", "E")}
    plot_timeseries(res)
    assert np.array_equal(corr, corr_copy) and g.number_of_edges() == 2
    assert sorted(g2.edges(data="sign")) == snapshot[0]
    assert np.array_equal(states, snapshot[1])
    assert all(
        np.array_equal(getattr(res, k), v, equal_nan=True) for k, v in arrays.items()
    )
