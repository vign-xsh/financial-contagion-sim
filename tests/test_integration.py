"""Full-system integration tests (Phase 5 Part A).

Each pipeline runs end to end on small deterministic networks and checks that
saved files equal the in-memory results and that plotted data equals saved
data. All output is synthetic simulation data.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from financial_contagion_sim import cli
from financial_contagion_sim.artifacts import (
    config_from_dict,
    read_metrics_csv,
    write_metrics_csv,
    write_run_artifacts,
)
from financial_contagion_sim.graph_builder import graph_from_correlation
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation, simulate
from financial_contagion_sim.sweep import (
    SteadyStateSettings,
    SweepGrid,
    append_record,
    load_sweep,
    run_sweep,
    sweep_definition,
    write_results_header,
)
from financial_contagion_sim.visualizer import (
    plot_surface,
    plot_timeseries,
    save_figure,
    write_surface_html,
)

PARAMS = SAISParams(beta=1.0, beta_a=0.2, kappa=0.1, delta=1.0)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Hand-built correlation matrix: symmetric, unit diagonal, entries in [-1, 1].
CORR = np.array(
    [
        [1.0, 0.8, 0.7, -0.6, 0.1],
        [0.8, 1.0, 0.65, -0.55, 0.2],
        [0.7, 0.65, 1.0, -0.5, 0.0],
        [-0.6, -0.55, -0.5, 1.0, 0.75],
        [0.1, 0.2, 0.0, 0.75, 1.0],
    ]
)


def _config(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "n": 30,
        "topology": "erdos_renyi",
        "er_p": 0.2,
        "sais_params": PARAMS,
        "alpha": 0.5,
        "r0": 0.6,
        "initial_infected_fraction": 0.2,
        "edge_update_rate": 1.0,
        "t_max": 3.0,
        "seed": 5,
        "num_points": 61,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def _assert_metrics_equal_result(
    metrics: dict[str, np.ndarray], result: object
) -> None:
    for column, attribute in (
        ("time", "times"),
        ("S", "S"),
        ("A", "A"),
        ("I", "I"),
        ("r", "r"),
        ("E", "E"),
        ("E_delta", "E_delta"),
        ("E_p", "E_p"),
        ("infected_fraction", "infected_fraction"),
    ):
        expected = getattr(result, attribute)
        assert np.array_equal(metrics[column], expected, equal_nan=True), column


def _assert_plot_equals_metrics(result: object, metrics: dict[str, np.ndarray]) -> None:
    fig = plot_timeseries(result)
    states_ax, r_ax, energy_ax = fig.axes
    for line, column in zip(states_ax.get_lines(), ("S", "A", "I"), strict=True):
        assert np.array_equal(
            line.get_xydata(), np.column_stack([metrics["time"], metrics[column]])
        )
    assert np.array_equal(
        r_ax.get_lines()[0].get_xydata(),
        np.column_stack([metrics["time"], metrics["r"]]),
        equal_nan=True,
    )
    assert np.array_equal(
        energy_ax.get_lines()[0].get_xydata(),
        np.column_stack([metrics["time"], metrics["E"]]),
    )


def test_correlation_matrix_to_saved_metrics_and_png(tmp_path: Path) -> None:
    corr_before = CORR.copy()
    graph = graph_from_correlation(CORR, tau=0.5)
    signs_before = sorted(graph.edges(data="sign"))
    states = np.array([-1, 1, 1, 1, 1], dtype=np.int8)
    states_before = states.copy()

    result = simulate(graph, states, PARAMS, 0.5, 1.0, 4.0, seed=3, verify_every=1)
    metrics_path = tmp_path / "metrics.csv"
    write_metrics_csv(metrics_path, result)
    png_path = tmp_path / "timeseries.png"
    save_figure(plot_timeseries(result), png_path)

    metrics = read_metrics_csv(metrics_path)
    _assert_metrics_equal_result(metrics, result)
    _assert_plot_equals_metrics(result, metrics)
    assert png_path.read_bytes().startswith(PNG_SIGNATURE)
    assert result.edges.tolist() == [[u, v] for u, v, _ in signs_before]
    assert result.initial_signs.tolist() == [s for _, _, s in signs_before]
    # Nothing the caller passed in was modified anywhere in the pipeline.
    assert np.array_equal(CORR, corr_before)
    assert sorted(graph.edges(data="sign")) == signs_before
    assert np.array_equal(states, states_before)


def test_config_to_artifacts_to_cli_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config()
    result = run_simulation(config)
    library_dir = write_run_artifacts(
        tmp_path / "lib", "r", result, argv=["test"], plot=True
    )
    library_metrics = read_metrics_csv(library_dir / "metrics.csv")
    _assert_metrics_equal_result(library_metrics, result)
    _assert_plot_equals_metrics(result, library_metrics)

    saved = json.loads((library_dir / "config.json").read_text(encoding="utf-8"))
    assert config_from_dict(saved["config"]) == config
    code = cli.main(
        [
            "run",
            "--config",
            str(library_dir / "config.json"),
            "--output-dir",
            str(tmp_path / "cli"),
        ]
    )
    capsys.readouterr()
    assert code == 0
    (cli_dir,) = (tmp_path / "cli").iterdir()
    assert (cli_dir / "metrics.csv").read_bytes() == (
        library_dir / "metrics.csv"
    ).read_bytes()
    assert (cli_dir / "config.json").read_bytes() == (
        library_dir / "config.json"
    ).read_bytes()


def test_reduced_sweep_saved_and_plotted_data_agree(tmp_path: Path) -> None:
    base = _config(n=15, er_p=0.3, t_max=2.0, num_points=41)
    grid = SweepGrid(alpha=(0.0, 0.5, 1.0), r0=(0.3, 0.7), rho0=(0.2,))
    settings = SteadyStateSettings()
    results_csv = tmp_path / "sweep_results.csv"
    write_results_header(results_csv)
    result = run_sweep(
        base, grid, 2, settings, on_run=lambda r: append_record(results_csv, r)
    )

    reloaded = load_sweep(sweep_definition(base, grid, 2, settings), results_csv)
    assert reloaded.config_hash == result.config_hash
    for metric in ("rho_inf", "energy"):
        saved = getattr(reloaded, metric)
        live = getattr(result, metric)
        for field in ("mean", "std", "se", "k_converged"):
            assert np.array_equal(
                getattr(saved, field), getattr(live, field), equal_nan=True
            )
        fig = plot_surface(reloaded, metric)
        for p, trace in enumerate(fig.data):
            assert np.array_equal(
                np.asarray(trace.z, dtype=float), saved.mean[p], equal_nan=True
            )
        html = tmp_path / f"{metric}.html"
        write_surface_html(fig, html)
        assert html.stat().st_size > 1_000_000
    assert len(result.runs) == grid.n_points * 2
    assert all(math.isfinite(r.alpha) for r in result.runs)
