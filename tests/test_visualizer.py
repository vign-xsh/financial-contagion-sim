"""Tests for the visualization engine (``visualizer``); spec §5, step 3."""

from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from financial_contagion_sim.simulator import SimulationResult
from financial_contagion_sim.sweep import (
    MetricSurface,
    SteadyStateSettings,
    SweepGrid,
    SweepResult,
)
from financial_contagion_sim.visualizer import (
    SYNTHETIC_CAPTION,
    plot_surface,
    plot_timeseries,
    save_figure,
    write_surface_html,
)

Factory = Callable[..., SimulationResult]

# Distinct values in every series, so a swapped line is detectable.
TIMES = [0.0, 1.0, 2.5, 4.0]
S = [9, 7, 6, 5]
A = [0, 1, 1, 3]
I = [1, 2, 3, 2]  # noqa: E741
R = [0.6, 0.55, 0.5, 0.45]
E = [-0.01, -0.02, -0.015, -0.03]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def result(make_result: Factory) -> SimulationResult:
    return make_result(TIMES, S, A, I, R, E)


def _lines(fig: Figure) -> list[list[tuple[str, np.ndarray]]]:
    return [
        [(str(line.get_label()), line.get_xydata()) for line in ax.get_lines()]
        for ax in fig.axes
    ]


def test_each_line_is_exactly_its_result_array(result: SimulationResult) -> None:
    fig = plot_timeseries(result)
    states, r_lines, e_lines = _lines(fig)
    expected_states = [("S (susceptible)", S), ("A (alert)", A), ("I (infected)", I)]
    assert [label for label, _ in states] == [label for label, _ in expected_states]
    for (_, xy), (_, values) in zip(states, expected_states, strict=True):
        assert np.array_equal(xy, np.column_stack([TIMES, values]))
    assert np.array_equal(r_lines[0][1], np.column_stack([TIMES, R]))
    assert np.array_equal(e_lines[0][1], np.column_stack([TIMES, E]))
    assert all(
        line.get_drawstyle() == "steps-post"
        for ax in fig.axes
        for line in ax.get_lines()
    )


def test_layout_labels_and_caption(result: SimulationResult) -> None:
    fig = plot_timeseries(result)
    states_ax, r_ax, energy_ax = fig.axes
    assert len(fig.axes) == 3
    shared = states_ax.get_shared_x_axes()
    assert shared.joined(states_ax, r_ax) and shared.joined(r_ax, energy_ax)
    assert states_ax.get_ylabel() == "number of institutions"
    assert r_ax.get_ylabel() == "r(t) = positive edges / total edges"
    assert energy_ax.get_ylabel() == "E(G_t)"
    assert energy_ax.get_xlabel() == "time (model units)"
    for ax in fig.axes:
        assert ax.get_legend() is not None and ax.get_title()
    assert [t.get_text() for t in r_ax.get_legend().get_texts()] == ["r(t)"]
    assert SYNTHETIC_CAPTION in [t.get_text() for t in fig.texts]
    title = fig.get_suptitle()
    assert "seed=3" in title and "α=0.5" in title and "Erdős–Rényi p=0.3" in title


def test_no_pyplot_state(
    result: SimulationResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("plt.show must not be called")

    monkeypatch.setattr(plt, "show", forbidden)
    before = plt.get_fignums()
    fig = plot_timeseries(result)
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == before == []


def test_normalize_divides_counts_by_n(result: SimulationResult) -> None:
    fig = plot_timeseries(result, normalize=True)
    states = _lines(fig)[0]
    for (_, xy), counts in zip(states, (S, A, I), strict=True):
        assert np.array_equal(xy[:, 1], np.asarray(counts) / 10)
    assert fig.axes[0].get_ylabel() == "fraction of institutions"


def test_graph_without_edges_shows_r_as_a_gap(make_result: Factory) -> None:
    nan = float("nan")
    fig = plot_timeseries(make_result(TIMES, S, A, I, [nan] * 4, E, m=0))
    r_xy = _lines(fig)[1][0][1]
    assert np.all(np.isnan(r_xy[:, 1])) and not np.any(r_xy[:, 1] == 0)
    assert any("undefined (NaN)" in t.get_text() for t in fig.axes[1].texts)


@pytest.mark.parametrize(
    ("times", "s", "a", "i", "r", "e", "reason"),
    [
        ([0.0], [9], [0], [1], [0.6], [-0.01], "max_events"),
        ([0.0, 4.0], [10, 10], [0, 0], [0, 0], [0.5, 0.5], [0.0, 0.0], "absorbed"),
        (
            [0.0, 2.0, 4.0],
            [8, 8, 8],
            [1, 1, 1],
            [1, 1, 1],
            [0.7] * 3,
            [-0.2] * 3,
            "t_max",
        ),
    ],
    ids=["single-point", "absorbed", "constant"],
)
def test_edge_case_series(
    make_result: Factory,
    times: list[float],
    s: list[int],
    a: list[int],
    i: list[int],
    r: list[float],
    e: list[float],
    reason: str,
) -> None:
    fig = plot_timeseries(make_result(times, s, a, i, r, e, reason=reason))
    assert np.array_equal(_lines(fig)[2][0][1], np.column_stack([times, e]))


def test_title_without_config(make_result: Factory) -> None:
    fig = plot_timeseries(make_result(TIMES, S, A, I, R, E, config=None))
    assert fig.get_suptitle().startswith("Simulation on a supplied graph: n=10, m=4")


def test_save_png(result: SimulationResult, tmp_path: Path) -> None:
    path = tmp_path / "timeseries.png"
    save_figure(plot_timeseries(result), path)
    data = path.read_bytes()
    assert len(data) > 1000 and data.startswith(PNG_SIGNATURE)


def test_rejects_bad_input(result: SimulationResult, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="SimulationResult"):
        plot_timeseries({"times": []})
    with pytest.raises(TypeError, match="normalize"):
        plot_timeseries(result, normalize=1)
    with pytest.raises(ValueError, match=r"\.png"):
        save_figure(plot_timeseries(result), f"{tmp_path}/out.pdf")
    with pytest.raises(TypeError, match="Figure"):
        save_figure("figure", f"{tmp_path}/out.png")


# --- 3D surfaces ---------------------------------------------------------------

# Asymmetric grid (3 alpha x 2 r0 x 2 rho_0) with distinct values per cell, so
# a transposed or mis-indexed z cannot match.
SURFACE_GRID = SweepGrid(alpha=(0.0, 0.5, 1.0), r0=(0.2, 0.8), rho0=(0.05, 0.2))


def _sweep_result() -> SweepResult:
    from tests.conftest import FIXTURE_CONFIG

    shape = SURFACE_GRID.shape
    rho_mean = np.arange(12, dtype=float).reshape(shape) / 100
    rho_mean[1, 0, 2] = np.nan
    energy_mean = -np.arange(12, dtype=float).reshape(shape) / 1000

    def surface(mean: np.ndarray) -> MetricSurface:
        return MetricSurface(
            mean, mean / 10, mean / 20, np.arange(12, dtype=np.int64).reshape(shape) % 4
        )

    return SweepResult(
        grid=SURFACE_GRID,
        repetitions=3,
        settings=SteadyStateSettings(),
        base_config=FIXTURE_CONFIG,
        seed=3,
        config_hash="0" * 64,
        runs=(),
        rho_inf=surface(rho_mean),
        energy=surface(energy_mean),
    )


@pytest.mark.parametrize("metric", ["rho_inf", "energy"])
def test_surface_data_matches_sweep_arrays(metric: str) -> None:
    sweep = _sweep_result()
    expected = sweep.rho_inf if metric == "rho_inf" else sweep.energy
    fig = plot_surface(sweep, metric)
    assert len(fig.data) == len(SURFACE_GRID.rho0)
    for p, trace in enumerate(fig.data):
        z = np.asarray(trace.z, dtype=float)
        assert z.shape == (2, 3)  # z[r0_index][alpha_index]
        assert np.array_equal(z, expected.mean[p], equal_nan=True)
        assert list(trace.x) == list(SURFACE_GRID.alpha)
        assert list(trace.y) == list(SURFACE_GRID.r0)
    assert np.isnan(np.asarray(fig.data[1].z, dtype=float)[0, 2]) == (
        metric == "rho_inf"
    )


def test_surface_orientation_cell_by_cell() -> None:
    sweep = _sweep_result()
    trace = plot_surface(sweep, "rho_inf").data[0]
    z = np.asarray(trace.z, dtype=float)
    for r in range(2):
        for a in range(3):
            assert z[r][a] == sweep.rho_inf.mean[0, r, a]
            assert (
                f"alpha={SURFACE_GRID.alpha[a]:g}<br>r0={SURFACE_GRID.r0[r]:g}"
                in trace.text[r][a]
            )


def test_dropdown_and_visibility() -> None:
    fig = plot_surface(_sweep_result(), "energy")
    assert [t.visible for t in fig.data] == [True, False]
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["rho_0 = 0.05", "rho_0 = 0.2"]
    assert [b.args[0]["visible"] for b in buttons] == [[True, False], [False, True]]


def test_hover_text_reports_variability() -> None:
    sweep = _sweep_result()
    text = plot_surface(sweep, "rho_inf").data[0].text[1][2]
    mean = sweep.rho_inf.mean[0, 1, 2]
    assert f"std={mean / 10:.6g}" in text and f"SE={mean / 20:.6g}" in text
    assert f"k={sweep.rho_inf.k_converged[0, 1, 2]}/3 converged" in text
    assert "mean=NaN" in plot_surface(sweep, "rho_inf").data[1].text[0][2]


def test_surface_labels_and_caption() -> None:
    fig = plot_surface(_sweep_result(), "rho_inf")
    assert "rho_inf" in fig.layout.title.text and "synthetic" in fig.layout.title.text
    assert fig.layout.annotations[0].text == SYNTHETIC_CAPTION
    assert fig.layout.scene.xaxis.title.text.startswith("alpha")
    assert fig.layout.scene.yaxis.title.text.startswith("r0")


def test_surface_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="metric"):
        plot_surface(_sweep_result(), "clustering")
    with pytest.raises(TypeError, match="SweepResult"):
        plot_surface({"grid": None}, "rho_inf")


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []
        self.inline = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            src = dict(attrs).get("src")
            if src is None:
                self.inline += 1
            else:
                self.sources.append(src)


def test_html_is_self_contained(tmp_path: Path) -> None:
    path = tmp_path / "surface_rho_infinity.html"
    write_surface_html(plot_surface(_sweep_result(), "rho_inf"), path)
    html = path.read_text(encoding="utf-8")
    parser = _Scripts()
    parser.feed(html)
    assert parser.inline >= 2 and parser.sources == []
    assert '<script src="http' not in html and 'src="//' not in html
    assert "rho_0 = 0.05" in html and SYNTHETIC_CAPTION in html
    assert path.stat().st_size > 1_000_000  # plotly.js is embedded


def test_write_surface_html_rejects_bad_input(tmp_path: Path) -> None:
    fig = plot_surface(_sweep_result(), "rho_inf")
    with pytest.raises(ValueError, match=r"\.html"):
        write_surface_html(fig, tmp_path / "x.png")
    with pytest.raises(TypeError, match="plotly Figure"):
        write_surface_html(plot_timeseries, tmp_path / "x.html")
