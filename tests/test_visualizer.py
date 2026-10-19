"""Tests for the visualization engine (``visualizer``); spec §5, step 3."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from financial_contagion_sim.simulator import SimulationResult
from financial_contagion_sim.visualizer import (
    SYNTHETIC_CAPTION,
    plot_timeseries,
    save_figure,
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
