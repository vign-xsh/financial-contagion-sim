"""Tests for parameter sweeps and steady-state estimation (``sweep``)."""

import dataclasses
import math
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import financial_contagion_sim.sweep as sweep_module
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from financial_contagion_sim.sweep import (
    RunRecord,
    SteadyStateSettings,
    SweepGrid,
    SweepResult,
    aggregate,
    append_record,
    count_runs,
    estimate_steady_state,
    load_sweep,
    make_grid,
    read_records,
    rewrite_records,
    run_sweep,
    save_sweep_summary,
    sweep_config_for,
    sweep_definition,
    write_results_header,
)

Factory = Callable[..., SimulationResult]

BASE = SimulationConfig(
    n=12,
    topology="erdos_renyi",
    er_p=0.3,
    sais_params=SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7),
    alpha=0.0,
    r0=0.0,
    initial_infected_fraction=0.25,
    edge_update_rate=1.0,
    t_max=2.0,
    seed=11,
    num_points=51,
)
GRID = SweepGrid(alpha=(0.0, 1.0), r0=(0.25, 0.75), rho0=(0.25,))
TIMES = np.linspace(0.0, 4.0, 41)  # FIXTURE_CONFIG has t_max = 4


def _key(record: RunRecord) -> tuple[object, ...]:
    return tuple(
        "nan" if isinstance(v, float) and math.isnan(v) else v
        for v in dataclasses.astuple(record)
    )


def _assert_same_sweep(a: SweepResult, b: SweepResult) -> None:
    assert (a.grid, a.repetitions, a.settings, a.seed, a.config_hash) == (
        b.grid,
        b.repetitions,
        b.settings,
        b.seed,
        b.config_hash,
    )
    assert [_key(r) for r in a.runs] == [_key(r) for r in b.runs]
    for metric in ("rho_inf", "energy"):
        for field in ("mean", "std", "se", "k_converged"):
            x, y = (
                getattr(getattr(a, metric), field),
                getattr(getattr(b, metric), field),
            )
            assert np.array_equal(x, y, equal_nan=True), (metric, field)


def _series(
    make_result: Factory,
    infected: np.ndarray,
    energy: np.ndarray,
    reason: str = "t_max",
    n: int = 1000,
) -> SimulationResult:
    infected = np.asarray(infected, dtype=np.int64)
    zeros = np.zeros_like(infected)
    return make_result(
        TIMES,
        n - infected,
        zeros,
        infected,
        np.full(len(TIMES), 0.5),
        energy,
        n=n,
        reason=reason,
    )


# --- steady state ------------------------------------------------------------


def test_constant_series_is_converged_exactly(make_result: Factory) -> None:
    state = estimate_steady_state(
        _series(make_result, np.full(41, 300), np.full(41, -0.01))
    )
    assert (state.rho_inf, state.e_ss) == (0.3, -0.01)
    assert state.rho_converged and state.e_converged


def test_exponential_decay_on_both_sides_of_the_tolerance(make_result: Factory) -> None:
    limit = -0.01
    energy = limit + 0.02 * np.exp(-TIMES)
    window, first, second = TIMES >= 3.2, (TIMES >= 3.2) & (TIMES < 3.6), TIMES >= 3.6
    gap = abs(energy[first].mean() - energy[second].mean())
    ratio = gap / abs(energy[window].mean())
    result = _series(make_result, np.full(41, 200), energy)
    above = estimate_steady_state(result, SteadyStateSettings(tol_e_rel=ratio * 1.01))
    below = estimate_steady_state(result, SteadyStateSettings(tol_e_rel=ratio * 0.99))
    assert above.e_converged and above.e_ss == pytest.approx(energy[window].mean())
    assert abs(above.e_ss - limit) < 0.001
    assert not below.e_converged and math.isnan(below.e_ss)


def test_drifting_ramp_is_not_converged(make_result: Factory) -> None:
    ramp = np.linspace(100, 500, 41).astype(np.int64)
    state = estimate_steady_state(_series(make_result, ramp, np.full(41, -0.01)))
    assert not state.rho_converged and math.isnan(state.rho_inf)
    assert state.e_converged  # independent of rho


def test_metrics_converge_independently(make_result: Factory) -> None:
    energy_ramp = np.linspace(-0.01, -0.03, 41)
    state = estimate_steady_state(_series(make_result, np.full(41, 100), energy_ramp))
    assert state.rho_converged and state.rho_inf == 0.1
    assert not state.e_converged and math.isnan(state.e_ss)


def test_absorbed_run_is_converged(make_result: Factory) -> None:
    state = estimate_steady_state(
        _series(make_result, np.zeros(41), np.full(41, -0.002), reason="absorbed")
    )
    assert (state.rho_inf, state.e_ss, state.rho_converged, state.e_converged) == (
        0.0,
        -0.002,
        True,
        True,
    )


@pytest.mark.parametrize("reason", ["max_events", "sais_absorbed"])
def test_truncated_run_is_not_converged(make_result: Factory, reason: str) -> None:
    state = estimate_steady_state(
        _series(make_result, np.full(41, 100), np.full(41, -0.01), reason=reason)
    )
    assert not state.rho_converged and not state.e_converged
    assert math.isnan(state.rho_inf) and math.isnan(state.e_ss)


def test_window_needs_points_in_both_halves(make_result: Factory) -> None:
    result = make_result([0.0, 4.0], [9, 9], [0, 0], [1, 1], [0.5, 0.5], [-0.01, -0.01])
    state = estimate_steady_state(result)
    assert not state.rho_converged and math.isnan(state.rho_inf)


def test_estimator_requires_grid_recording(make_result: Factory) -> None:
    events_config = dataclasses.replace(BASE, record="events")
    with pytest.raises(ValueError, match="record='grid'"):
        estimate_steady_state(
            make_result([0.0], [9], [0], [1], [0.5], [0.0], config=events_config)
        )
    with pytest.raises(ValueError, match="record='grid'"):
        estimate_steady_state(
            make_result([0.0], [9], [0], [1], [0.5], [0.0], config=None)
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window_start": 0.0}, ValueError),
        ({"window_start": 1.0}, ValueError),
        ({"tol_rho": -0.1}, ValueError),
        ({"tol_e_rel": math.nan}, ValueError),
        ({"tol_rho": "0.1"}, TypeError),
        ({"min_converged_fraction": 0.0}, ValueError),
        ({"min_converged_fraction": 1.5}, ValueError),
    ],
)
def test_settings_validation(kwargs: dict[str, object], error: type) -> None:
    with pytest.raises(error):
        SteadyStateSettings(**kwargs)


# --- grid --------------------------------------------------------------------


def test_make_grid_sizes_endpoints_and_values() -> None:
    grid = make_grid((0.0, 1.0, 11), (0.0, 1.0, 11), (0.05, 0.1, 0.2))
    assert grid.shape == (3, 11, 11) and grid.n_points == 363
    assert grid.alpha[0] == 0.0 and grid.alpha[-1] == 1.0
    assert grid.alpha == tuple(float(v) for v in np.linspace(0, 1, 11))
    assert count_runs(grid, 5) == 1815
    assert make_grid((0.3, 0.3, 1), (0.0, 1.0, 2), (0.1,)).alpha == (0.3,)


def test_grid_index_order_and_round_trip() -> None:
    grid = SweepGrid(alpha=(0.0, 0.5, 1.0), r0=(0.1, 0.9), rho0=(0.05, 0.2))
    assert grid.point(1).alpha_index == 1 and grid.point(1).r0_index == 0
    assert grid.point(3).r0_index == 1 and grid.point(3).alpha_index == 0
    assert grid.point(6).rho0_index == 1
    seen = set()
    for index in range(grid.n_points):
        p = grid.point(index)
        assert grid.index_of(p.alpha_index, p.r0_index, p.rho0_index) == index
        assert (p.alpha, p.r0, p.rho0) == (
            grid.alpha[p.alpha_index],
            grid.r0[p.r0_index],
            grid.rho0[p.rho0_index],
        )
        seen.add((p.alpha, p.r0, p.rho0))
    assert len(seen) == 12
    with pytest.raises(ValueError):
        grid.point(12)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"alpha": (0.5, 0.5)}, ValueError),
        ({"r0": (0.9, 0.1)}, ValueError),
        ({"rho0": (0.1, 0.1)}, ValueError),
        ({"alpha": ()}, ValueError),
        ({"alpha": (1.5,)}, ValueError),
        ({"alpha": [0.0]}, TypeError),
    ],
)
def test_grid_validation(kwargs: dict[str, object], error: type) -> None:
    values: dict[str, object] = {"alpha": (0.0, 1.0), "r0": (0.0, 1.0), "rho0": (0.1,)}
    values.update(kwargs)
    with pytest.raises(error):
        SweepGrid(**values)


def test_make_grid_validation() -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        make_grid((0.0, 1.0, 0), (0.0, 1.0, 2), (0.1,))
    with pytest.raises(TypeError, match="start, stop, count"):
        make_grid((0.0, 1.0), (0.0, 1.0, 2), (0.1,))


# --- running sweeps ----------------------------------------------------------


def test_every_run_uses_grid_recording_and_its_own_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SimulationConfig] = []

    def spy(config: SimulationConfig) -> SimulationResult:
        seen.append(config)
        return run_simulation(config)

    monkeypatch.setattr(sweep_module, "run_simulation", spy)
    result = run_sweep(BASE, GRID, 2)
    assert len(seen) == count_runs(GRID, 2) == len(result.runs) == 8
    for config, record in zip(seen, result.runs, strict=True):
        assert config.record == "grid" and config.stop_on_sais_absorbed is False
        assert isinstance(config.seed, np.random.SeedSequence)
        assert config.seed.entropy == 11 and config.seed.spawn_key == record.key
        assert (config.alpha, config.r0) == (record.alpha, record.r0)
        assert config.initial_infected_fraction == record.rho0
        assert (config.n, config.er_p, config.t_max) == (12, 0.3, 2.0)


def test_results_do_not_depend_on_execution_order() -> None:
    result = run_sweep(BASE, GRID, 2)
    keys = [(p, r) for p in range(GRID.n_points) for r in range(2)]
    random.Random(4).shuffle(keys)
    manual = {}
    for p, r in reversed(keys):
        state = estimate_steady_state(
            run_simulation(sweep_config_for(BASE, GRID, p, r))
        )
        manual[(p, r)] = (state.rho_inf, state.e_ss)
    for record in result.runs:
        expected = manual[record.key]
        assert np.array_equal((record.rho_inf, record.e_ss), expected, equal_nan=True)


def test_resume_reproduces_the_uninterrupted_sweep() -> None:
    full = run_sweep(BASE, GRID, 2)
    partial = {r.key: r for r in full.runs[:3]}
    new_keys: list[tuple[int, int]] = []
    resumed = run_sweep(
        BASE, GRID, 2, completed=partial, on_run=lambda r: new_keys.append(r.key)
    )
    assert new_keys == [r.key for r in full.runs[3:]]
    _assert_same_sweep(full, resumed)


def test_same_seed_reproduces_and_different_seed_differs() -> None:
    _assert_same_sweep(run_sweep(BASE, GRID, 2), run_sweep(BASE, GRID, 2))
    other = run_sweep(dataclasses.replace(BASE, seed=12), GRID, 2)
    assert [r.n_events for r in other.runs] != [
        r.n_events for r in run_sweep(BASE, GRID, 2).runs
    ]


def test_component_exception_aborts_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def boom(config: SimulationConfig) -> SimulationResult:
        raise RuntimeError("bug in a component")

    monkeypatch.setattr(sweep_module, "run_simulation", boom)
    with pytest.raises(RuntimeError, match="bug in a component"):
        run_sweep(BASE, GRID, 2, on_run=lambda r: calls.append(r.key))
    assert calls == []


@pytest.mark.parametrize(
    ("base", "error", "match"),
    [
        (dataclasses.replace(BASE, record="events"), ValueError, "record='grid'"),
        (dataclasses.replace(BASE, stop_on_sais_absorbed=True), ValueError, "stop_on"),
        (
            dataclasses.replace(BASE, seed=np.random.SeedSequence(1)),
            TypeError,
            "base seed",
        ),
    ],
    ids=["events", "stop", "seedseq"],
)
def test_sweep_rejects_unsuitable_base(
    base: SimulationConfig, error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        run_sweep(base, GRID, 2)


def test_rejects_foreign_completed_records() -> None:
    record = run_sweep(BASE, GRID, 1).runs[0]
    with pytest.raises(ValueError, match="does not belong"):
        run_sweep(
            BASE,
            GRID,
            1,
            completed={(99, 0): dataclasses.replace(record, point_index=99)},
        )


# --- aggregation -------------------------------------------------------------


def _record(point: int, rep: int, rho: float, ok: bool) -> RunRecord:
    nan = math.nan
    return RunRecord(
        point,
        rep,
        point,
        0,
        0,
        float(point),
        0.0,
        0.1,
        rho if ok else nan,
        rho if ok else nan,
        ok,
        ok,
        "t_max",
        10,
    )


def test_aggregation_counts_min_rule_std_and_se() -> None:
    grid = SweepGrid(alpha=(0.0, 1.0), r0=(0.0,), rho0=(0.1,))
    records = [
        _record(0, 0, 0.1, True),
        _record(0, 1, 0.3, True),
        _record(0, 2, 0.0, False),
        _record(1, 0, 0.5, True),
        _record(1, 1, 0.0, False),
        _record(1, 2, 0.0, False),
    ]
    rho, energy = aggregate(grid, 3, SteadyStateSettings(), records)
    assert rho.k_converged.ravel().tolist() == [2, 1]
    assert rho.mean[0, 0, 0] == pytest.approx(0.2)
    assert rho.std[0, 0, 0] == pytest.approx(math.sqrt(0.02))
    assert rho.se[0, 0, 0] == pytest.approx(math.sqrt(0.02) / math.sqrt(2))
    assert math.isnan(rho.mean[0, 0, 1]) and math.isnan(rho.std[0, 0, 1])
    assert np.array_equal(energy.mean, rho.mean, equal_nan=True)


def test_single_converged_repetition_has_nan_spread() -> None:
    grid = SweepGrid(alpha=(0.0,), r0=(0.0,), rho0=(0.1,))
    rho, _ = aggregate(grid, 1, SteadyStateSettings(), [_record(0, 0, 0.4, True)])
    assert rho.mean[0, 0, 0] == 0.4 and rho.k_converged[0, 0, 0] == 1
    assert math.isnan(rho.std[0, 0, 0]) and math.isnan(rho.se[0, 0, 0])


def test_min_converged_fraction_one_requires_all() -> None:
    grid = SweepGrid(alpha=(0.0,), r0=(0.0,), rho0=(0.1,))
    records = [_record(0, 0, 0.4, True), _record(0, 1, 0.0, False)]
    rho, _ = aggregate(
        grid, 2, SteadyStateSettings(min_converged_fraction=1.0), records
    )
    assert math.isnan(rho.mean[0, 0, 0]) and rho.k_converged[0, 0, 0] == 1


# --- persistence -------------------------------------------------------------


def test_records_round_trip_and_truncation(tmp_path: Path) -> None:
    result = run_sweep(BASE, GRID, 2)
    path = tmp_path / "sweep_results.csv"
    write_results_header(path)
    for record in result.runs:
        append_record(path, record)
    assert [_key(r) for r in read_records(path)] == [_key(r) for r in result.runs]
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text[: text.rstrip().rfind(",") + 2], encoding="utf-8"
    )  # cut mid-row
    survivors = read_records(path)
    assert [_key(r) for r in survivors] == [_key(r) for r in result.runs[:-1]]
    rewrite_records(path, survivors)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert len(read_records(path)) == len(result.runs) - 1


def test_read_records_rejects_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a sweep results file"):
        read_records(path)


def test_load_sweep_reproduces_result(tmp_path: Path) -> None:
    result = run_sweep(BASE, GRID, 2)
    path = tmp_path / "sweep_results.csv"
    write_results_header(path)
    for record in result.runs:
        append_record(path, record)
    definition = sweep_definition(BASE, GRID, 2, SteadyStateSettings())
    _assert_same_sweep(load_sweep(definition, path), result)
    rewrite_records(path, result.runs[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        load_sweep(definition, path)


def test_summary_contents(tmp_path: Path) -> None:
    import json

    result = run_sweep(BASE, GRID, 2)
    save_sweep_summary(tmp_path / "s.json", result, runtime_seconds=1.25)
    summary = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert summary["config_hash"] == result.config_hash and summary["n_runs"] == 8
    assert summary["runtime_seconds"] == 1.25 and summary["synthetic"] is True
    assert set(summary["non_converged_points"]) == {"rho_inf", "energy"}
    assert summary["grid"] == GRID.to_dict()


def test_hash_ignores_swept_base_values_but_not_others() -> None:
    settings = SteadyStateSettings()
    a = run_sweep(BASE, GRID, 1).config_hash
    b = run_sweep(dataclasses.replace(BASE, alpha=0.9, r0=0.3), GRID, 1).config_hash
    assert a == b
    assert sweep_definition(BASE, GRID, 2, settings) != sweep_definition(
        BASE, GRID, 1, settings
    )
