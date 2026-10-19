"""Tests for the command-line interface (``cli`` and the root ``main.py``)."""

import dataclasses
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import financial_contagion_sim.cli as cli
import financial_contagion_sim.sweep as sweep_module
from financial_contagion_sim.artifacts import config_from_dict, read_metrics_csv
from financial_contagion_sim.simulator import SimulationConfig, run_simulation
from financial_contagion_sim.sweep import (
    SteadyStateSettings,
    SweepGrid,
    SweepResult,
    load_sweep,
    read_records,
    run_sweep,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL = [
    "--model",
    "erdos_renyi",
    "--n",
    "12",
    "--er-p",
    "0.3",
    "--beta",
    "2.0",
    "--beta-a",
    "0.5",
    "--kappa",
    "3.0",
    "--delta",
    "0.7",
    "--gamma",
    "1.0",
    "--seed",
    "7",
]
RUN = [
    "run",
    *MODEL,
    "--t-max",
    "1.0",
    "--alpha",
    "0.5",
    "--r0",
    "0.6",
    "--rho0",
    "0.25",
    "--num-points",
    "21",
]
SWEEP = [
    "sweep",
    *MODEL,
    "--t-max",
    "2.0",
    "--alpha-range",
    "0",
    "1",
    "2",
    "--r0-range",
    "0.25",
    "0.75",
    "2",
    "--rho0-values",
    "0.25",
    "--reps",
    "2",
    "--num-points",
    "51",
]


def _main(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Exit code and output; argparse's SystemExit (help/usage) becomes a code."""
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
    out, err = capsys.readouterr()
    return code, out, err


def _only_dir(parent: Path) -> Path:
    children = list(parent.iterdir())
    assert len(children) == 1, children
    return children[0]


# --- help, version, defaults -------------------------------------------------


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _main(["--help"], capsys)
    assert code == 0 and "run" in out and "sweep" in out and "exit codes" in out
    code, out, _ = _main(["--version"], capsys)
    assert code == 0
    assert out.strip() == "financial-contagion-sim " + importlib.metadata.version(
        "financial-contagion-sim"
    )


RUN_OPTIONS = [
    "--config",
    "--model",
    "--n",
    "--ba-m",
    "--er-p",
    "--alpha",
    "--r0",
    "--rho0",
    "--beta",
    "--beta-a",
    "--kappa",
    "--delta",
    "--gamma",
    "--t-max",
    "--seed",
    "--max-events",
    "--num-points",
    "--stop-on-sais-absorbed",
    "--output-dir",
    "--run-name",
    "--dry-run",
    "--no-plot",
]
SWEEP_OPTIONS = [
    "--alpha-range",
    "--r0-range",
    "--rho0-values",
    "--reps",
    "--tol-rho",
    "--tol-e",
    "--max-runs",
    "--resume",
    "--no-html",
    "--output-dir",
    "--dry-run",
    "--seed",
]


@pytest.mark.parametrize(
    ("command", "options"), [("run", RUN_OPTIONS), ("sweep", SWEEP_OPTIONS)]
)
def test_subcommand_help_lists_every_option(
    command: str, options: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _main([command, "--help"], capsys)
    assert code == 0
    for option in options:
        assert re.search(rf"(^|\s|,){re.escape(option)}(\s|,|=|$)", out, re.M), option
    assert "exit codes" in out


def test_control_defaults_come_from_simulation_config() -> None:
    fields = {f.name: f.default for f in dataclasses.fields(SimulationConfig)}
    assert cli._config_defaults() == {
        name: fields[name]
        for name in ("max_events", "num_points", "stop_on_sais_absorbed")
    }
    args = cli.build_parser().parse_args(["run"])
    for dest in (
        "n",
        "model",
        "alpha",
        "r0",
        "rho0",
        "beta",
        "beta_a",
        "kappa",
        "delta",
        "gamma",
        "t_max",
        "seed",
        "max_events",
        "num_points",
        "stop_on_sais_absorbed",
    ):
        assert getattr(args, dest) is None, dest
    help_text = (
        cli.build_parser()._subparsers._group_actions[0].choices["run"].format_help()
    )
    assert f"(default {fields['num_points']})" in help_text


def test_root_main_py_runs_as_a_script() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "sweep" in proc.stdout


# --- invalid input -----------------------------------------------------------


def _replace(argv: list[str], flag: str, value: str | None) -> list[str]:
    out = list(argv)
    index = out.index(flag)
    if value is None:
        del out[index : index + 2]
    else:
        out[index + 1] = value
    return out


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (_replace(RUN, "--r0", "2.0"), "--r0"),
        (_replace(RUN, "--alpha", "1.5"), "--alpha"),
        (_replace(RUN, "--beta-a", "2.0"), "--beta-a"),
        (_replace(RUN, "--n", "0"), "--n"),
        (_replace(RUN, "--t-max", "-1"), "--t-max"),
        (_replace(RUN, "--model", "watts_strogatz"), "--model"),
        (_replace(RUN, "--n", "twelve"), "--n"),
        ([*RUN, "--ba-m", "2"], "--model"),
        (_replace(RUN, "--seed", None), "--seed"),
        (_replace(RUN, "--beta", None), "--beta"),
        (_replace(SWEEP, "--reps", "0"), "--reps"),
        ([*SWEEP, "--alpha", "0.5"], "--alpha"),
        (_replace(SWEEP, "--r0-range", "0.75"), "--r0-range"),
        ([*RUN, "--run-name", "a/b"], "--run-name"),
    ],
    ids=[
        "r0",
        "alpha",
        "beta_a=beta",
        "n=0",
        "t_max<0",
        "model",
        "n-type",
        "ba_m-with-er",
        "missing-seed",
        "missing-beta",
        "reps=0",
        "swept-fixed",
        "range-args",
        "run-name",
    ],
)
def test_invalid_input_exits_2_naming_the_option(
    argv: list[str], flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _main([*argv, "--output-dir", str(tmp_path / "out")], capsys)
    assert code == 2
    assert flag in err and "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_invalid_config_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _, err = _main(["run", "--config", str(bad)], capsys)
    assert code == 2 and "--config" in err and "Traceback" not in err
    code, _, err = _main(["run", "--config", str(tmp_path / "missing.json")], capsys)
    assert code == 2 and "--config" in err


# --- run ---------------------------------------------------------------------


def test_run_end_to_end_matches_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _main([*RUN, "--output-dir", str(tmp_path)], capsys)
    assert code == 0, err
    run_dir = _only_dir(tmp_path)
    assert re.fullmatch(r"run_[0-9a-f]{8}_seed7", run_dir.name)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.json",
        "metadata.json",
        "metrics.csv",
        "timeseries.png",
    ]
    assert str(run_dir) in out and "termination:" in out and "synthetic" in out
    config = config_from_dict(
        json.loads((run_dir / "config.json").read_text())["config"]
    )
    expected = run_simulation(config)
    metrics = read_metrics_csv(run_dir / "metrics.csv")
    assert np.array_equal(metrics["time"], expected.times)
    assert np.array_equal(metrics["I"], expected.I)
    assert np.array_equal(metrics["r"], expected.r, equal_nan=True)
    assert np.array_equal(metrics["E"], expected.E)
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["argv"][0] == "run" and meta["synthetic"] is True


def test_config_file_round_trip_and_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main([*RUN, "--output-dir", str(tmp_path / "a")], capsys)[0] == 0
    first = _only_dir(tmp_path / "a")
    config_path = first / "config.json"
    assert (
        _main(
            ["run", "--config", str(config_path), "--output-dir", str(tmp_path / "b")],
            capsys,
        )[0]
        == 0
    )
    second = _only_dir(tmp_path / "b")
    assert second.name == first.name
    assert (second / "metrics.csv").read_bytes() == (first / "metrics.csv").read_bytes()
    assert (
        _main(
            [
                "run",
                "--config",
                str(config_path),
                "--seed",
                "8",
                "--output-dir",
                str(tmp_path / "c"),
            ],
            capsys,
        )[0]
        == 0
    )
    third = _only_dir(tmp_path / "c")
    assert json.loads((third / "config.json").read_text())["config"]["seed"] == 8
    assert (third / "metrics.csv").read_bytes() != (first / "metrics.csv").read_bytes()


def test_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "never"
    code, out, _ = _main([*RUN, "--output-dir", str(out_dir), "--dry-run"], capsys)
    assert code == 0 and "metrics.csv" in out and "timeseries.png" in out
    assert not out_dir.exists()
    code, out, _ = _main([*SWEEP, "--output-dir", str(out_dir), "--dry-run"], capsys)
    assert code == 0 and "runs: 8" in out and "estimated runtime" in out
    assert not out_dir.exists()


def test_existing_run_directory_is_refused_before_simulating(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _main([*RUN, "--output-dir", str(tmp_path)], capsys)[0] == 0
    run_dir = _only_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}

    def must_not_run(config: object) -> None:
        raise AssertionError("simulation must not run when the target exists")

    monkeypatch.setattr(cli, "run_simulation", must_not_run)
    code, _, err = _main([*RUN, "--output-dir", str(tmp_path)], capsys)
    assert code == 3 and "refusing to overwrite" in err
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_output_dir_that_is_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code, _, err = _main([*RUN, "--output-dir", str(blocker)], capsys)
    assert code == 2 and "--output-dir" in err and "Traceback" not in err


def test_no_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main([*RUN, "--output-dir", str(tmp_path), "--no-plot"], capsys)[0] == 0
    assert not (_only_dir(tmp_path) / "timeseries.png").exists()


# --- sweep -------------------------------------------------------------------


def _library_sweep() -> SweepResult:
    base = config_from_dict(
        {
            "n": 12,
            "topology": "erdos_renyi",
            "er_p": 0.3,
            "alpha": 0.0,
            "r0": 0.25,
            "initial_infected_fraction": 0.25,
            "edge_update_rate": 1.0,
            "t_max": 2.0,
            "seed": 7,
            "num_points": 51,
            "sais_params": {"beta": 2.0, "beta_a": 0.5, "kappa": 3.0, "delta": 0.7},
        }
    )
    grid = SweepGrid(alpha=(0.0, 1.0), r0=(0.25, 0.75), rho0=(0.25,))
    return run_sweep(base, grid, 2, SteadyStateSettings())


def test_sweep_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _main([*SWEEP, "--output-dir", str(tmp_path)], capsys)
    assert code == 0, err
    run_dir = _only_dir(tmp_path)
    assert re.fullmatch(r"sweep_[0-9a-f]{8}_seed7", run_dir.name)
    files = sorted(
        p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()
    )
    assert files == [
        "config.json",
        "metadata.json",
        "sweep/surface_energy.html",
        "sweep/surface_rho_infinity.html",
        "sweep/sweep_results.csv",
        "sweep/sweep_summary.json",
    ]
    definition = json.loads((run_dir / "config.json").read_text())
    loaded = load_sweep(definition, run_dir / "sweep" / "sweep_results.csv")
    direct = _library_sweep()
    assert loaded.config_hash == direct.config_hash
    assert np.array_equal(loaded.rho_inf.mean, direct.rho_inf.mean, equal_nan=True)
    summary = json.loads((run_dir / "sweep" / "sweep_summary.json").read_text())
    assert summary["n_runs"] == 8 and summary["config_hash"] == loaded.config_hash
    assert "sweep directory" in out


def test_sweep_no_html(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main([*SWEEP, "--output-dir", str(tmp_path), "--no-html"], capsys)[0] == 0
    assert not list(_only_dir(tmp_path).rglob("*.html"))


def test_sweep_budget_refusal_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _main(
        [*SWEEP, "--output-dir", str(tmp_path / "out"), "--max-runs", "3"], capsys
    )
    assert code == 2 and "--max-runs" in err and "8 runs" in err
    assert not (tmp_path / "out").exists()


def _interrupt_after(monkeypatch: pytest.MonkeyPatch, calls: int) -> None:
    real = sweep_module.run_simulation
    count = {"n": 0}

    def flaky(config: SimulationConfig) -> object:
        count["n"] += 1
        if count["n"] > calls:
            raise KeyboardInterrupt
        return real(config)

    monkeypatch.setattr(sweep_module, "run_simulation", flaky)


def test_resume_after_interruption_matches_uninterrupted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _main([*SWEEP, "--output-dir", str(tmp_path / "full")], capsys)[0] == 0
    full = _only_dir(tmp_path / "full")

    _interrupt_after(monkeypatch, 5)
    with pytest.raises(KeyboardInterrupt):
        cli.main([*SWEEP, "--output-dir", str(tmp_path / "part")])
    monkeypatch.undo()
    incomplete = _only_dir(tmp_path / "part")
    assert incomplete.name.endswith(".incomplete")
    results = incomplete / "sweep" / "sweep_results.csv"
    assert len(read_records(results)) == 5
    text = results.read_text(encoding="utf-8")
    results.write_text(text[: len(text) - 7], encoding="utf-8")  # truncate the last row
    assert len(read_records(results)) == 4

    code, _, err = _main([*SWEEP, "--output-dir", str(tmp_path / "part")], capsys)
    assert code == 3 and "--resume" in err
    code, _, err = _main(
        [*SWEEP, "--output-dir", str(tmp_path / "part"), "--resume"], capsys
    )
    assert code == 0, err
    resumed = _only_dir(tmp_path / "part")
    assert resumed.name == full.name
    a = read_records(full / "sweep" / "sweep_results.csv")
    b = read_records(resumed / "sweep" / "sweep_results.csv")
    assert [_nan_safe(r) for r in a] == [_nan_safe(r) for r in b]
    assert len(a) == 8


def _nan_safe(record: object) -> tuple[object, ...]:
    """Record fields with NaN replaced by a marker, so equal records compare equal."""
    return tuple(
        "nan" if isinstance(v, float) and np.isnan(v) else v
        for v in dataclasses.astuple(record)
    )


def test_resume_refuses_a_different_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _interrupt_after(monkeypatch, 2)
    with pytest.raises(KeyboardInterrupt):
        cli.main([*SWEEP, "--output-dir", str(tmp_path)])
    monkeypatch.undo()
    incomplete = _only_dir(tmp_path)
    saved = json.loads((incomplete / "config.json").read_text())
    saved["repetitions"] = 9
    (incomplete / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    code, _, err = _main([*SWEEP, "--output-dir", str(tmp_path), "--resume"], capsys)
    assert code == 3 and "hash differs" in err


def test_unexpected_errors_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(config: object) -> None:
        raise RuntimeError("component failure")

    monkeypatch.setattr(cli, "run_simulation", boom)
    with pytest.raises(RuntimeError, match="component failure"):
        cli.main([*RUN, "--output-dir", str(tmp_path)])
    assert not tmp_path.joinpath("x").exists() and list(tmp_path.iterdir()) == []
