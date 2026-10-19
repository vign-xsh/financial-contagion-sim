"""Tests for run artifacts (``artifacts``); Phase 4 decisions D1-D7."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import financial_contagion_sim.artifacts as artifacts
from financial_contagion_sim.artifacts import (
    METRICS_COLUMNS,
    config_from_dict,
    config_to_dict,
    digest,
    environment_metadata,
    git_revision,
    make_run_id,
    read_metrics_csv,
    run_config_hash,
    write_run_artifacts,
)
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation

REPO_ROOT = Path(__file__).resolve().parent.parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _config(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "n": 12,
        "topology": "erdos_renyi",
        "er_p": 0.3,
        "sais_params": SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7),
        "alpha": 0.5,
        "r0": 0.6,
        "initial_infected_fraction": 0.25,
        "edge_update_rate": 1.0,
        "t_max": 1.0,
        "seed": 7,
        "num_points": 21,
    }
    values.update(overrides)
    return SimulationConfig(**values)


# --- serialization -----------------------------------------------------------


@pytest.mark.parametrize(
    "seed", [7, np.random.SeedSequence(5, spawn_key=(3, 1))], ids=["int", "seedseq"]
)
def test_config_round_trip(seed: object) -> None:
    config = _config(seed=seed, max_events=10, record="events")
    data = config_to_dict(config)
    json.dumps(data)  # JSON-ready
    rebuilt = config_from_dict(json.loads(json.dumps(data)))
    assert config_to_dict(rebuilt) == data


def test_config_from_dict_delegates_validation() -> None:
    data = config_to_dict(_config())
    data["r0"] = 2.0
    with pytest.raises(ValueError, match=r"r0 must be in \[0, 1\]"):
        config_from_dict(data)
    with pytest.raises(TypeError, match="sais_params"):
        config_from_dict({"n": 3})


def test_digest_and_run_id_are_deterministic() -> None:
    a, b = run_config_hash(_config()), run_config_hash(_config())
    assert a == b and re.fullmatch(r"[0-9a-f]{64}", a)
    assert run_config_hash(_config(seed=8)) != a
    assert digest({"x": 1, "y": [1.5]}) == digest({"y": [1.5], "x": 1})
    assert make_run_id("run", a, 7) == f"run_{a[:8]}_seed7"
    assert make_run_id("sweep", a, 7).startswith("sweep_")
    assert make_run_id("run", a, np.random.SeedSequence(9)) == f"run_{a[:8]}_seed9"
    assert make_run_id("run", a, 7, run_name="trial-1") == "trial-1"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "a/b", "a\\b", "x.tmp-1", "x.incomplete"]
)
def test_invalid_run_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid run name"):
        make_run_id("run", "0" * 64, 1, run_name=name)


# --- metadata ----------------------------------------------------------------


def test_metadata_keys() -> None:
    meta = environment_metadata(["run", "--n", "5"])
    for key in (
        "created_utc",
        "argv",
        "git_commit",
        "git_dirty",
        "versions",
        "synthetic",
    ):
        assert key in meta
    assert meta["synthetic"] is True and meta["argv"] == ["run", "--n", "5"]
    assert set(meta["versions"]) == {
        "financial_contagion_sim",
        "python",
        "numpy",
        "networkx",
        "matplotlib",
        "plotly",
    }


def test_git_absent_gives_null(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(artifacts.subprocess, "run", no_git)
    assert git_revision() == (None, None)
    meta = environment_metadata([])
    assert meta["git_commit"] is None and meta["git_dirty"] is None


def test_not_a_repository_gives_null(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(artifacts.subprocess, "run", failing)
    assert git_revision() == (None, None)


# --- run artifacts -----------------------------------------------------------


def test_run_artifacts_written_and_exact(tmp_path: Path) -> None:
    result = run_simulation(_config())
    run_dir = write_run_artifacts(tmp_path, "r1", result, argv=["run"], plot=True)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.json",
        "metadata.json",
        "metrics.csv",
        "timeseries.png",
    ]
    doc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert doc["kind"] == "run" and doc["config"] == config_to_dict(result.config)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["termination_reason"] == result.termination_reason
    assert meta["n_events"] == result.n_events
    assert meta["config_hash"] == run_config_hash(result.config)
    metrics = read_metrics_csv(run_dir / "metrics.csv")
    assert tuple(metrics) == METRICS_COLUMNS
    for name, series in (
        ("time", result.times),
        ("S", result.S),
        ("A", result.A),
        ("I", result.I),
        ("r", result.r),
        ("E", result.E),
        ("E_delta", result.E_delta),
        ("E_p", result.E_p),
        ("infected_fraction", result.infected_fraction),
    ):
        assert np.array_equal(metrics[name], series), name
    assert (run_dir / "timeseries.png").read_bytes().startswith(PNG_SIGNATURE)
    assert list(tmp_path.iterdir()) == [run_dir]


def test_nan_round_trips(tmp_path: Path) -> None:
    result = run_simulation(_config(er_p=0.0))
    assert np.all(np.isnan(result.r))
    run_dir = write_run_artifacts(tmp_path, "r", result, argv=[], plot=False)
    assert "nan" in (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert np.all(np.isnan(read_metrics_csv(run_dir / "metrics.csv")["r"]))
    assert not (run_dir / "timeseries.png").exists()


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    result = run_simulation(_config())
    run_dir = write_run_artifacts(tmp_path, "r1", result, argv=[], plot=False)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_run_artifacts(tmp_path, "r1", result, argv=[], plot=True)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before
    assert [p.name for p in tmp_path.iterdir()] == ["r1"]


def test_output_path_is_a_file(tmp_path: Path) -> None:
    blocker = tmp_path / "outputs"
    blocker.write_text("not a directory", encoding="utf-8")
    result = run_simulation(_config())
    with pytest.raises(NotADirectoryError, match="is a file"):
        write_run_artifacts(blocker / "nested", "r1", result, argv=[], plot=False)
    assert blocker.read_text(encoding="utf-8") == "not a directory"


def test_unwritable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(f"permission denied: {self}")

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(PermissionError):
        write_run_artifacts(
            tmp_path / "out", "r1", run_simulation(_config()), argv=[], plot=False
        )
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_failure_mid_write_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path, result: object) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("disk full")

    monkeypatch.setattr(artifacts, "write_metrics_csv", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        write_run_artifacts(
            tmp_path, "r1", run_simulation(_config()), argv=[], plot=False
        )
    assert list(tmp_path.iterdir()) == []


def test_requires_run_simulation_result(tmp_path: Path) -> None:
    from financial_contagion_sim.graph_builder import generate_erdos_renyi
    from financial_contagion_sim.simulator import simulate

    g = generate_erdos_renyi(5, 0.5, 0.5, rng=0)
    result = simulate(
        g, np.ones(5, dtype=np.int8), _config().sais_params, 0.5, 1.0, 1.0, seed=0
    )
    with pytest.raises(TypeError, match="run_simulation"):
        write_run_artifacts(tmp_path, "r1", result, argv=[], plot=False)


# --- ignore rules ------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs git and a repository checkout",
)
def test_default_output_directory_is_git_ignored() -> None:
    probe = subprocess.run(
        ["git", "check-ignore", "-q", "outputs/run_x/timeseries.png"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert probe.returncode == 0
