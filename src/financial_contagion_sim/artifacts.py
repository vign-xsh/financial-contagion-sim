"""Run artifacts: serialization, metadata and safe directory handling (Phase 4 D).

Responsibility: turn simulation configurations and results into files, using
only the standard library (``json``, ``csv``), and never leave a directory
that looks complete when it is not.

Layout of one run (Phase 4 D1)::

    <output_dir>/<run_id>/
        config.json       resolved configuration, seed and schema version
        metadata.json     package and dependency versions, git commit and dirty
                          flag (null when git or the repository is unavailable),
                          UTC timestamp, argv, "synthetic": true, termination
                          reason and event counts
        metrics.csv       time,S,A,I,r,E,E_delta,E_p,infected_fraction
        timeseries.png    unless plotting is switched off

Run identifiers (D2) are ``run_<8 hex of the config hash>_seed<seed>`` (or
``sweep_...``) unless a name is supplied, so the same configuration and seed
always map to the same directory; the timestamp lives only in metadata.json.

Safety (D4, D5): an existing target directory is never overwritten
(``FileExistsError``). A run is built in a temporary sibling directory
``<run_id>.tmp-<hex>`` and renamed into place only after every file is
written; on any failure the temporary directory is removed. A sweep is built in
``<run_id>.incomplete/``, which is kept if the sweep stops early so that
``--resume`` can continue it, and renamed to ``<run_id>`` only on success.

Floats are written with ``repr`` so they round-trip exactly; NaN is written as
``nan``. Every artifact is synthetic simulation output, not market data.
"""

import contextlib
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import secrets
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np

from financial_contagion_sim import __version__
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import (
    SCHEMA_VERSION,
    SimulationConfig,
    SimulationResult,
)

ARTIFACT_SCHEMA_VERSION = 1
"""Version of the on-disk artifact layout."""

METRICS_COLUMNS = (
    "time",
    "S",
    "A",
    "I",
    "r",
    "E",
    "E_delta",
    "E_p",
    "infected_fraction",
)
INCOMPLETE_SUFFIX = ".incomplete"

RunKind = Literal["run", "sweep"]


# --- configuration serialization ---------------------------------------------


def config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Return a JSON-ready dict of every :class:`SimulationConfig` field.

    A ``SeedSequence`` seed is stored as ``{"entropy": ..., "spawn_key": [...]}``.

    Raises:
        TypeError: ``config`` is not a ``SimulationConfig``.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError(f"config must be SimulationConfig, got {type(config).__name__}")
    data = dataclasses.asdict(config)
    seed = config.seed
    if isinstance(seed, np.random.SeedSequence):
        data["seed"] = {"entropy": seed.entropy, "spawn_key": list(seed.spawn_key)}
    return data


def config_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Rebuild a :class:`SimulationConfig` from :func:`config_to_dict` output.

    Validation is left to ``SAISParams`` and ``SimulationConfig``.

    Raises:
        TypeError: ``data`` is not a mapping or has the wrong shape.
        ValueError: a value is invalid (raised by the config classes).
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"config data must be a mapping, got {type(data).__name__}")
    values = dict(data)
    params = values.pop("sais_params", None)
    if not isinstance(params, Mapping):
        raise TypeError("config data must contain a 'sais_params' mapping")
    seed = values.get("seed")
    if isinstance(seed, Mapping):
        values["seed"] = np.random.SeedSequence(
            entropy=seed["entropy"], spawn_key=tuple(seed.get("spawn_key", ()))
        )
    return SimulationConfig(sais_params=SAISParams(**params), **values)


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: object) -> str:
    """SHA-256 hex digest of :func:`canonical_json` of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def make_run_id(
    kind: RunKind, config_hash: str, seed: object, run_name: str | None = None
) -> str:
    """Directory name for a run: ``run_<hash8>_seed<seed>`` or ``run_name``.

    Raises:
        ValueError: ``kind`` is unknown, or ``run_name`` is empty or contains a
            path separator or reserved suffix.
    """
    if kind not in ("run", "sweep"):
        raise ValueError(f"kind must be 'run' or 'sweep', got {kind!r}")
    if run_name is not None:
        if (
            not run_name
            or run_name in (".", "..")
            or any(sep in run_name for sep in ("/", "\\"))
            or ".tmp-" in run_name
            or run_name.endswith(INCOMPLETE_SUFFIX)
        ):
            raise ValueError(f"invalid run name {run_name!r}")
        return run_name
    if isinstance(seed, np.random.SeedSequence):
        seed = seed.entropy
    return f"{kind}_{config_hash[:8]}_seed{seed}"


# --- metadata ----------------------------------------------------------------


def git_revision() -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for this package's repository, or ``(None, None)``.

    ``None`` means git or the repository is unavailable; that is not an error.
    """
    cwd = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def environment_metadata(argv: Sequence[str]) -> dict[str, Any]:
    """Versions, git state, UTC timestamp, argv and the synthetic flag."""
    commit, dirty = git_revision()
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "result_schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "note": "Synthetic simulation output; not empirical market data.",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": list(argv),
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": {
            "financial_contagion_sim": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "plotly": importlib.metadata.version("plotly"),
        },
        "platform": platform.platform(),
        "executable": sys.executable,
    }


# --- file writers ------------------------------------------------------------


def write_json(path: Path, obj: object) -> None:
    """Write indented JSON with a trailing newline (UTF-8)."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_float(value: float) -> str:
    """``repr`` of a float, so it round-trips exactly; NaN becomes ``nan``."""
    return repr(float(value))


def write_metrics_csv(path: Path, result: SimulationResult) -> None:
    """Write the recorded series of ``result`` as CSV (:data:`METRICS_COLUMNS`)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(METRICS_COLUMNS)
        for k in range(len(result.times)):
            writer.writerow(
                [
                    format_float(result.times[k]),
                    int(result.S[k]),
                    int(result.A[k]),
                    int(result.I[k]),
                    format_float(result.r[k]),
                    format_float(result.E[k]),
                    format_float(result.E_delta[k]),
                    format_float(result.E_p[k]),
                    format_float(result.infected_fraction[k]),
                ]
            )


def read_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a metrics CSV back into arrays (ints for S, A, I; floats otherwise)."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    if tuple(header) != METRICS_COLUMNS:
        raise ValueError(f"unexpected metrics header {header!r}")
    out: dict[str, np.ndarray] = {}
    for position, name in enumerate(header):
        values = [row[position] for row in body]
        if name in ("S", "A", "I"):
            out[name] = np.array([int(v) for v in values], dtype=np.int64)
        else:
            out[name] = np.array([float(v) for v in values], dtype=np.float64)
    return out


def run_config_document(config: SimulationConfig) -> dict[str, Any]:
    """The content of ``config.json`` for a single run."""
    return {
        "kind": "run",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": config_to_dict(config),
    }


def run_config_hash(config: SimulationConfig) -> str:
    """Hash identifying a single-run configuration (seed included)."""
    return digest(run_config_document(config))


# --- directories -------------------------------------------------------------


def check_output_dir(output_dir: Path) -> None:
    """Raise ``NotADirectoryError`` if ``output_dir`` (or a parent) is a file."""
    for candidate in (output_dir, *output_dir.parents):
        if candidate.exists() and not candidate.is_dir():
            raise NotADirectoryError(f"output path {str(candidate)!r} is a file")


@contextlib.contextmanager
def staged_directory(output_dir: Path, run_id: str) -> Iterator[Path]:
    """Yield a temporary directory that becomes ``output_dir / run_id`` on success.

    Raises:
        FileExistsError: the final directory already exists (nothing is
            touched).
        NotADirectoryError: ``output_dir`` or a parent is a file.
    """
    final = output_dir / run_id
    check_output_dir(output_dir)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing {str(final)!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"{run_id}.tmp-{secrets.token_hex(4)}"
    staging.mkdir()
    try:
        yield staging
        if final.exists():
            raise FileExistsError(f"refusing to overwrite existing {str(final)!r}")
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_run_artifacts(
    output_dir: Path,
    run_id: str,
    result: SimulationResult,
    *,
    argv: Sequence[str],
    plot: bool,
) -> Path:
    """Write config.json, metadata.json, metrics.csv and (optionally) the PNG.

    Returns:
        The final run directory.

    Raises:
        TypeError: ``result`` has no ``config`` (it was not made by
            ``run_simulation``).
        FileExistsError: the run directory already exists.
        NotADirectoryError: ``output_dir`` or a parent is a file.
    """
    if result.config is None:
        raise TypeError("run artifacts need a result from run_simulation (config set)")
    with staged_directory(output_dir, run_id) as staging:
        write_json(staging / "config.json", run_config_document(result.config))
        metadata = environment_metadata(argv)
        metadata.update(
            {
                "run_id": run_id,
                "config_hash": run_config_hash(result.config),
                "termination_reason": result.termination_reason,
                "final_time": result.final_time,
                "n_events": result.n_events,
                "n_sais_events": result.n_sais_events,
                "n_proposed": result.n_proposed,
                "n_accepted": result.n_accepted,
                "n_rejected": result.n_rejected,
                "n_zero_accepted": result.n_zero_accepted,
                "realized_r0": result.realized_r0,
                "realized_initial_infected_fraction": (
                    result.realized_initial_infected_fraction
                ),
            }
        )
        write_json(staging / "metadata.json", metadata)
        write_metrics_csv(staging / "metrics.csv", result)
        if plot:
            # Imported here: visualizer -> sweep -> artifacts would otherwise
            # form an import cycle.
            from financial_contagion_sim.visualizer import plot_timeseries, save_figure

            save_figure(plot_timeseries(result), staging / "timeseries.png")
    return output_dir / run_id


def run_artifact_names(plot: bool) -> list[str]:
    """Files a run writes, for ``--dry-run`` listings."""
    names = ["config.json", "metadata.json", "metrics.csv"]
    return [*names, "timeseries.png"] if plot else names
