"""Parameter sweeps and steady-state estimation (spec §5, step 3; Phase 4 B).

Responsibility: build the (alpha, r0, rho_0) grid, run every grid point with the
Phase 3 :func:`~financial_contagion_sim.simulator.run_simulation`, estimate the
steady state of each run, and aggregate the repetitions into surfaces. No
simulation, SAIS or energy logic lives here.

Grid (B1, B2): ``alpha`` and ``r0`` are the two surface axes and ``rho_0``
(the initial infected fraction) is the third parameter, one surface per value.
Grid points are numbered with ``rho_0`` varying slowest, then ``r0``, then
``alpha``: ``index = (rho0_index * n_r0 + r0_index) * n_alpha + alpha_index``.
Every other parameter comes from a required base configuration.

Steady state (B3; the spec does not define one). Each run is recorded on the
``num_points`` grid of :class:`~financial_contagion_sim.simulator.SimulationConfig`
(``record="grid"``, ``stop_on_sais_absorbed=False``). The window is every grid
time ``t >= 0.8 * t_max``, split into two halves at ``0.9 * t_max``:

- ``rho_inf`` = mean infected fraction over the window; converged iff the two
  half-means differ by at most ``tol_rho`` (absolute, default 0.02).
- ``E_ss`` = mean energy E over the window; converged iff the half-means differ
  by at most ``tol_e_rel * |mean over the window|`` (relative, default 0.05).
  Relative because E is of order 1e-4 on sparse graphs, where any fixed
  absolute tolerance would accept everything.
- A run that ended ``"absorbed"`` is constant and therefore converged. A run
  that stopped before ``t_max`` (``"max_events"``) cannot fill the window and
  is not converged. A metric that is not converged is NaN, never its last
  value.

Aggregation (B3, B4): per grid point and metric, the mean over the converged
repetitions, provided at least ``ceil(min_converged_fraction * repetitions)``
converged (default 0.5); otherwise NaN. The sample standard deviation
(``ddof=1``) and standard error are reported too (NaN when fewer than two
repetitions converged), with the converged count ``k``.

Seeding (B7): repetition ``rep`` of grid point ``index`` uses
``SeedSequence(entropy=base_seed, spawn_key=(index, rep))``, so every run's
result depends only on its key: execution order and resuming do not change it.

Interpretation notes (from Phase 3): with the spec's flip rule (§3.3), edges
with no triangles and no infected endpoint random-walk towards r = 0.5
regardless of contagion; at alpha = 1 the acceptance energy ignores node
states, and at alpha = 0 only the spreading energy counts. All output is
synthetic simulation data.
"""

import csv
import dataclasses
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
import numpy.typing as npt

from financial_contagion_sim.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    config_from_dict,
    config_to_dict,
    digest,
    format_float,
    write_json,
)
from financial_contagion_sim.graph_builder import _check_int, _check_unit_interval
from financial_contagion_sim.simulator import (
    SimulationConfig,
    SimulationResult,
    TerminationReason,
    run_simulation,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

RESULTS_COLUMNS = (
    "point_index",
    "repetition",
    "alpha_index",
    "r0_index",
    "rho0_index",
    "alpha",
    "r0",
    "rho0",
    "rho_inf",
    "e_ss",
    "rho_converged",
    "e_converged",
    "termination_reason",
    "n_events",
)
_SWEPT_FIELDS = ("alpha", "r0", "initial_infected_fraction", "seed")


# --- grid ------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """One grid point: its flat index, axis indices and parameter values."""

    index: int
    alpha_index: int
    r0_index: int
    rho0_index: int
    alpha: float
    r0: float
    rho0: float


@dataclass(frozen=True)
class SweepGrid:
    """The (alpha, r0, rho_0) grid.

    ``alpha`` and ``r0`` must be strictly increasing (they are surface axes);
    ``rho0`` values must be distinct. Every value lies in ``[0, 1]``.

    Raises:
        TypeError: an axis is not a tuple of real numbers.
        ValueError: an axis is empty, out of range, or not ordered.
    """

    alpha: tuple[float, ...]
    r0: tuple[float, ...]
    rho0: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("alpha", "r0", "rho0"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(values).__name__}")
            if not values:
                raise ValueError(f"{name} must have at least one value")
            for value in values:
                _check_unit_interval(name, value)
        for name in ("alpha", "r0"):
            values = getattr(self, name)
            if any(b <= a for a, b in zip(values, values[1:], strict=False)):
                raise ValueError(f"{name} values must be strictly increasing")
        if len(set(self.rho0)) != len(self.rho0):
            raise ValueError("rho0 values must be distinct")

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_rho0, n_r0, n_alpha)``: the shape of every surface array."""
        return len(self.rho0), len(self.r0), len(self.alpha)

    @property
    def n_points(self) -> int:
        return len(self.rho0) * len(self.r0) * len(self.alpha)

    def index_of(self, alpha_index: int, r0_index: int, rho0_index: int) -> int:
        """Flat index of a grid point (rho_0 slowest, then r0, then alpha)."""
        n_rho0, n_r0, n_alpha = self.shape
        if not (0 <= alpha_index < n_alpha and 0 <= r0_index < n_r0):
            raise ValueError("axis index out of range")
        if not 0 <= rho0_index < n_rho0:
            raise ValueError("axis index out of range")
        return (rho0_index * n_r0 + r0_index) * n_alpha + alpha_index

    def point(self, index: int) -> GridPoint:
        """The grid point with flat index ``index``."""
        _check_int("index", index, minimum=0)
        if index >= self.n_points:
            raise ValueError(f"index must be < {self.n_points}, got {index}")
        _, n_r0, n_alpha = self.shape
        rest, alpha_index = divmod(index, n_alpha)
        rho0_index, r0_index = divmod(rest, n_r0)
        return GridPoint(
            index,
            alpha_index,
            r0_index,
            rho0_index,
            self.alpha[alpha_index],
            self.r0[r0_index],
            self.rho0[rho0_index],
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"alpha": list(self.alpha), "r0": list(self.r0), "rho0": list(self.rho0)}


def make_grid(
    alpha_range: tuple[float, float, int],
    r0_range: tuple[float, float, int],
    rho0_values: Iterable[float],
) -> SweepGrid:
    """Build a grid from ``(start, stop, count)`` ranges, endpoints included.

    Raises:
        TypeError: a range is malformed.
        ValueError: a count is below 1 or a value is out of range.
    """
    axes = []
    for name, spec in (("alpha_range", alpha_range), ("r0_range", r0_range)):
        if not isinstance(spec, tuple) or len(spec) != 3:
            raise TypeError(f"{name} must be a (start, stop, count) tuple")
        start, stop, count = spec
        _check_unit_interval(f"{name} start", start)
        _check_unit_interval(f"{name} stop", stop)
        _check_int(f"{name} count", count, minimum=1)
        values = np.linspace(start, stop, count) if count > 1 else np.array([start])
        axes.append(tuple(float(v) for v in values))
    return SweepGrid(axes[0], axes[1], tuple(float(v) for v in rho0_values))


def count_runs(grid: SweepGrid, repetitions: int) -> int:
    """Total number of simulations a sweep performs."""
    _check_int("repetitions", repetitions, minimum=1)
    return grid.n_points * repetitions


# --- steady state --------------------------------------------------------------


@dataclass(frozen=True)
class SteadyStateSettings:
    """Steady-state rule (Phase 4 B3).

    Attributes:
        window_start: Window start as a fraction of ``t_max``, in ``(0, 1)``.
        tol_rho: Absolute tolerance on the infected-fraction half-means.
        tol_e_rel: Relative tolerance on the energy half-means.
        min_converged_fraction: Fraction of repetitions that must converge for
            a grid point to get a value, in ``(0, 1]``.
    """

    window_start: float = 0.8
    tol_rho: float = 0.02
    tol_e_rel: float = 0.05
    min_converged_fraction: float = 0.5

    def __post_init__(self) -> None:
        _check_unit_interval("window_start", self.window_start)
        if not 0 < self.window_start < 1:
            raise ValueError(f"window_start must be in (0, 1), got {self.window_start}")
        for name in ("tol_rho", "tol_e_rel"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        _check_unit_interval("min_converged_fraction", self.min_converged_fraction)
        if self.min_converged_fraction == 0:
            raise ValueError("min_converged_fraction must be > 0")

    @property
    def window_split(self) -> float:
        """Split point between the two halves, as a fraction of ``t_max``."""
        return (1 + self.window_start) / 2


@dataclass(frozen=True)
class SteadyState:
    """Steady-state estimate of one run; a non-converged metric is NaN."""

    rho_inf: float
    e_ss: float
    rho_converged: bool
    e_converged: bool


def estimate_steady_state(
    result: SimulationResult, settings: SteadyStateSettings | None = None
) -> SteadyState:
    """Estimate ``rho_inf`` and ``E_ss`` of one run (Phase 4 B3).

    Args:
        result: A grid-recorded result of
            :func:`~financial_contagion_sim.simulator.run_simulation`.
        settings: Steady-state rule; defaults to :class:`SteadyStateSettings`.

    Returns:
        The estimate; a metric that did not converge is NaN.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: ``result`` was not recorded on the time grid.
    """
    settings = SteadyStateSettings() if settings is None else settings
    if not isinstance(settings, SteadyStateSettings):
        raise TypeError("settings must be SteadyStateSettings")
    if not isinstance(result, SimulationResult):
        raise TypeError(f"result must be SimulationResult, got {type(result).__name__}")
    config = result.config
    if config is None or config.record != "grid":
        raise ValueError("steady-state estimation needs a run with record='grid'")
    nan = math.nan
    if result.termination_reason == "absorbed":
        return SteadyState(
            float(result.infected_fraction[-1]), float(result.E[-1]), True, True
        )
    if result.termination_reason != "t_max":
        return SteadyState(nan, nan, False, False)

    times = result.times
    start, split = (
        settings.window_start * config.t_max,
        settings.window_split * config.t_max,
    )
    window = times >= start
    first, second = window & (times < split), times >= split
    if not first.any() or not second.any():
        return SteadyState(nan, nan, False, False)

    def halves(series: FloatArray) -> tuple[float, float, float]:
        return (
            float(series[window].mean()),
            float(series[first].mean()),
            float(series[second].mean()),
        )

    rho_mean, rho_a, rho_b = halves(result.infected_fraction)
    e_mean, e_a, e_b = halves(result.E)
    rho_ok = abs(rho_a - rho_b) <= settings.tol_rho
    e_ok = abs(e_a - e_b) <= settings.tol_e_rel * abs(e_mean)
    return SteadyState(
        rho_mean if rho_ok else nan, e_mean if e_ok else nan, rho_ok, e_ok
    )


# --- sweep results ---------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """Outcome of one (grid point, repetition) run."""

    point_index: int
    repetition: int
    alpha_index: int
    r0_index: int
    rho0_index: int
    alpha: float
    r0: float
    rho0: float
    rho_inf: float
    e_ss: float
    rho_converged: bool
    e_converged: bool
    termination_reason: str
    n_events: int

    @property
    def key(self) -> tuple[int, int]:
        return self.point_index, self.repetition


@dataclass(frozen=True, eq=False)
class MetricSurface:
    """Per-grid-point aggregate of one metric; arrays have shape ``grid.shape``."""

    mean: FloatArray
    std: FloatArray
    se: FloatArray
    k_converged: IntArray


@dataclass(frozen=True, eq=False)
class SweepResult:
    """A completed sweep, serializable with :func:`save_sweep`/:func:`load_sweep`."""

    grid: SweepGrid
    repetitions: int
    settings: SteadyStateSettings
    base_config: SimulationConfig
    seed: int
    config_hash: str
    runs: tuple[RunRecord, ...]
    rho_inf: MetricSurface
    energy: MetricSurface
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @property
    def non_converged_points(self) -> dict[str, int]:
        """Grid points whose surface value is NaN, per metric."""
        return {
            "rho_inf": int(np.isnan(self.rho_inf.mean).sum()),
            "energy": int(np.isnan(self.energy.mean).sum()),
        }


def sweep_definition(
    base: SimulationConfig,
    grid: SweepGrid,
    repetitions: int,
    settings: SteadyStateSettings,
) -> dict[str, Any]:
    """Canonical description of a sweep; its digest is the config hash (B8)."""
    _check_sweep_inputs(base, grid, repetitions, settings)
    base_dict = config_to_dict(base)
    for name in _SWEPT_FIELDS[:3]:
        base_dict[name] = None
    return {
        "kind": "sweep",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "base": base_dict,
        "grid": grid.to_dict(),
        "repetitions": repetitions,
        "steady_state": dataclasses.asdict(settings),
    }


def sweep_config_for(
    base: SimulationConfig, grid: SweepGrid, point_index: int, repetition: int
) -> SimulationConfig:
    """The exact configuration simulated for one (point, repetition) key."""
    point = grid.point(point_index)
    _check_int("repetition", repetition, minimum=0)
    seed = base.seed
    assert isinstance(seed, int)  # checked by _check_sweep_inputs
    return dataclasses.replace(
        base,
        alpha=point.alpha,
        r0=point.r0,
        initial_infected_fraction=point.rho0,
        seed=np.random.SeedSequence(entropy=seed, spawn_key=(point_index, repetition)),
    )


def run_sweep(
    base: SimulationConfig,
    grid: SweepGrid,
    repetitions: int,
    settings: SteadyStateSettings | None = None,
    *,
    completed: Mapping[tuple[int, int], RunRecord] | None = None,
    on_run: Callable[[RunRecord], None] | None = None,
) -> SweepResult:
    """Run every (grid point, repetition) and aggregate (Phase 4 B).

    Runs are executed serially in ascending ``(point_index, repetition)`` order;
    keys present in ``completed`` are reused instead of re-simulated. Any
    exception from a simulation propagates and aborts the sweep.

    Args:
        base: Every non-swept parameter. ``alpha``, ``r0``,
            ``initial_infected_fraction`` and ``seed`` are replaced per run;
            ``base.seed`` must be an ``int`` (the sweep's base seed). It must use
            ``record="grid"`` and ``stop_on_sais_absorbed=False``.
        grid: The parameter grid.
        repetitions: Runs per grid point, ``>= 1``.
        settings: Steady-state rule; defaults to :class:`SteadyStateSettings`.
        completed: Records from an earlier, interrupted sweep.
        on_run: Called with each newly computed record (for checkpointing).

    Returns:
        The aggregated :class:`SweepResult`.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    settings = SteadyStateSettings() if settings is None else settings
    definition = sweep_definition(base, grid, repetitions, settings)
    done = dict(completed or {})
    for key, record in done.items():
        if key != record.key or not (
            0 <= key[0] < grid.n_points and 0 <= key[1] < repetitions
        ):
            raise ValueError(f"completed record {key} does not belong to this sweep")
    for point_index in range(grid.n_points):
        point = grid.point(point_index)
        for repetition in range(repetitions):
            if (point_index, repetition) in done:
                continue
            result = run_simulation(
                sweep_config_for(base, grid, point_index, repetition)
            )
            state = estimate_steady_state(result, settings)
            record = RunRecord(
                point_index,
                repetition,
                point.alpha_index,
                point.r0_index,
                point.rho0_index,
                point.alpha,
                point.r0,
                point.rho0,
                state.rho_inf,
                state.e_ss,
                state.rho_converged,
                state.e_converged,
                result.termination_reason,
                result.n_events,
            )
            done[record.key] = record
            if on_run is not None:
                on_run(record)
    return _assemble(base, grid, repetitions, settings, definition, done.values())


def aggregate(
    grid: SweepGrid,
    repetitions: int,
    settings: SteadyStateSettings,
    records: Iterable[RunRecord],
) -> tuple[MetricSurface, MetricSurface]:
    """Per-point mean, std (ddof=1), SE and converged count for both metrics."""
    _check_int("repetitions", repetitions, minimum=1)
    required = math.ceil(settings.min_converged_fraction * repetitions)
    values: dict[str, list[list[float]]] = {
        "rho": [[] for _ in range(grid.n_points)],
        "e": [[] for _ in range(grid.n_points)],
    }
    for record in records:
        if record.rho_converged:
            values["rho"][record.point_index].append(record.rho_inf)
        if record.e_converged:
            values["e"][record.point_index].append(record.e_ss)
    surfaces = []
    for metric in ("rho", "e"):
        mean = np.full(grid.n_points, np.nan)
        std = np.full(grid.n_points, np.nan)
        k = np.zeros(grid.n_points, dtype=np.int64)
        for index, sample in enumerate(values[metric]):
            k[index] = len(sample)
            if len(sample) >= max(required, 1):
                mean[index] = float(np.mean(sample))
                if len(sample) >= 2:
                    std[index] = float(np.std(sample, ddof=1))
        se = std / np.sqrt(np.where(k > 0, k, 1))
        surfaces.append(
            MetricSurface(
                mean.reshape(grid.shape),
                std.reshape(grid.shape),
                se.reshape(grid.shape),
                k.reshape(grid.shape),
            )
        )
    return surfaces[0], surfaces[1]


# --- persistence --------------------------------------------------------------------


def write_results_header(path: Path) -> None:
    """Create ``sweep_results.csv`` with its header row."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(RESULTS_COLUMNS)


def append_record(path: Path, record: RunRecord) -> None:
    """Append one record and flush it to disk (the B8 checkpoint)."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(_record_row(record))
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> list[RunRecord]:
    """Read complete records; a truncated or malformed trailing row is dropped.

    Every complete row ends with a newline, so a final line without one was
    cut off mid-write and is ignored even if it happens to parse.
    """
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    rows = list(csv.reader(text.splitlines()))
    if not rows or tuple(rows[0]) != RESULTS_COLUMNS:
        raise ValueError(f"{str(path)!r} is not a sweep results file")
    records = []
    for row in rows[1:]:
        try:
            records.append(_parse_row(row))
        except (ValueError, IndexError):
            break
    return records


def rewrite_records(path: Path, records: Iterable[RunRecord]) -> None:
    """Replace ``path`` with exactly ``records`` (used before resuming)."""
    temporary = path.with_name(path.name + ".rewrite")
    write_results_header(temporary)
    for record in records:
        append_record(temporary, record)
    temporary.replace(path)


def save_sweep_summary(
    path: Path, result: SweepResult, *, runtime_seconds: float
) -> None:
    """Write ``sweep_summary.json``: grid, settings, counts, runtime and hash."""

    write_json(
        path,
        {
            "schema_version": result.schema_version,
            "config_hash": result.config_hash,
            "seed": result.seed,
            "grid": result.grid.to_dict(),
            "repetitions": result.repetitions,
            "steady_state": dataclasses.asdict(result.settings),
            "n_runs": len(result.runs),
            "non_converged_points": result.non_converged_points,
            "non_converged_runs": {
                "rho_inf": sum(not r.rho_converged for r in result.runs),
                "energy": sum(not r.e_converged for r in result.runs),
            },
            "runtime_seconds": runtime_seconds,
            "synthetic": True,
        },
    )


def load_sweep(definition: Mapping[str, Any], results_csv: Path) -> SweepResult:
    """Rebuild a :class:`SweepResult` from its definition and results file.

    Raises:
        ValueError: the results file is incomplete or does not match.
    """
    base_dict = dict(definition["base"])
    grid = SweepGrid(*(tuple(definition["grid"][k]) for k in ("alpha", "r0", "rho0")))
    base_dict["alpha"], base_dict["r0"] = grid.alpha[0], grid.r0[0]
    base_dict["initial_infected_fraction"] = grid.rho0[0]
    base = config_from_dict(base_dict)
    settings = SteadyStateSettings(**definition["steady_state"])
    repetitions = int(definition["repetitions"])
    records = read_records(results_csv)
    if len(records) != grid.n_points * repetitions:
        raise ValueError("sweep results are incomplete")
    definition_now = sweep_definition(base, grid, repetitions, settings)
    return _assemble(base, grid, repetitions, settings, definition_now, records)


def _assemble(
    base: SimulationConfig,
    grid: SweepGrid,
    repetitions: int,
    settings: SteadyStateSettings,
    definition: Mapping[str, Any],
    records: Iterable[RunRecord],
) -> SweepResult:
    ordered = tuple(sorted(records, key=lambda r: r.key))
    expected = [(p, r) for p in range(grid.n_points) for r in range(repetitions)]
    if [r.key for r in ordered] != expected:
        raise ValueError("sweep records do not cover every (point, repetition) once")
    rho, energy = aggregate(grid, repetitions, settings, ordered)
    seed = base.seed
    assert isinstance(seed, int)
    return SweepResult(
        grid,
        repetitions,
        settings,
        base,
        seed,
        digest(definition),
        ordered,
        rho,
        energy,
    )


def _check_sweep_inputs(
    base: object, grid: object, repetitions: object, settings: object
) -> None:
    if not isinstance(base, SimulationConfig):
        raise TypeError(f"base must be SimulationConfig, got {type(base).__name__}")
    if not isinstance(grid, SweepGrid):
        raise TypeError(f"grid must be SweepGrid, got {type(grid).__name__}")
    if not isinstance(settings, SteadyStateSettings):
        raise TypeError("settings must be SteadyStateSettings")
    _check_int("repetitions", repetitions, minimum=1)
    if not isinstance(base.seed, int) or isinstance(base.seed, bool):
        raise TypeError("a sweep's base seed must be an int")
    if base.record != "grid":
        raise ValueError("sweep runs must use record='grid'")
    if base.stop_on_sais_absorbed:
        raise ValueError("sweep runs must use stop_on_sais_absorbed=False")


def _record_row(record: RunRecord) -> list[object]:
    return [
        record.point_index,
        record.repetition,
        record.alpha_index,
        record.r0_index,
        record.rho0_index,
        format_float(record.alpha),
        format_float(record.r0),
        format_float(record.rho0),
        format_float(record.rho_inf),
        format_float(record.e_ss),
        record.rho_converged,
        record.e_converged,
        record.termination_reason,
        record.n_events,
    ]


def _parse_bool(text: str) -> bool:
    if text not in ("True", "False"):
        raise ValueError(f"not a bool: {text!r}")
    return text == "True"


def _parse_row(row: list[str]) -> RunRecord:
    if len(row) != len(RESULTS_COLUMNS):
        raise ValueError("wrong number of columns")
    if row[12] not in get_args(TerminationReason):
        raise ValueError(f"unknown termination reason {row[12]!r}")
    return RunRecord(
        int(row[0]),
        int(row[1]),
        int(row[2]),
        int(row[3]),
        int(row[4]),
        float(row[5]),
        float(row[6]),
        float(row[7]),
        float(row[8]),
        float(row[9]),
        _parse_bool(row[10]),
        _parse_bool(row[11]),
        row[12],
        int(row[13]),
    )
