"""Command-line interface (spec §4: ``main.py``, an argparse interface).

Responsibility: parse arguments, build the configuration, call the package and
write artifacts. The CLI is orchestration only: model checks come from the
configuration classes, defaults from :class:`SimulationConfig` and
:mod:`financial_contagion_sim.sweep`, and files from
:mod:`financial_contagion_sim.artifacts`.

Subcommands::

    financial-contagion-sim run   ...   one simulation -> config, metadata,
                                        metrics.csv, timeseries.png
    financial-contagion-sim sweep ...   alpha x r0 x rho_0 grid -> results CSV,
                                        summary, two HTML surfaces

Exit codes: 0 success; 1 runtime failure; 2 argument or configuration error;
3 refused to overwrite (or resume) an existing run directory.

All output is synthetic simulation data, not empirical market data.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from financial_contagion_sim import artifacts, sweep
from financial_contagion_sim.simulator import SimulationConfig, run_simulation
from financial_contagion_sim.visualizer import plot_surface, write_surface_html

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3

PROG = "financial-contagion-sim"
DEFAULT_OUTPUT_DIR = "outputs"

# SimulationConfig field -> command-line flag (Phase 4 C2).
CONFIG_FLAGS = {
    "n": "--n",
    "topology": "--model",
    "ba_m": "--ba-m",
    "er_p": "--er-p",
    "alpha": "--alpha",
    "r0": "--r0",
    "initial_infected_fraction": "--rho0",
    "edge_update_rate": "--gamma",
    "t_max": "--t-max",
    "seed": "--seed",
    "max_events": "--max-events",
    "num_points": "--num-points",
    "stop_on_sais_absorbed": "--stop-on-sais-absorbed",
}
SAIS_FLAGS = {
    "beta": "--beta",
    "beta_a": "--beta-a",
    "kappa": "--kappa",
    "delta": "--delta",
}
CONTROL_FIELDS = ("max_events", "num_points", "stop_on_sais_absorbed")
SWEPT_FIELDS = ("alpha", "r0", "initial_infected_fraction")
SWEEP_FLAGS = {
    "alpha": "--alpha-range",
    "alpha_range": "--alpha-range",
    "r0": "--r0-range",
    "r0_range": "--r0-range",
    "rho0": "--rho0-values",
    "initial_infected_fraction": "--rho0-values",
    "repetitions": "--reps",
    "tol_rho": "--tol-rho",
    "tol_e_rel": "--tol-e",
    "window_start": "--config",
    "min_converged_fraction": "--config",
}

EPILOG = """\
exit codes:
  0  success
  1  runtime failure
  2  argument or configuration error
  3  refused to overwrite (or resume) an existing run directory

All output is synthetic simulation data, not empirical market data."""


class _UsageError(Exception):
    """An argument or configuration problem, reported as ``error: <flag>: <msg>``."""

    def __init__(self, flag: str, message: str) -> None:
        super().__init__(f"{flag}: {message}")
        self.flag = flag
        self.message = message


class _Refused(Exception):
    """A refusal to overwrite or resume (exit code 3)."""


# --- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``run`` and ``sweep``."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Co-evolutionary financial contagion & systemic risk simulator "
            "(synthetic simulation output)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_package_version()}"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="run one simulation and save its artifacts",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model_options(run, sweep_mode=False)
    run.add_argument(
        "--no-plot", action="store_true", help="do not write timeseries.png"
    )

    sweep_parser = commands.add_parser(
        "sweep",
        help="sweep alpha x r0 x rho_0 and export steady-state surfaces",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model_options(sweep_parser, sweep_mode=True)
    grid = sweep_parser.add_argument_group("sweep grid and steady state")
    grid.add_argument(
        "--alpha-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "COUNT"),
        help="alpha axis, endpoints included "
        f"(default {_fmt(sweep.DEFAULT_ALPHA_RANGE)})",
    )
    grid.add_argument(
        "--r0-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "COUNT"),
        help=f"r0 axis, endpoints included (default {_fmt(sweep.DEFAULT_R0_RANGE)})",
    )
    grid.add_argument(
        "--rho0-values",
        nargs="+",
        type=float,
        metavar="RHO0",
        help="rho_0 values, one surface each "
        f"(default {_fmt(sweep.DEFAULT_RHO0_VALUES)})",
    )
    grid.add_argument(
        "--reps",
        type=int,
        help=f"repetitions per grid point (default {sweep.DEFAULT_REPETITIONS})",
    )
    settings = sweep.SteadyStateSettings()
    grid.add_argument(
        "--tol-rho",
        type=float,
        help=f"absolute tolerance for rho_inf (default {settings.tol_rho})",
    )
    grid.add_argument(
        "--tol-e",
        type=float,
        help=f"relative tolerance for E_ss (default {settings.tol_e_rel})",
    )
    grid.add_argument(
        "--max-runs",
        type=int,
        help="refuse sweeps with more runs than this "
        f"(default {sweep.DEFAULT_MAX_RUNS})",
    )
    grid.add_argument(
        "--resume", action="store_true", help="continue an interrupted sweep"
    )
    sweep_parser.add_argument(
        "--no-html", action="store_true", help="do not write the HTML surfaces"
    )
    return parser


def _add_model_options(parser: argparse.ArgumentParser, *, sweep_mode: bool) -> None:
    model = parser.add_argument_group(
        "model parameters (required unless given in --config)"
    )
    model.add_argument(
        "--config",
        type=Path,
        metavar="FILE.json",
        help="load a resolved config.json; flags override its values",
    )
    model.add_argument(
        "--model",
        choices=("barabasi_albert", "erdos_renyi"),
        help="network topology (spec §5)",
    )
    model.add_argument("--n", type=int, help="number of institutions, >= 1")
    model.add_argument("--ba-m", type=int, help="Barabási–Albert edges per new node")
    model.add_argument("--er-p", type=float, help="Erdős–Rényi edge probability")
    swept = " (swept: use the range options)" if sweep_mode else ""
    model.add_argument("--alpha", type=float, help=f"energy weight in [0, 1]{swept}")
    model.add_argument("--r0", type=float, help=f"positive-link ratio in [0, 1]{swept}")
    model.add_argument(
        "--rho0", type=float, help=f"initial infected fraction in [0, 1]{swept}"
    )
    model.add_argument("--beta", type=float, help="S->I rate per infected neighbour")
    model.add_argument("--beta-a", type=float, help="A->I rate, 0 <= beta_a < beta")
    model.add_argument("--kappa", type=float, help="S->A rate per infected neighbour")
    model.add_argument("--delta", type=float, help="I->S recapitalization rate")
    model.add_argument("--gamma", type=float, help="edge-flip proposal rate per edge")
    model.add_argument("--t-max", type=float, help="time horizon")
    model.add_argument("--seed", type=int, help="random seed, >= 0")

    defaults = _config_defaults()
    control = parser.add_argument_group("control options")
    control.add_argument(
        "--max-events",
        type=int,
        help=f"cap on all events (default {defaults['max_events']})",
    )
    control.add_argument(
        "--num-points",
        type=int,
        help=f"recorded grid points (default {defaults['num_points']})",
    )
    if not sweep_mode:
        control.add_argument(
            "--stop-on-sais-absorbed",
            action=argparse.BooleanOptionalAction,
            help=f"stop when no SAIS jump is possible "
            f"(default {defaults['stop_on_sais_absorbed']})",
        )
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"parent directory for run directories (default {DEFAULT_OUTPUT_DIR}/)",
    )
    output.add_argument(
        "--run-name", help="directory name instead of the derived run id"
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan; write nothing",
    )


# --- entry point -------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code.

    As usual with argparse, ``--help``, ``--version`` and malformed arguments
    exit through ``SystemExit`` (codes 0 and 2) while parsing; every other
    outcome is returned as an exit code (:data:`EXIT_OK` etc.).
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    handlers: dict[str, Callable[[argparse.Namespace, list[str]], int]] = {
        "run": _command_run,
        "sweep": _command_sweep,
    }
    try:
        return handlers[args.command](args, args_list)
    except _UsageError as exc:
        print(f"error: {exc.flag}: {exc.message}", file=sys.stderr)
        return EXIT_USAGE
    except _Refused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def _command_run(args: argparse.Namespace, argv: list[str]) -> int:
    values = _resolve_values(args, sweep_mode=False)
    config = _build_config(values, sweep_mode=False)
    run_id = _run_id(
        "run", artifacts.run_config_hash(config), config.seed, args.run_name
    )
    output_dir: Path = args.output_dir
    target = output_dir / run_id
    names = artifacts.run_artifact_names(plot=not args.no_plot)
    if args.dry_run:
        print(f"dry run: would write {target}")
        for name in names:
            print(f"  {name}")
        return EXIT_OK
    _check_target(output_dir, target)
    result = run_simulation(config)
    try:
        run_dir = artifacts.write_run_artifacts(
            output_dir, run_id, result, argv=argv, plot=not args.no_plot
        )
    except FileExistsError as exc:
        raise _Refused(str(exc)) from exc
    except PermissionError as exc:
        print(f"error: --output-dir: cannot write: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    print(f"run directory: {run_dir}")
    _print_tree(run_dir)
    print(
        f"termination: {result.termination_reason} at t={result.final_time:g}; "
        f"events {result.n_events} (SAIS {sum(result.n_sais_events.values())}, "
        f"proposals {result.n_proposed}, accepted {result.n_accepted})"
    )
    print("synthetic simulation output; not empirical market data")
    return EXIT_OK


def _command_sweep(args: argparse.Namespace, argv: list[str]) -> int:
    for name in ("alpha", "r0", "rho0"):
        if getattr(args, name) is not None:
            flag = {
                "alpha": "--alpha-range",
                "r0": "--r0-range",
                "rho0": "--rho0-values",
            }
            raise _UsageError(f"--{name}", f"is swept; use {flag[name]} instead")
    values = _resolve_values(args, sweep_mode=True)
    file_doc = values.pop("__sweep__", {})
    grid = _build_grid(args, file_doc)
    repetitions = _first(
        args.reps, file_doc.get("repetitions"), sweep.DEFAULT_REPETITIONS
    )
    settings = _build_settings(args, file_doc)
    max_runs = _first(args.max_runs, None, sweep.DEFAULT_MAX_RUNS)
    values.update(
        alpha=grid.alpha[0], r0=grid.r0[0], initial_infected_fraction=grid.rho0[0]
    )
    values["record"], values["stop_on_sais_absorbed"] = "grid", False
    base = _build_config(values, sweep_mode=True)
    try:
        definition = sweep.sweep_definition(base, grid, repetitions, settings)
        total = sweep.count_runs(grid, repetitions)
    except (ValueError, TypeError) as exc:
        raise _usage_from(exc, sweep_mode=True) from exc
    if total > max_runs:
        raise _UsageError(
            "--max-runs",
            f"sweep needs {total} runs, more than --max-runs {max_runs}; "
            "raise --max-runs explicitly to allow it",
        )
    config_hash = artifacts.digest(definition)
    run_id = _run_id("sweep", config_hash, base.seed, args.run_name)
    output_dir: Path = args.output_dir
    final = output_dir / run_id
    incomplete = output_dir / f"{run_id}{artifacts.INCOMPLETE_SUFFIX}"
    files = [
        "config.json",
        "metadata.json",
        "sweep/sweep_results.csv",
        "sweep/sweep_summary.json",
    ]
    if not args.no_html:
        files += ["sweep/surface_rho_infinity.html", "sweep/surface_energy.html"]
    if args.dry_run:
        start = time.perf_counter()
        run_simulation(sweep.sweep_config_for(base, grid, 0, 0))
        pilot = time.perf_counter() - start
        print(f"dry run: would write {final}")
        for name in files:
            print(f"  {name}")
        print(
            f"runs: {total} ({grid.n_points} grid points x {repetitions} repetitions)"
        )
        print(
            f"estimated runtime: {pilot * total:.0f} s "
            f"(pilot run {pilot:.3f} s, serial)"
        )
        return EXIT_OK

    _check_target(output_dir, final)
    completed = _prepare_incomplete(incomplete, definition, config_hash, args.resume)
    results_csv = incomplete / "sweep" / "sweep_results.csv"
    start = time.perf_counter()
    result = sweep.run_sweep(
        base,
        grid,
        repetitions,
        settings,
        completed={r.key: r for r in completed},
        on_run=lambda record: sweep.append_record(results_csv, record),
    )
    runtime = time.perf_counter() - start
    sweep_dir = incomplete / "sweep"
    if not args.no_html:
        write_surface_html(
            plot_surface(result, "rho_inf"), sweep_dir / "surface_rho_infinity.html"
        )
        write_surface_html(
            plot_surface(result, "energy"), sweep_dir / "surface_energy.html"
        )
    sweep.save_sweep_summary(
        sweep_dir / "sweep_summary.json", result, runtime_seconds=runtime
    )
    metadata = artifacts.environment_metadata(argv)
    metadata.update(
        run_id=run_id,
        config_hash=config_hash,
        n_runs=len(result.runs),
        resumed_runs=len(completed),
        non_converged_points=result.non_converged_points,
        runtime_seconds=runtime,
    )
    artifacts.write_json(incomplete / "metadata.json", metadata)
    if final.exists():
        raise _Refused(f"refusing to overwrite existing {str(final)!r}")
    incomplete.rename(final)
    print(f"sweep directory: {final}")
    _print_tree(final)
    print(
        f"runs: {len(result.runs)} ({len(completed)} resumed) in {runtime:.1f} s; "
        f"non-converged grid points: {result.non_converged_points}"
    )
    print("synthetic simulation output; not empirical market data")
    return EXIT_OK


# --- configuration resolution ------------------------------------------------


def _config_defaults() -> dict[str, Any]:
    """Control-option defaults, read from :class:`SimulationConfig` itself."""
    return {
        f.name: f.default
        for f in dataclasses.fields(SimulationConfig)
        if f.name in CONTROL_FIELDS
    }


def _resolve_values(args: argparse.Namespace, *, sweep_mode: bool) -> dict[str, Any]:
    """Merge --config file values with explicit flags (flags win)."""
    values: dict[str, Any] = {}
    sais: dict[str, Any] = {}
    if args.config is not None:
        document = _load_config_file(args.config)
        config_part = document.get("config", document.get("base", document))
        if not isinstance(config_part, Mapping):
            raise _UsageError("--config", "file does not contain a configuration")
        values.update({k: v for k, v in config_part.items() if k != "sais_params"})
        if isinstance(config_part.get("sais_params"), Mapping):
            sais.update(config_part["sais_params"])
        if document.get("kind") == "sweep":
            values["__sweep__"] = document
        if sweep_mode:
            for name in SWEPT_FIELDS:
                values.pop(name, None)
    for field, flag in CONFIG_FLAGS.items():
        value = getattr(args, _dest(flag), None)
        if value is not None:
            values[field] = value
    for field, flag in SAIS_FLAGS.items():
        value = getattr(args, _dest(flag), None)
        if value is not None:
            sais[field] = value
    required = [f for f in CONFIG_FLAGS if f not in CONTROL_FIELDS + ("ba_m", "er_p")]
    if sweep_mode:
        required = [f for f in required if f not in SWEPT_FIELDS]
    for field in required:
        if values.get(field) is None:
            raise _UsageError(CONFIG_FLAGS[field], "required (give it or use --config)")
    for field, flag in SAIS_FLAGS.items():
        if sais.get(field) is None:
            raise _UsageError(flag, "required (give it or use --config)")
    values["sais_params"] = sais
    return values


def _build_config(values: Mapping[str, Any], *, sweep_mode: bool) -> SimulationConfig:
    data = {k: v for k, v in values.items() if k != "__sweep__"}
    try:
        return artifacts.config_from_dict(data)
    except (ValueError, TypeError) as exc:
        raise _usage_from(exc, sweep_mode=sweep_mode) from exc


def _build_grid(
    args: argparse.Namespace, document: Mapping[str, Any]
) -> sweep.SweepGrid:
    file_grid = document.get("grid") if isinstance(document, Mapping) else None
    try:
        if (
            file_grid is not None
            and args.alpha_range is None
            and args.r0_range is None
            and args.rho0_values is None
        ):
            return sweep.SweepGrid(
                tuple(file_grid["alpha"]),
                tuple(file_grid["r0"]),
                tuple(file_grid["rho0"]),
            )
        return sweep.make_grid(
            _range(args.alpha_range, sweep.DEFAULT_ALPHA_RANGE, "--alpha-range"),
            _range(args.r0_range, sweep.DEFAULT_R0_RANGE, "--r0-range"),
            args.rho0_values
            if args.rho0_values is not None
            else sweep.DEFAULT_RHO0_VALUES,
        )
    except (ValueError, TypeError) as exc:
        raise _usage_from(exc, sweep_mode=True) from exc


def _build_settings(
    args: argparse.Namespace, document: Mapping[str, Any]
) -> sweep.SteadyStateSettings:
    values = (
        dict(document.get("steady_state", {})) if isinstance(document, Mapping) else {}
    )
    if args.tol_rho is not None:
        values["tol_rho"] = args.tol_rho
    if args.tol_e is not None:
        values["tol_e_rel"] = args.tol_e
    try:
        return sweep.SteadyStateSettings(**values)
    except (ValueError, TypeError) as exc:
        raise _usage_from(exc, sweep_mode=True) from exc


def _range(
    given: list[float] | None, default: tuple[float, float, int], flag: str
) -> tuple[float, float, int]:
    if given is None:
        return default
    start, stop, count = given
    if not float(count).is_integer():
        raise _UsageError(flag, f"COUNT must be a whole number, got {count:g}")
    return start, stop, int(count)


def _load_config_file(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _UsageError(
            "--config", f"cannot read {str(path)!r}: {exc.strerror}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _UsageError(
            "--config", f"{str(path)!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise _UsageError("--config", "file must contain a JSON object")
    return document


def _usage_from(exc: Exception, *, sweep_mode: bool) -> _UsageError:
    """Name the flag whose field the validator's message starts with."""
    message = str(exc)
    flags = {**CONFIG_FLAGS, **SAIS_FLAGS}
    if sweep_mode:
        flags.update(SWEEP_FLAGS)
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", message):
        if token in flags:
            return _UsageError(flags[token], message)
    return _UsageError("--config" if sweep_mode else "configuration", message)


def _run_id(
    kind: artifacts.RunKind, config_hash: str, seed: object, name: str | None
) -> str:
    try:
        return artifacts.make_run_id(kind, config_hash, seed, name)
    except ValueError as exc:
        raise _UsageError("--run-name", str(exc)) from exc


# --- directories and output --------------------------------------------------


def _check_target(output_dir: Path, target: Path) -> None:
    try:
        artifacts.check_output_dir(output_dir)
    except NotADirectoryError as exc:
        raise _UsageError("--output-dir", str(exc)) from exc
    if target.exists():
        raise _Refused(
            f"refusing to overwrite existing {str(target)!r}; "
            "choose another --run-name or --output-dir"
        )


def _prepare_incomplete(
    incomplete: Path, definition: Mapping[str, Any], config_hash: str, resume: bool
) -> list[sweep.RunRecord]:
    """Create or reopen ``<run_id>.incomplete`` and return completed records."""
    results = incomplete / "sweep" / "sweep_results.csv"
    if incomplete.exists():
        if not resume:
            raise _Refused(
                f"an unfinished sweep exists at {str(incomplete)!r}; "
                "pass --resume to continue it"
            )
        saved = json.loads((incomplete / "config.json").read_text(encoding="utf-8"))
        if artifacts.digest(saved) != config_hash:
            raise _Refused(
                f"cannot resume {str(incomplete)!r}: its configuration hash differs"
            )
        records = sweep.read_records(results)
        sweep.rewrite_records(results, records)
        return records
    (incomplete / "sweep").mkdir(parents=True)
    artifacts.write_json(incomplete / "config.json", dict(definition))
    sweep.write_results_header(results)
    return []


def _print_tree(directory: Path) -> None:
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        print(
            f"  {path.relative_to(directory).as_posix()} "
            f"({path.stat().st_size:,} bytes)"
        )


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _first(*values: Any) -> Any:
    return next(v for v in values if v is not None)


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:g}" for v in values)


def _package_version() -> str:
    try:
        return importlib.metadata.version("financial-contagion-sim")
    except importlib.metadata.PackageNotFoundError:
        from financial_contagion_sim import __version__

        return __version__
