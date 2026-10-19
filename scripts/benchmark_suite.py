"""Phase 5 performance measurement (A2-A4). A measurement, not a test.

Reports numbers, not verdicts: there are no performance targets in the spec.

- Timing: per tier, 1 warm-up run plus 5 timed runs with
  ``time.perf_counter``; median and range are reported.
- Memory: measured in SEPARATE runs (tracemalloc slows execution, so it would
  distort the timings). Python-level peak via ``tracemalloc`` in-process;
  peak process memory via a fresh subprocess per tier (Windows: peak working
  set; elsewhere: ``ru_maxrss``), labelled by platform.
- Dense exact-energy engine: time and tracemalloc peak for increasing n.
- Profiling: ``cProfile`` of one medium run and of the reduced sweep.

Refuses to run on battery power (Phase 5 A3), because throttling changes the
results several-fold. Writes results to ``outputs/benchmarks/<UTC time>/``
(git-ignored). Usage::

    python scripts/benchmark_suite.py
"""

import cProfile
import ctypes
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np

from financial_contagion_sim import __version__
from financial_contagion_sim.energy_calculator import exact_structural_energy
from financial_contagion_sim.graph_builder import generate_erdos_renyi
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation
from financial_contagion_sim.sweep import SweepGrid, run_sweep

SEED = 0
REPEATS = 5
PARAMS = SAISParams(beta=1.0, beta_a=0.2, kappa=0.1, delta=1.0)
COMMON = {
    "sais_params": PARAMS,
    "alpha": 0.5,
    "r0": 0.6,
    "initial_infected_fraction": 0.1,
    "edge_update_rate": 1.0,
    "seed": SEED,
}
TIERS = {
    "small ER n=100 p=0.1 t_max=50": {
        "n": 100,
        "topology": "erdos_renyi",
        "er_p": 0.1,
        "t_max": 50.0,
    },
    "medium BA n=1000 m=3 t_max=20": {
        "n": 1000,
        "topology": "barabasi_albert",
        "ba_m": 3,
        "t_max": 20.0,
    },
    "medium ER n=1000 p=0.006 t_max=20": {
        "n": 1000,
        "topology": "erdos_renyi",
        "er_p": 0.006,
        "t_max": 20.0,
    },
    "larger BA n=5000 m=3 t_max=5": {
        "n": 5000,
        "topology": "barabasi_albert",
        "ba_m": 3,
        "t_max": 5.0,
    },
    "larger ER n=5000 p=0.0012 t_max=5": {
        "n": 5000,
        "topology": "erdos_renyi",
        "er_p": 0.0012,
        "t_max": 5.0,
    },
}
SWEEP_BASE = {"n": 100, "topology": "erdos_renyi", "er_p": 0.1, "t_max": 30.0}
SWEEP_GRID = SweepGrid(alpha=(0.0, 0.5, 1.0), r0=(0.0, 0.5, 1.0), rho0=(0.1,))
SWEEP_REPS = 3
DENSE_SIZES = (250, 500, 1000, 2000)


def power_state() -> str:
    """'ac', 'battery' or 'unknown'."""
    if sys.platform != "win32":
        return "unknown"

    class Status(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("Reserved1", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return "unknown"
    return {0: "battery", 1: "ac"}.get(status.ACLineStatus, "unknown")


def config(**fields: object) -> SimulationConfig:
    return SimulationConfig(**COMMON, **fields)  # type: ignore[arg-type]


def timed(fn) -> dict[str, object]:  # type: ignore[no-untyped-def]
    fn()  # warm-up
    seconds, events = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        seconds.append(time.perf_counter() - start)
        events = getattr(out, "n_events", None)
    median = statistics.median(seconds)
    result: dict[str, object] = {
        "median_s": median,
        "min_s": min(seconds),
        "max_s": max(seconds),
        "repeats": REPEATS,
    }
    if events is not None:
        result.update(events=events, events_per_s_median=events / median)
    return result


def python_peak(fn) -> int:  # type: ignore[no-untyped-def]
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


PEAK_SNIPPET = """
import sys, json, ctypes
from financial_contagion_sim.artifacts import config_from_dict
from financial_contagion_sim.simulator import run_simulation
run_simulation(config_from_dict(json.loads(sys.argv[1])))
if sys.platform == "win32":
    size_t = ctypes.c_size_t
    names = ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]
    class C(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)]
        _fields_ += [(name, size_t) for name in names]
    c = C()
    c.cb = ctypes.sizeof(C)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    void_p = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [void_p, void_p, ctypes.c_ulong]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("K32GetProcessMemoryInfo failed")
    print("windows-peak-working-set", c.PeakWorkingSetSize)
else:
    import resource
    print("ru_maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def process_peak(cfg: SimulationConfig) -> str:
    from financial_contagion_sim.artifacts import config_to_dict

    out = subprocess.run(
        [sys.executable, "-c", PEAK_SNIPPET, json.dumps(config_to_dict(cfg))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return f"{out[0]}={int(out[1]):,}" + (
        " bytes" if out[0].startswith("windows") else " (platform units)"
    )


def profile(fn, limit: int = 12) -> str:  # type: ignore[no-untyped-def]
    profiler = cProfile.Profile()
    profiler.runcall(fn)
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(limit)
    return text.getvalue()


def main() -> int:
    power = power_state()
    if power == "battery":
        print(
            "refusing to benchmark on battery power (Phase 5 A3); "
            "connect AC power and rerun"
        )
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs") / "benchmarks" / stamp
    out_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "synthetic": True,
        "created_utc": stamp,
        "power": power,
        "seed": SEED,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "financial_contagion_sim": __version__,
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "params": {
            "beta": 1.0,
            "beta_a": 0.2,
            "kappa": 0.1,
            "delta": 1.0,
            "gamma": 1.0,
            "alpha": 0.5,
            "r0": 0.6,
            "rho0": 0.1,
        },
    }
    timings: dict[str, object] = {}
    for label, fields in TIERS.items():
        cfg = config(**fields)
        timings[label] = timed(lambda cfg=cfg: run_simulation(cfg))
        print(f"{label}: {timings[label]}")
    sweep_cfg = config(**SWEEP_BASE)
    timings[f"reduced sweep 3x3x1 x{SWEEP_REPS} reps (n=100, t_max=30)"] = timed(
        lambda: run_sweep(sweep_cfg, SWEEP_GRID, SWEEP_REPS)
    )
    report["timing"] = timings

    memory: dict[str, object] = {}
    for label, fields in TIERS.items():
        cfg = config(**fields)
        memory[label] = {
            "tracemalloc_peak_bytes": python_peak(lambda cfg=cfg: run_simulation(cfg)),
            "process_peak": process_peak(cfg),
        }
        print(f"memory {label}: {memory[label]}")
    report["memory"] = memory

    dense: dict[str, object] = {}
    for n in DENSE_SIZES:
        g = generate_erdos_renyi(n, 0.05, 0.5, rng=SEED)
        start = time.perf_counter()
        exact_structural_energy(g)
        seconds = time.perf_counter() - start
        peak = python_peak(lambda g=g: exact_structural_energy(g))
        dense[str(n)] = {
            "seconds": seconds,
            "tracemalloc_peak_bytes": peak,
            "adjacency_bytes": n * n * 8,
        }
        print(f"dense exact engine n={n}: {dense[str(n)]}")
    report["dense_exact_energy"] = dense

    profiles = {
        "medium BA run": profile(
            lambda: run_simulation(config(**TIERS["medium BA n=1000 m=3 t_max=20"]))
        ),
        "reduced sweep": profile(lambda: run_sweep(sweep_cfg, SWEEP_GRID, SWEEP_REPS)),
        "graph generation (BA n=5000)": profile(
            lambda: run_simulation(
                config(**TIERS["larger BA n=5000 m=3 t_max=5"], max_events=0)
            )
        ),
    }
    for name, text in profiles.items():
        (out_dir / f"profile_{name.split()[0]}_{name.split()[1]}.txt").write_text(
            text, encoding="utf-8"
        )
    (out_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
