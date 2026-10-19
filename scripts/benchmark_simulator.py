"""Measure simulator throughput (events per second). A measurement, not a test.

Runs a few fixed configurations with a fixed seed and prints the event count,
wall time and events per second, plus the machine and package versions. It
makes no assertions and no complexity claims; numbers vary between machines
and runs.

Usage::

    python scripts/benchmark_simulator.py
"""

import platform
import time

import networkx as nx
import numpy as np

from financial_contagion_sim import __version__
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation

SEED = 0
MAX_EVENTS = 50_000
PARAMS = SAISParams(beta=0.5, beta_a=0.1, kappa=0.3, delta=1.0)

CASES = [
    ("ER n=200 p=0.05, gamma=1", {"n": 200, "topology": "erdos_renyi", "er_p": 0.05}),
    ("BA n=1000 m=3, gamma=1", {"n": 1000, "topology": "barabasi_albert", "ba_m": 3}),
    ("BA n=5000 m=3, gamma=1", {"n": 5000, "topology": "barabasi_albert", "ba_m": 3}),
]


def main() -> None:
    print(f"machine: {platform.platform()} | {platform.processor()}")
    print(
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"networkx {nx.__version__}, financial_contagion_sim {__version__}"
    )
    print(f"seed={SEED}, max_events={MAX_EVENTS}, params={PARAMS}")
    for label, topology in CASES:
        config = SimulationConfig(
            **topology,
            sais_params=PARAMS,
            alpha=0.5,
            r0=0.5,
            initial_infected_fraction=0.1,
            edge_update_rate=1.0,
            t_max=1e9,
            seed=SEED,
            max_events=MAX_EVENTS,
        )
        start = time.perf_counter()
        result = run_simulation(config)
        elapsed = time.perf_counter() - start
        sais = sum(result.n_sais_events.values())
        print(
            f"{label}: {result.n_events} events ({sais} SAIS, "
            f"{result.n_proposed} proposals, {result.n_accepted} accepted) in "
            f"{elapsed:.2f} s = {result.n_events / elapsed:,.0f} events/s"
        )


if __name__ == "__main__":
    main()
