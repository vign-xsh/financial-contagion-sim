"""System-level mathematical invariants and reproducibility (Phase 5 Part B).

Consequence checks (B2) follow directly from the approved rules (spec §3.1-3.3
and the Phase 2-3 decisions); they are not new assumptions. Integer quantities
are compared exactly; floats use rtol = atol = 1e-12 (B3); statistical checks
use fixed seeds and 5-sigma bounds.
"""

import hashlib
import itertools
import math
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.energy_calculator import (
    energy_from_sums,
    exact_structural_energy,
    flip_energy_numerators,
    spreading_energy,
    spreading_pair_sum,
    structural_triad_sum,
    total_energy,
)
from financial_contagion_sim.graph_builder import (
    generate_barabasi_albert,
    generate_erdos_renyi,
)
from financial_contagion_sim.sais_engine import SAISParams, make_initial_state
from financial_contagion_sim.simulator import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
    simulate,
)

TOL = {"rel": 1e-12, "abs": 1e-12}
PARAMS = SAISParams(beta=1.0, beta_a=0.2, kappa=0.3, delta=0.7)


def _graph(n: int, edges: list[tuple[int, int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def _states(codes: str) -> np.ndarray:
    return np.array([{"S": 1, "A": 0, "I": -1}[c] for c in codes], dtype=np.int8)


def _graph_at(result: SimulationResult, signs: np.ndarray) -> nx.Graph:
    edges = [
        (int(u), int(v), int(s)) for (u, v), s in zip(result.edges, signs, strict=True)
    ]
    return _graph(len(result.initial_states), edges)


def _star(n: int, sign: int) -> nx.Graph:
    return _graph(n, [(0, k, sign) for k in range(1, n)])


# --- B2 consequence checks -------------------------------------------------------


def test_no_attenuated_default_keeps_alert_nodes_alert() -> None:
    params = SAISParams(beta=2.0, beta_a=0.0, kappa=3.0, delta=0.5)
    states = _states("IAAASSSS")
    g = _star(8, 1)
    for seed in range(20):
        res = simulate(g, states, params, 0.5, 0.0, 20.0, seed=seed, record="events")
        assert res.n_sais_events["A->I"] == 0
        assert np.all(res.final_states[states == 0] == 0)  # initial A nodes stay A
        assert np.all(np.diff(res.A) >= 0)  # A is never left


def test_zero_kappa_means_no_hedging() -> None:
    params = SAISParams(beta=2.0, beta_a=0.5, kappa=0.0, delta=0.5)
    g = generate_erdos_renyi(25, 0.3, 0.8, rng=1)
    for seed in range(10):
        states = make_initial_state(25, 0.3, rng=seed).states
        res = simulate(g, states, params, 0.5, 1.0, 10.0, seed=seed, record="events")
        assert res.n_sais_events["S->A"] == 0 and np.all(res.A == 0)


def test_zero_delta_means_infection_never_decreases() -> None:
    params = SAISParams(beta=1.0, beta_a=0.5, kappa=1.0, delta=0.0)
    g = generate_erdos_renyi(25, 0.3, 0.8, rng=2)
    for seed in range(10):
        states = make_initial_state(25, 0.2, rng=seed).states
        res = simulate(g, states, params, 0.5, 1.0, 10.0, seed=seed, record="events")
        assert res.n_sais_events["I->S"] == 0 and np.all(np.diff(res.I) >= 0)


def test_all_negative_edges_transmit_nothing() -> None:
    g = generate_erdos_renyi(25, 0.4, 0.0, rng=3)
    assert all(s == -1 for _, _, s in g.edges(data="sign"))
    for seed in range(10):
        states = make_initial_state(25, 0.4, rng=seed).states
        res = simulate(g, states, PARAMS, 0.5, 0.0, 20.0, seed=seed, record="events")
        for key in ("S->I", "A->I", "S->A"):
            assert res.n_sais_events[key] == 0
        assert np.all(np.diff(res.I) <= 0)
        assert res.termination_reason == "absorbed" and res.I[-1] == 0


def test_gamma_zero_never_touches_signs() -> None:
    for seed in range(10):
        res = run_simulation(_config(seed=seed, edge_update_rate=0.0, record="events"))
        assert res.n_proposed == 0
        assert np.array_equal(res.final_signs, res.initial_signs)
        assert np.all(res.r == res.r[0])


def test_no_infection_and_no_flips_is_absorbed_immediately() -> None:
    res = run_simulation(_config(initial_infected_fraction=0.0, edge_update_rate=0.0))
    assert (res.termination_reason, res.n_events) == ("absorbed", 0)
    assert np.all(res.I == 0) and res.final_time == res.config.t_max


@pytest.mark.parametrize("n", [4, 6, 9])
def test_complete_positive_graph_rejects_every_flip(n: int) -> None:
    g = _graph(n, [(u, v, 1) for u, v in itertools.combinations(range(n), 2)])
    res = simulate(g, np.ones(n, dtype=np.int8), PARAMS, 1.0, 2.0, 30.0, seed=n)
    assert res.n_proposed > 50 and res.n_accepted == 0
    assert np.all(res.r == 1.0) and np.all(res.E_delta == -1.0)


def _replay_signs(g: nx.Graph, seed: int, proposals: int) -> list[int]:
    """Sign-only acceptance (alpha = 1) replayed from the edge_updates stream."""
    h = g.copy()
    n = h.number_of_nodes()
    edges = sorted((min(u, v), max(u, v)) for u, v in h.edges)
    calm = np.ones(n, dtype=np.int8)  # node states are irrelevant at alpha = 1
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[4])
    for _ in range(proposals):
        u, v = edges[int(rng.integers(len(edges)))]
        d_s3, _ = flip_energy_numerators(h, calm, u, v)
        delta = energy_from_sums(d_s3, 0, n, 1.0, exact=True)
        if delta < 0 or (delta == 0 and rng.random() < 0.5):
            h.edges[u, v]["sign"] = -h.edges[u, v]["sign"]
    return [h.edges[e]["sign"] for e in edges]


def test_alpha_one_topology_ignores_node_states() -> None:
    g = generate_erdos_renyi(18, 0.4, 0.5, rng=8)
    states = make_initial_state(18, 0.4, rng=8).states
    fast = SAISParams(beta=4.0, beta_a=1.0, kappa=2.0, delta=3.0)
    slow = SAISParams(beta=0.2, beta_a=0.0, kappa=0.0, delta=0.1)
    runs = [
        simulate(g, states, p, 1.0, 1.0, 15.0, seed=21, record="events")
        for p in (fast, slow)
    ]
    assert runs[0].n_proposed != runs[1].n_proposed  # the SAIS rates really differ
    for res in runs:
        assert np.array_equal(res.E, res.E_delta)  # E = E_delta exactly at alpha = 1
        assert res.final_signs.tolist() == _replay_signs(g, 21, res.n_proposed)


def test_alpha_zero_without_infection_makes_every_flip_a_tie() -> None:
    g = generate_barabasi_albert(40, 2, 0.5, rng=4)
    res = simulate(g, np.ones(40, dtype=np.int8), PARAMS, 0.0, 1.0, 300.0, seed=4)
    n = res.n_proposed
    assert n > 10_000 and res.n_zero_accepted == res.n_accepted
    assert abs(res.n_accepted - n / 2) <= 5 * math.sqrt(n / 4)


# --- recorded series vs independent recomputation --------------------------------


def _config(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "n": 25,
        "topology": "erdos_renyi",
        "er_p": 0.3,
        "sais_params": PARAMS,
        "alpha": 0.6,
        "r0": 0.6,
        "initial_infected_fraction": 0.3,
        "edge_update_rate": 1.0,
        "t_max": 4.0,
        "seed": 13,
    }
    values.update(overrides)
    return SimulationConfig(**values)


@pytest.mark.parametrize("topology", ["erdos_renyi", "barabasi_albert"])
def test_records_match_exact_recomputation_at_sampled_events(topology: str) -> None:
    extra = {} if topology == "erdos_renyi" else {"er_p": None, "ba_m": 3}
    full = run_simulation(_config(topology=topology, record="events", **extra))
    assert full.n_events > 200
    for k in (1, 7, 50, full.n_events // 2, full.n_events):
        prefix = run_simulation(
            _config(topology=topology, record="events", max_events=k, **extra)
        )
        assert prefix.n_events == k
        g = _graph_at(prefix, prefix.final_signs)
        states = prefix.final_states
        counts = tuple(int(np.count_nonzero(states == s)) for s in (1, 0, -1))
        assert (full.S[k], full.A[k], full.I[k]) == counts
        assert full.S[k] + full.A[k] + full.I[k] == 25
        assert full.r[k] == np.count_nonzero(prefix.final_signs == 1) / len(full.edges)
        assert full.E_delta[k] == exact_structural_energy(g)
        assert full.E_p[k] == spreading_energy(g, states)
        assert full.E[k] == pytest.approx(total_energy(g, states, 0.6), **TOL)


@pytest.mark.parametrize("topology", ["erdos_renyi", "barabasi_albert"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("gamma", [0.5, 3.0])
def test_maintained_integers_equal_recomputation_every_event(
    topology: str, alpha: float, gamma: float
) -> None:
    extra = {} if topology == "erdos_renyi" else {"er_p": None, "ba_m": 2}
    res = run_simulation(
        _config(
            topology=topology,
            alpha=alpha,
            edge_update_rate=gamma,
            verify_every=1,
            record="events",
            **extra,
        )
    )
    assert res.n_events > 50
    final = _graph_at(res, res.final_signs)
    n = len(res.initial_states)
    s3, p = structural_triad_sum(final), spreading_pair_sum(final, res.final_states)
    assert res.E_delta[-1] == energy_from_sums(s3, p, n, 1.0)
    assert res.E_p[-1] == energy_from_sums(s3, p, n, 0.0)


@pytest.mark.parametrize("seed", range(40))
def test_local_flip_change_on_scale_free_and_dense_graphs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    if seed % 2:
        g = generate_barabasi_albert(
            int(rng.integers(5, 30)), 2, float(rng.uniform()), rng=seed
        )
    else:
        g = generate_erdos_renyi(
            int(rng.integers(4, 12)), 0.9, float(rng.uniform()), rng=seed
        )
    n = g.number_of_nodes()
    states = rng.choice(np.array([1, 0, -1], dtype=np.int8), size=n)
    for u, v in sorted(g.edges)[:10]:
        h = g.copy()
        h.edges[u, v]["sign"] = -h.edges[u, v]["sign"]
        d_s3, d_p = flip_energy_numerators(g, states, u, v)
        assert d_s3 == structural_triad_sum(h) - structural_triad_sum(g)
        assert d_p == spreading_pair_sum(h, states) - spreading_pair_sum(g, states)


# --- B5 reproducibility --------------------------------------------------------------

_DIGEST_SCRIPT = """
import hashlib, numpy as np
from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import SimulationConfig, run_simulation
cfg = SimulationConfig(n=25, topology="erdos_renyi", er_p=0.3,
    sais_params=SAISParams(1.0, 0.2, 0.3, 0.7), alpha=0.6, r0=0.6,
    initial_infected_fraction=0.3, edge_update_rate=1.0, t_max=4.0, seed=13)
res = run_simulation(cfg)
h = hashlib.sha256()
arrays = (res.times, res.S, res.A, res.I, res.r, res.E)
for a in (*arrays, res.final_states, res.final_signs):
    h.update(np.ascontiguousarray(a).tobytes())
print(h.hexdigest())
"""


def _digest(result: SimulationResult) -> str:
    h = hashlib.sha256()
    for array in (
        result.times,
        result.S,
        result.A,
        result.I,
        result.r,
        result.E,
        result.final_states,
        result.final_signs,
    ):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def test_same_config_same_result_in_one_process() -> None:
    assert _digest(run_simulation(_config())) == _digest(run_simulation(_config()))


def test_same_result_across_processes_and_hash_seeds() -> None:
    digests = set()
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", _DIGEST_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        digests.add(proc.stdout.strip())
    assert digests == {_digest(run_simulation(_config()))}
