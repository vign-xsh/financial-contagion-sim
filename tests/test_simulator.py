"""Tests for the co-evolutionary simulation engine (``simulator``).

Spec §1, §3.3 and §5, step 2; Phase 3 decisions in DECISIONS.md.
"""

import dataclasses
import math

import networkx as nx
import numpy as np
import pytest

import financial_contagion_sim.simulator as simulator_module
from financial_contagion_sim.energy_calculator import (
    delta_total_energy,
    exact_structural_energy,
    spreading_energy,
    total_energy,
)
from financial_contagion_sim.graph_builder import (
    generate_erdos_renyi,
    graph_from_correlation,
)
from financial_contagion_sim.sais_engine import (
    SAISParams,
    make_initial_state,
    run_gillespie,
)
from financial_contagion_sim.simulator import (
    SCHEMA_VERSION,
    STREAM_NAMES,
    TRANSITION_NAMES,
    SimulationConfig,
    SimulationResult,
    _spawn_streams,
    run_simulation,
    simulate,
)

PARAMS = SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7)


def _cfg(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "n": 30,
        "topology": "erdos_renyi",
        "er_p": 0.2,
        "sais_params": PARAMS,
        "alpha": 0.5,
        "r0": 0.6,
        "initial_infected_fraction": 0.2,
        "edge_update_rate": 1.0,
        "t_max": 2.0,
        "seed": 7,
    }
    values.update(overrides)
    return SimulationConfig(**values)


# --- configuration ------------------------------------------------------------


def test_valid_configs() -> None:
    er = _cfg()
    assert (er.max_events, er.stop_on_sais_absorbed, er.record) == (None, False, "grid")
    assert (er.num_points, er.verify_every) == (501, None)
    _cfg(topology="barabasi_albert", er_p=None, ba_m=2)
    _cfg(edge_update_rate=0.0, t_max=0.0, r0=0.0, initial_infected_fraction=1.0)
    _cfg(seed=np.random.SeedSequence(3), record="events", num_points=2)


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cfg().n = 5


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"topology": "watts_strogatz"}, ValueError, "topology must be one of"),
        ({"topology": 1}, TypeError, "topology must be a str"),
        ({"er_p": None}, ValueError, "needs er_p and no ba_m"),
        ({"ba_m": 2}, ValueError, "needs er_p and no ba_m"),
        ({"topology": "barabasi_albert"}, ValueError, "needs ba_m and no er_p"),
        ({"topology": "barabasi_albert", "er_p": None}, ValueError, "needs ba_m"),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 30},
            ValueError,
            "1 <= ba_m < n",
        ),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 0},
            ValueError,
            "ba_m must be >= 1",
        ),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 2.0},
            TypeError,
            "ba_m must be an int",
        ),
        ({"er_p": 1.5}, ValueError, r"er_p must be in \[0, 1\]"),
        ({"n": 0}, ValueError, "n must be >= 1"),
        ({"n": 30.0}, TypeError, "n must be an int"),
        ({"n": True}, TypeError, "n must be an int"),
        ({"alpha": 1.5}, ValueError, r"alpha must be in \[0, 1\]"),
        ({"alpha": "0.5"}, TypeError, "alpha must be a real number"),
        ({"r0": math.nan}, ValueError, r"r0 must be in \[0, 1\]"),
        ({"initial_infected_fraction": 2.0}, ValueError, "initial_infected_fraction"),
        ({"edge_update_rate": -1.0}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": math.nan}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": math.inf}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": True}, TypeError, "edge_update_rate must be a real"),
        ({"t_max": -1.0}, ValueError, "t_max must be finite and >= 0"),
        ({"t_max": math.inf}, ValueError, "t_max must be finite and >= 0"),
        ({"t_max": math.nan}, ValueError, "t_max must be finite and >= 0"),
        ({"seed": -1}, ValueError, "seed must be a non-negative int"),
        ({"seed": 1.5}, TypeError, "seed must be a non-negative int or"),
        ({"seed": True}, TypeError, "seed must be a non-negative int or"),
        ({"seed": None}, TypeError, "seed must be a non-negative int or"),
        ({"max_events": -1}, ValueError, "max_events must be >= 0"),
        ({"max_events": 1.0}, TypeError, "max_events must be an int"),
        (
            {"stop_on_sais_absorbed": 1},
            TypeError,
            "stop_on_sais_absorbed must be a bool",
        ),
        ({"record": "snapshots"}, ValueError, "record must be one of"),
        ({"record": None}, TypeError, "record must be a str"),
        ({"num_points": 1}, ValueError, "num_points must be >= 2"),
        ({"num_points": True}, TypeError, "num_points must be an int"),
        ({"verify_every": 0}, ValueError, "verify_every must be >= 1"),
        ({"sais_params": (2.0, 0.5, 3.0, 0.7)}, TypeError, "sais_params must be"),
    ],
)
def test_config_rejects_invalid(
    overrides: dict[str, object], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        _cfg(**overrides)


def test_beta_a_equal_beta_is_rejected_by_phase2_params() -> None:
    with pytest.raises(ValueError, match="0 <= beta_a < beta"):
        SAISParams(beta=1.0, beta_a=1.0, kappa=0.0, delta=0.0)


# --- random streams -----------------------------------------------------------


def test_streams_follow_documented_spawn_order() -> None:
    assert STREAM_NAMES == (
        "graph",
        "initial_state",
        "clock",
        "sais_selection",
        "edge_updates",
    )
    streams = _spawn_streams(11)
    children = np.random.SeedSequence(11).spawn(5)
    for name, child in zip(STREAM_NAMES, children, strict=True):
        expected = np.random.default_rng(child).random(4)
        assert np.array_equal(streams[name].random(4), expected)


def test_seed_sequence_argument_is_not_consumed() -> None:
    seq = np.random.SeedSequence(11)
    first = _spawn_streams(seq)["clock"].random(3)
    second = _spawn_streams(seq)["clock"].random(3)
    assert np.array_equal(first, second)
    assert seq.n_children_spawned == 0
    assert np.array_equal(_spawn_streams(11)["clock"].random(3), first)


def test_schema_version_is_int() -> None:
    assert type(SCHEMA_VERSION) is int and SCHEMA_VERSION >= 1


# --- helpers for engine tests -------------------------------------------------

CODES = {"S": 1, "A": 0, "I": -1}
ALLOWED = set(TRANSITION_NAMES)


def _st(code: str) -> np.ndarray:
    return np.array([CODES[c] for c in code], dtype=np.int8)


def _graph(n: int, edges: list[tuple[int, int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def _graph_with_signs(res: SimulationResult, signs: np.ndarray) -> nx.Graph:
    n = len(res.initial_states)
    edges = [
        (int(u), int(v), int(s)) for (u, v), s in zip(res.edges, signs, strict=True)
    ]
    return _graph(n, edges)


def _counts(states: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.count_nonzero(states == s)) for s in (1, 0, -1))


def _check_invariants(res: SimulationResult, alpha: float, t_max: float) -> None:
    n, m = len(res.initial_states), len(res.edges)
    k = len(res.times)
    series = (
        res.S,
        res.A,
        res.I,
        res.r,
        res.E,
        res.E_delta,
        res.E_p,
        res.infected_fraction,
    )
    assert all(len(x) == k for x in series) and k >= 1
    assert res.times.dtype == np.float64 and res.S.dtype == np.int64
    assert np.all(n == res.S + res.A + res.I)
    assert np.all(res.S >= 0) and np.all(res.A >= 0) and np.all(res.I >= 0)
    assert np.array_equal(res.infected_fraction, res.I / n)
    if m:
        assert np.all((res.r >= 0) & (res.r <= 1))
    else:
        assert np.all(np.isnan(res.r))
    assert res.times[0] == 0.0 and np.all(np.diff(res.times) >= 0)
    assert res.times[-1] <= t_max and res.times[-1] == res.final_time
    assert res.n_events == sum(res.n_sais_events.values()) + res.n_proposed
    assert res.n_proposed == res.n_accepted + res.n_rejected
    assert 0 <= res.n_zero_accepted <= res.n_accepted
    for index, states, signs in (
        (0, res.initial_states, res.initial_signs),
        (-1, res.final_states, res.final_signs),
    ):
        g = _graph_with_signs(res, signs)
        assert (res.S[index], res.A[index], res.I[index]) == _counts(states)
        if m:
            assert res.r[index] == np.count_nonzero(signs == 1) / m
        assert res.E_delta[index] == exact_structural_energy(g)
        assert res.E_p[index] == spreading_energy(g, states)
        assert res.E[index] == pytest.approx(total_energy(g, states, alpha), abs=1e-12)


# --- invariants ---------------------------------------------------------------


@pytest.mark.parametrize("record", ["grid", "events"])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("topology", ["erdos_renyi", "barabasi_albert"])
def test_invariants_with_full_verification(
    record: str, seed: int, topology: str
) -> None:
    extra = {"er_p": 0.2} if topology == "erdos_renyi" else {"er_p": None, "ba_m": 2}
    cfg = _cfg(
        topology=topology,
        record=record,
        seed=seed,
        verify_every=1,
        t_max=3.0,
        num_points=50,
        **extra,
    )
    res = run_simulation(cfg)
    _check_invariants(res, cfg.alpha, cfg.t_max)
    assert res.termination_reason == "t_max" and res.n_events > 100
    if record == "events":
        assert np.all(np.diff(res.times) > 0)
        assert len(res.times) == res.n_events + 2  # t = 0, every event, t_max
    else:
        assert np.array_equal(res.times, np.linspace(0, 3.0, 50))
    assert res.schema_version == SCHEMA_VERSION and res.config == cfg
    assert set(res.versions) == {
        "financial_contagion_sim",
        "python",
        "numpy",
        "networkx",
    }


def test_verification_detects_wrong_pair_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = simulator_module.state_change_pair_delta
    monkeypatch.setattr(
        simulator_module, "state_change_pair_delta", lambda *a: real(*a) + 1
    )
    with pytest.raises(RuntimeError, match="maintained P"):
        run_simulation(_cfg(verify_every=1))


def test_verification_detects_wrong_triad_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = simulator_module.flip_energy_numerators

    def corrupted(*args: object) -> tuple[int, int]:
        d_s3, d_p = real(*args)
        return d_s3 + 2, d_p

    monkeypatch.setattr(simulator_module, "flip_energy_numerators", corrupted)
    with pytest.raises(RuntimeError, match="maintained S3"):
        run_simulation(_cfg(alpha=0.0, initial_infected_fraction=0.0, verify_every=1))


# --- initialization -----------------------------------------------------------


@pytest.mark.parametrize(
    ("fraction", "expected"), [(0.25, 2), (0.35, 4), (0.0, 0), (1.0, 10), (0.5, 5)]
)
def test_initial_infected_count_is_round_half_even(
    fraction: float, expected: int
) -> None:
    res = run_simulation(_cfg(n=10, initial_infected_fraction=fraction, t_max=0.0))
    assert expected == round(fraction * 10)
    assert _counts(res.initial_states) == (10 - expected, 0, expected)
    assert res.requested_initial_infected_fraction == fraction
    assert res.realized_initial_infected_fraction == expected / 10


def test_initial_graph_and_state_come_from_their_streams() -> None:
    cfg = _cfg(seed=21, t_max=0.0)
    res = run_simulation(cfg)
    children = np.random.SeedSequence(21).spawn(5)
    g = generate_erdos_renyi(30, 0.2, 0.6, np.random.default_rng(children[0]))
    edges = sorted(g.edges)
    assert res.edges.tolist() == [list(e) for e in edges]
    assert res.initial_signs.tolist() == [g.edges[e]["sign"] for e in edges]
    assert res.requested_r0 == 0.6 and res.realized_r0 == g.graph["r0_realized"]
    init = make_initial_state(30, 0.2, np.random.default_rng(children[1]))
    assert np.array_equal(res.initial_states, init.states)


def test_changing_dynamics_does_not_change_initial_draws() -> None:
    a = run_simulation(_cfg(seed=5))
    b = run_simulation(
        _cfg(
            seed=5,
            sais_params=SAISParams(9.0, 1.0, 0.1, 4.0),
            edge_update_rate=7.0,
            alpha=0.1,
        )
    )
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.initial_signs, b.initial_signs)
    assert np.array_equal(a.initial_states, b.initial_states)


# --- hand-derivable co-evolution ---------------------------------------------


def test_all_positive_triangle_rejects_every_flip() -> None:
    g = _graph(3, [(0, 1, 1), (1, 2, 1), (0, 2, 1)])
    for u, v in g.edges:
        assert delta_total_energy(g, _st("SSS"), u, v, 1.0, exact=True) == 2
    res = simulate(g, _st("SSS"), PARAMS, 1.0, 1.0, 50.0, seed=0, record="events")
    assert res.n_events == res.n_proposed > 50
    assert res.n_accepted == 0 and res.n_rejected == res.n_proposed
    assert np.all(res.r == 1.0) and np.all(res.E_delta == -1.0)


def test_unbalanced_triangle_accepts_only_the_first_flip() -> None:
    # In ++- every flip balances the triangle (++- -> +-- or +++), so dE = -2
    # for all three edges; once balanced, every flip is rejected (dE = +2).
    g = _graph(3, [(0, 1, 1), (1, 2, 1), (0, 2, -1)])
    for u, v in g.edges:
        assert delta_total_energy(g, _st("SSS"), u, v, 1.0, exact=True) == -2
    finals = set()
    for seed in range(30):
        res = simulate(
            g, _st("SSS"), PARAMS, 1.0, 1.0, 30.0, seed=seed, record="events"
        )
        assert res.n_accepted == 1 and res.n_zero_accepted == 0
        assert res.r[1] != res.r[0]  # the first event is the accepted flip
        assert int(np.prod(res.final_signs)) == 1 and res.E_delta[-1] == -1.0
        finals.add(float(res.r[-1]))
    assert finals == {1.0, 1 / 3}


def test_path_flips_are_ties_and_r_drifts_to_one_half() -> None:
    # alpha = 1 and no triangles: every dE is exactly 0, so each proposal is
    # accepted with probability 1/2 (binomial bound) and r random-walks.
    g = _graph(5, [(k, k + 1, 1) for k in range(4)])
    t_max = 10_000.0
    res = simulate(g, _st("SSSSS"), PARAMS, 1.0, 1.0, t_max, seed=3, record="events")
    n = res.n_proposed
    assert res.n_zero_accepted == res.n_accepted
    assert abs(res.n_accepted - n / 2) <= 5 * math.sqrt(n / 4)
    # Time-average of r over [T/2, T]. Each edge sign is a two-state chain
    # flipping at rate gamma/2, so its autocorrelation decays as exp(-t) and
    # the time average of one indicator has variance ~ 2 * 0.25 * 1 / T_w; four
    # independent edges divide that by 4, giving sd = sqrt(0.125 / T_w).
    start = t_max / 2
    t = np.clip(res.times, start, t_max)
    widths = np.diff(t)
    mean_r = float(np.sum(res.r[:-1] * widths) / (t_max - start))
    assert abs(mean_r - 0.5) <= 5 * math.sqrt(0.125 / (t_max - start))


# Star: centre I, leaves S, A, S over positive edges. Rates: S->I 2 x 2 = 4,
# S->A 3 x 2 = 6, A->I 0.5, I->S 0.7; R_sais = 11.2.
STAR = _graph(4, [(0, 1, 1), (0, 2, 1), (0, 3, 1)])
STAR_STATES = _st("ISAS")
STAR_SHARES = {"S->I": 4.0, "S->A": 6.0, "A->I": 0.5, "I->S": 0.7}


def test_first_event_frequencies_match_rate_shares() -> None:
    runs = 6000
    counts = dict.fromkeys(TRANSITION_NAMES, 0)
    for seed in range(runs):
        res = simulate(
            STAR, STAR_STATES, PARAMS, 0.5, 0.0, 1e6, seed=seed, max_events=1
        )
        assert res.n_events == 1 and res.n_proposed == 0
        for key, value in res.n_sais_events.items():
            counts[key] += value
    total = sum(STAR_SHARES.values())
    for key, rate in STAR_SHARES.items():
        p = rate / total
        assert abs(counts[key] - runs * p) <= 5 * math.sqrt(runs * p * (1 - p))


def test_first_event_channel_probability() -> None:
    runs, gamma = 6000, 11.2 / 3  # R_net = gamma * m = 11.2 = R_sais
    network = sum(
        simulate(
            STAR, STAR_STATES, PARAMS, 0.5, gamma, 1e6, seed=s, max_events=1
        ).n_proposed
        for s in range(runs)
    )
    assert abs(network - runs / 2) <= 5 * math.sqrt(runs / 4)


def test_gamma_zero_is_pure_sais_and_matches_run_gillespie() -> None:
    g = generate_erdos_renyi(20, 0.2, 0.7, rng=4)
    states = make_initial_state(20, 0.3, rng=4).states
    runs = 400
    sim_final, gil_final = [], []
    for seed in range(runs):
        res = simulate(g, states, PARAMS, 0.5, 0.0, 1.0, seed=seed, record="events")
        assert res.n_proposed == 0 and np.array_equal(
            res.final_signs, res.initial_signs
        )
        assert set(k for k, v in res.n_sais_events.items() if v) <= ALLOWED
        sim_final.append(int(np.count_nonzero(res.final_states == -1)))
        gil = run_gillespie(g, states, PARAMS, 1.0, rng=10_000 + seed)
        gil_final.append(int(np.count_nonzero(gil.states == -1)))
    a, b = np.array(sim_final, float), np.array(gil_final, float)
    z = abs(a.mean() - b.mean()) / math.sqrt(
        a.var(ddof=1) / runs + b.var(ddof=1) / runs
    )
    assert z <= 5


def test_sais_rates_follow_accepted_flips() -> None:
    # alpha = 0: flipping a positive S-I edge to negative lowers E_p, so such
    # flips are accepted and the S leaves lose their infected neighbour.
    params = SAISParams(beta=0.5, beta_a=0.1, kappa=0.2, delta=0.0)
    g = _graph(6, [(0, k, 1) for k in range(1, 6)])
    res = simulate(
        g, _st("ISSSSS"), params, 0.0, 5.0, 5.0, seed=1, verify_every=1, record="events"
    )
    assert res.n_accepted > 0
    _check_invariants(res, 0.0, 5.0)


# --- termination --------------------------------------------------------------


def test_absorbed_without_infection_or_flips_fills_grid() -> None:
    g = generate_erdos_renyi(15, 0.3, 0.5, rng=2)
    res = simulate(
        g, np.ones(15, dtype=np.int8), PARAMS, 0.5, 0.0, 5.0, seed=0, num_points=11
    )
    assert (res.termination_reason, res.final_time, res.n_events) == (
        "absorbed",
        5.0,
        0,
    )
    assert np.array_equal(res.times, np.linspace(0, 5.0, 11))
    assert np.all(res.S == 15)


def test_all_infected_without_recovery_is_absorbed() -> None:
    params = SAISParams(beta=1.0, beta_a=0.0, kappa=0.0, delta=0.0)
    g = _graph(3, [(0, 1, 1), (1, 2, 1)])
    res = simulate(g, _st("III"), params, 0.5, 0.0, 2.0, seed=0)
    assert res.termination_reason == "absorbed" and res.n_events == 0


def test_flips_continue_after_infection_is_gone() -> None:
    g = generate_erdos_renyi(15, 0.3, 0.5, rng=2)
    res = simulate(g, np.ones(15, dtype=np.int8), PARAMS, 0.5, 1.0, 3.0, seed=0)
    assert res.termination_reason == "t_max" and res.n_proposed > 0


def test_stop_on_sais_absorbed_only_when_enabled() -> None:
    g = generate_erdos_renyi(15, 0.3, 0.5, rng=2)
    states = np.ones(15, dtype=np.int8)
    on = simulate(g, states, PARAMS, 0.5, 1.0, 3.0, seed=0, stop_on_sais_absorbed=True)
    assert (on.termination_reason, on.final_time, on.n_events) == (
        "sais_absorbed",
        0.0,
        0,
    )
    assert len(on.times) == 1
    off = simulate(g, states, PARAMS, 0.5, 1.0, 3.0, seed=0)
    assert off.termination_reason == "t_max"


@pytest.mark.parametrize("record", ["grid", "events"])
def test_max_events_truncates_records(record: str) -> None:
    res = run_simulation(_cfg(max_events=25, t_max=1e6, record=record, num_points=1000))
    assert res.termination_reason == "max_events" and res.n_events == 25
    assert res.final_time < 1e6 and res.times[-1] == res.final_time
    if record == "events":
        assert len(res.times) == 26
    else:
        assert len(res.times) < 1000
    _check_invariants(res, 0.5, 1e6)


def test_event_after_t_max_is_not_applied() -> None:
    res = run_simulation(_cfg(t_max=1e-12, record="events"))
    assert (res.termination_reason, res.n_events) == ("t_max", 0)
    assert np.array_equal(res.final_states, res.initial_states)
    assert res.times.tolist() == [0.0, 1e-12]


def test_t_max_zero() -> None:
    res = run_simulation(_cfg(t_max=0.0, record="events"))
    assert (res.termination_reason, res.n_events, res.times.tolist()) == (
        "t_max",
        0,
        [0.0],
    )


def test_graph_without_edges_has_nan_r() -> None:
    res = simulate(_graph(4, []), _st("SISA"), PARAMS, 0.5, 2.0, 3.0, seed=0)
    assert res.realized_r0 is None and np.all(np.isnan(res.r)) and res.n_proposed == 0
    _check_invariants(res, 0.5, 3.0)


# --- reproducibility and streams ----------------------------------------------


def _assert_same(a: SimulationResult, b: SimulationResult) -> None:
    for f in dataclasses.fields(SimulationResult):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True), f.name
        elif f.name not in ("config", "seed_entropy"):
            assert x == y, f.name


def test_same_seed_identical_and_different_seed_differs() -> None:
    _assert_same(run_simulation(_cfg(seed=8)), run_simulation(_cfg(seed=8)))
    a, b = run_simulation(_cfg(seed=8)), run_simulation(_cfg(seed=9))
    assert not np.array_equal(a.E, b.E)


def test_int_seed_and_seed_sequence_agree() -> None:
    _assert_same(
        run_simulation(_cfg(seed=8)),
        run_simulation(_cfg(seed=np.random.SeedSequence(8))),
    )


def test_grid_and_event_records_describe_the_same_trajectory() -> None:
    grid = run_simulation(_cfg(seed=12, record="grid", num_points=77))
    events = run_simulation(_cfg(seed=12, record="events"))
    assert np.array_equal(grid.final_states, events.final_states)
    assert np.array_equal(grid.final_signs, events.final_signs)
    assert grid.n_sais_events == events.n_sais_events
    assert (grid.n_events, grid.n_accepted) == (events.n_events, events.n_accepted)
    index = np.searchsorted(events.times, grid.times, side="right") - 1
    for name in ("S", "A", "I", "r", "E", "E_delta", "E_p"):
        assert np.array_equal(getattr(grid, name), getattr(events, name)[index]), name


# --- safety -------------------------------------------------------------------


def test_simulate_does_not_mutate_caller_data() -> None:
    g = generate_erdos_renyi(20, 0.3, 0.5, rng=6)
    signs = sorted(g.edges(data="sign"))
    states = make_initial_state(20, 0.3, rng=6).states
    original = states.copy()
    res = simulate(g, states, PARAMS, 0.5, 2.0, 3.0, seed=0)
    assert res.n_accepted > 0
    assert sorted(g.edges(data="sign")) == signs
    assert np.array_equal(states, original)


def test_component_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(simulator_module, "select_event", boom)
    with pytest.raises(RuntimeError, match="boom"):
        run_simulation(_cfg())


def test_simulate_on_correlation_graph() -> None:
    corr = np.array([[1.0, 0.8, -0.6], [0.8, 1.0, 0.7], [-0.6, 0.7, 1.0]])
    g = graph_from_correlation(corr, tau=0.5)
    res = simulate(g, _st("ISS"), PARAMS, 0.5, 1.0, 2.0, seed=0, verify_every=1)
    assert res.requested_r0 is None and res.config is None
    _check_invariants(res, 0.5, 2.0)


def test_simulate_validates_inputs() -> None:
    g = _graph(2, [(0, 1, 1)])
    with pytest.raises(TypeError, match="dtype int8"):
        simulate(g, np.array([1, -1]), PARAMS, 0.5, 1.0, 1.0, seed=0)
    with pytest.raises(ValueError, match="edge_update_rate"):
        simulate(g, _st("SI"), PARAMS, 0.5, -1.0, 1.0, seed=0)
    with pytest.raises(TypeError, match="undirected simple"):
        simulate(nx.DiGraph(), _st(""), PARAMS, 0.5, 1.0, 1.0, seed=0)
    with pytest.raises(TypeError, match="SimulationConfig"):
        run_simulation({"n": 3})


def test_exact_tie_counting_uses_fractions() -> None:
    res = simulate(
        _graph(3, [(0, 1, 1), (1, 2, 1)]), _st("SSS"), PARAMS, 1.0, 1.0, 20.0, seed=0
    )
    assert res.n_zero_accepted == res.n_accepted > 0


# --- gaps found by mutation testing -------------------------------------------


def test_acceptance_uses_full_energy() -> None:
    # alpha = 0 and the centre stays infected (delta = 0) with alert leaves that
    # cannot be infected (beta_a = 0): every edge is I-incident, so positive
    # edges flip to negative with dE_p < 0 (accepted) and never flip back
    # (dE_p > 0). An E_delta-only rule would see dE = 0 ties instead.
    params = SAISParams(beta=0.5, beta_a=0.0, kappa=0.1, delta=0.0)
    g = _graph(6, [(0, k, 1) for k in range(1, 6)])
    res = simulate(g, _st("IAAAAA"), params, 0.0, 2.0, 50.0, seed=4)
    assert res.n_accepted == 5 and res.n_zero_accepted == 0
    assert np.all(res.final_signs == -1)


def test_edge_proposals_draw_only_from_the_edge_stream() -> None:
    # alpha = 1 on a path: every flip is a tie, so each proposal draws one edge
    # index and one tie-break from the edge_updates stream, in that order.
    # The whole r(t) sequence (one record per proposal) is replayed, so a
    # wrong stream cannot match by chance.
    g = _graph(5, [(k, k + 1, 1) for k in range(4)])
    for seed in range(5):
        res = simulate(
            g, _st("SSSSS"), PARAMS, 1.0, 1.0, 40.0, seed=seed, record="events"
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[4])
        signs = res.initial_signs.astype(int)
        expected_r = []
        for _ in range(res.n_proposed):
            index = int(rng.integers(4))
            if rng.random() < 0.5:
                signs[index] = -signs[index]
            expected_r.append(np.count_nonzero(signs == 1) / 4)
        assert res.n_proposed > 50
        assert res.r[1 : 1 + res.n_proposed].tolist() == expected_r
        assert signs.tolist() == res.final_signs.tolist()


class _RowStub:
    def row(self, time: float, m: int) -> tuple[float, ...]:
        return (time, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)


def test_grid_is_right_continuous_at_event_times() -> None:
    recorder = simulator_module._Recorder("grid", 1.0, 3)  # grid 0, 0.5, 1
    stub = _RowStub()
    recorder.before_event(0.5, stub, 1)
    assert [row[0] for row in recorder.rows] == [0.0]  # 0.5 waits for the event
    recorder.finish(0.5, stub, 1)
    assert [row[0] for row in recorder.rows] == [0.0, 0.5]
