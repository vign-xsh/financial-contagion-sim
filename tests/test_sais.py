"""Verify CTMC rate/probability bounds and state conservation.

Spec §4, for ``sais_engine``; the rates themselves are spec §3.1.
"""

import dataclasses
import math

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.graph_builder import generate_erdos_renyi
from financial_contagion_sim.sais_engine import (
    TARGETS,
    SAISParams,
    State,
    _select_event,
    compute_rates,
    gillespie_step,
    infected_positive_neighbors,
    make_initial_state,
    refresh_after_edge_flip,
    run_gillespie,
    sample_waiting_time,
    select_event,
)

PARAMS = SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7)
CODES = {"S": 1, "A": 0, "I": -1}


def _st(code: str) -> np.ndarray:
    """State array from a string such as ``"SAI"``."""
    return np.array([CODES[c] for c in code], dtype=np.int8)


def _graph(n: int, edges: list[tuple[int, int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def _rates(g: nx.Graph, states: np.ndarray, params: SAISParams = PARAMS) -> np.ndarray:
    return compute_rates(states, infected_positive_neighbors(g, states), params)


# --- states and parameters ---------------------------------------------------


def test_state_codes_match_spec() -> None:
    assert (State.SUSCEPTIBLE, State.ALERT, State.INFECTED) == (1, 0, -1)
    assert TARGETS == (State.SUSCEPTIBLE, State.ALERT, State.INFECTED)


@pytest.mark.parametrize(
    ("states", "error", "match"),
    [
        (np.array([1, 0, -1], dtype=np.int64), TypeError, "dtype int8"),
        (np.array([1.0, 0.0, -1.0]), TypeError, "dtype int8"),
        (np.array([1.0, np.nan, -1.0]), TypeError, "dtype int8"),
        (np.array([1.0, 0.5, -1.0]), TypeError, "dtype int8"),
        ([1, 0, -1], TypeError, "numpy.ndarray"),
        (np.array([1, 0], dtype=np.int8), ValueError, r"shape \(3,\)"),
        (np.ones((3, 1), dtype=np.int8), ValueError, r"shape \(3,\)"),
        (np.array([1, 2, -1], dtype=np.int8), ValueError, r"found \[2\]"),
        (np.array([1, -2, -1], dtype=np.int8), ValueError, r"found \[-2\]"),
    ],
    ids=["int64", "float", "nan", "half", "list", "short", "2d", "two", "minus2"],
)
def test_rejects_invalid_states(states: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        infected_positive_neighbors(_graph(3, []), states)


def test_params_accept_boundaries() -> None:
    SAISParams(beta=1.0, beta_a=0.0, kappa=0.0, delta=0.0)
    SAISParams(beta=1.0, beta_a=math.nextafter(1.0, 0.0), kappa=1.0, delta=1.0)
    SAISParams(beta=2, beta_a=1, kappa=0, delta=3)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"beta_a": 2.0}, ValueError, "0 <= beta_a < beta"),
        ({"beta_a": 2.5}, ValueError, "0 <= beta_a < beta"),
        ({"beta_a": -0.1}, ValueError, "0 <= beta_a < beta"),
        ({"beta": 0.0, "beta_a": 0.0}, ValueError, "beta must be > 0"),
        ({"beta": -1.0, "beta_a": 0.0}, ValueError, "beta must be > 0"),
        ({"kappa": -0.1}, ValueError, "kappa must be >= 0"),
        ({"delta": -0.1}, ValueError, "delta must be >= 0"),
        ({"beta": math.nan}, ValueError, "beta must be finite"),
        ({"beta": math.inf}, ValueError, "beta must be finite"),
        ({"beta_a": math.nan}, ValueError, "beta_a must be finite"),
        ({"kappa": math.inf}, ValueError, "kappa must be finite"),
        ({"delta": math.nan}, ValueError, "delta must be finite"),
        ({"beta": True}, TypeError, "beta must be a real number"),
        ({"delta": "0.7"}, TypeError, "delta must be a real number"),
        ({"kappa": None}, TypeError, "kappa must be a real number"),
    ],
)
def test_params_reject_invalid(
    kwargs: dict[str, object], error: type, match: str
) -> None:
    values: dict[str, object] = {"beta": 2.0, "beta_a": 0.5, "kappa": 3.0, "delta": 0.7}
    values.update(kwargs)
    with pytest.raises(error, match=match):
        SAISParams(**values)


def test_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PARAMS.beta = 5.0


# --- initial state ----------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "fraction", "expected_infected"),
    [(5, 0.5, 2), (3, 0.5, 2), (10, 0.25, 2), (100, 0.3, 30), (7, 0.0, 0), (7, 1.0, 7)],
)
def test_initial_state_infects_exactly_round_half_even(
    n: int, fraction: float, expected_infected: int
) -> None:
    init = make_initial_state(n, fraction, rng=0)
    assert expected_infected == round(fraction * n)
    assert init.states.dtype == np.int8
    assert int(np.sum(init.states == -1)) == expected_infected
    assert int(np.sum(init.states == 1)) == n - expected_infected
    assert int(np.sum(init.states == 0)) == 0
    assert init.requested_infected_fraction == fraction
    assert init.realized_infected_fraction == expected_infected / n


def test_initial_state_seeding() -> None:
    a = make_initial_state(200, 0.3, rng=11).states
    assert np.array_equal(a, make_initial_state(200, 0.3, rng=11).states)
    b = make_initial_state(200, 0.3, rng=np.random.default_rng(11)).states
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_initial_state(200, 0.3, rng=12).states)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((0, 0.5, 0), ValueError),
        ((5.0, 0.5, 0), TypeError),
        ((5, 1.5, 0), ValueError),
        ((5, math.nan, 0), ValueError),
        ((5, "0.5", 0), TypeError),
        ((5, 0.5, None), TypeError),
        ((5, 0.5, -1), ValueError),
    ],
)
def test_initial_state_rejects_bad_input(args: tuple[object, ...], error: type) -> None:
    with pytest.raises(error):
        make_initial_state(*args)


# --- N_i and rates -----------------------------------------------------------

# Path 0-1-2-3 with signs (+, +, -) and states S I A I.
PATH = _graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, -1)])
PATH_STATES = _st("SIAI")


def test_infected_positive_neighbors_ignores_negative_edges() -> None:
    counts = infected_positive_neighbors(PATH, PATH_STATES)
    assert counts.dtype == np.int64
    assert counts.tolist() == [1, 0, 1, 0]


def test_rates_hand_computed_path() -> None:
    # Columns: to S, to A, to I.
    expected = [
        [0.0, 3.0, 2.0],  # S with N=1: kappa, beta
        [0.7, 0.0, 0.0],  # I: delta
        [0.0, 0.0, 0.5],  # A with N=1: beta_a
        [0.7, 0.0, 0.0],  # I: delta
    ]
    assert _rates(PATH, PATH_STATES).tolist() == expected


def test_rates_hand_computed_star() -> None:
    star = _graph(5, [(0, 1, 1), (0, 2, 1), (0, 3, -1), (0, 4, 1)])
    expected = [
        [0.7, 0.0, 0.0],  # centre, I
        [0.0, 3.0, 2.0],  # S, positive edge to I
        [0.0, 0.0, 0.5],  # A, positive edge to I
        [0.0, 0.0, 0.0],  # S, negative edge to I contributes nothing
        [0.0, 3.0, 2.0],  # S, positive edge to I
    ]
    assert _rates(star, _st("ISASS")).tolist() == expected


def test_rates_scale_with_n_i() -> None:
    star = _graph(4, [(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    rates = _rates(star, _st("SIII"))
    assert rates[0].tolist() == [0.0, 3 * 3.0, 3 * 2.0]
    rates = _rates(star, _st("AIII"))
    assert rates[0].tolist() == [0.0, 0.0, 3 * 0.5]


def test_rate_channels_per_state() -> None:
    g = generate_erdos_renyi(40, 0.3, 0.6, rng=4)
    states = make_initial_state(40, 0.4, rng=4).states
    states[::5] = 0
    rates = _rates(g, states)
    assert np.all(rates >= 0)
    assert np.all(rates[states == 1, 0] == 0)
    assert np.all(rates[states == 0, :2] == 0)
    assert np.all(rates[states == -1, 1:] == 0)
    assert np.all(rates[states == -1, 0] == PARAMS.delta)


@pytest.mark.parametrize(
    ("code", "expected_total"),
    [("SSSS", 0.0), ("AAAA", 0.0), ("IIII", 4 * 0.7), ("SAIS", 0.7)],
)
def test_rates_on_isolated_nodes(code: str, expected_total: float) -> None:
    rates = _rates(_graph(4, []), _st(code))
    assert rates.sum() == pytest.approx(expected_total)
    assert np.all(rates[:, 1:] == 0)


@pytest.mark.parametrize("code", ["SSSS", "AAAA", "SASA"])
def test_no_infected_means_zero_total_rate(code: str) -> None:
    g = _graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, -1)])
    assert _rates(g, _st(code)).sum() == 0.0


def test_all_infected_only_recovery() -> None:
    g = _graph(3, [(0, 1, 1), (1, 2, 1)])
    assert _rates(g, _st("III")).tolist() == [[0.7, 0.0, 0.0]] * 3


def test_negative_edges_only_with_zero_delta_gives_zero_rate() -> None:
    g = _graph(3, [(0, 1, -1), (1, 2, -1)])
    params = SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.0)
    assert _rates(g, _st("SIA"), params).sum() == 0.0


def test_selection_probabilities_are_bounded_and_sum_to_one() -> None:
    g = generate_erdos_renyi(60, 0.1, 0.5, rng=9)
    states = make_initial_state(60, 0.3, rng=9).states
    rates = _rates(g, states)
    p = rates / rates.sum()
    assert np.all((p >= 0) & (p <= 1))
    assert abs(p.sum() - 1.0) <= 1e-12


def test_compute_rates_rejects_bad_input() -> None:
    states = _st("SI")
    with pytest.raises(TypeError, match="dtype int64"):
        compute_rates(states, np.zeros(2, dtype=np.int32), PARAMS)
    with pytest.raises(ValueError, match="shape"):
        compute_rates(states, np.zeros(3, dtype=np.int64), PARAMS)
    with pytest.raises(ValueError, match="non-negative"):
        compute_rates(states, np.array([0, -1], dtype=np.int64), PARAMS)
    with pytest.raises(TypeError, match="SAISParams"):
        compute_rates(states, np.zeros(2, dtype=np.int64), (2.0, 0.5, 3.0, 0.7))


# --- edge-flip hook -----------------------------------------------------------


def test_refresh_after_edge_flip_matches_recomputation() -> None:
    rng = np.random.default_rng(21)
    g = generate_erdos_renyi(30, 0.2, 0.5, rng=21)
    states = make_initial_state(30, 0.4, rng=21).states
    n_pos = infected_positive_neighbors(g, states)
    edges = sorted(g.edges)
    for _ in range(200):
        u, v = edges[int(rng.integers(len(edges)))]
        g.edges[u, v]["sign"] = -g.edges[u, v]["sign"]
        refresh_after_edge_flip(n_pos, states, u, v, g.edges[u, v]["sign"])
        assert np.array_equal(n_pos, infected_positive_neighbors(g, states))


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((0, 1, 0), ValueError),
        ((0, 1, 2), ValueError),
        ((0, 1, 1.0), TypeError),
        ((0, 1, True), TypeError),
        ((0, 0, 1), ValueError),
        ((0, 5, 1), ValueError),
        ((-1, 1, 1), ValueError),
    ],
)
def test_refresh_rejects_bad_input(args: tuple[object, ...], error: type) -> None:
    n_pos = np.zeros(3, dtype=np.int64)
    with pytest.raises(error):
        refresh_after_edge_flip(n_pos, _st("SIS"), *args)


# --- waiting time and event selection ----------------------------------------

ALLOWED = {(1, -1), (0, -1), (1, 0), (-1, 1)}  # S->I, A->I, S->A, I->S


@pytest.mark.parametrize(
    ("total_rate", "error"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_waiting_time_rejects_bad_rate(total_rate: object, error: type) -> None:
    with pytest.raises(error):
        sample_waiting_time(total_rate, np.random.default_rng(0))


def test_select_event_single_live_rate() -> None:
    rates = np.zeros((4, 3))
    rates[2, 1] = 0.3
    rng = np.random.default_rng(0)
    assert {select_event(rates, rng) for _ in range(200)} == {(2, State.ALERT)}


class _FixedDraw:
    """Stands in for a Generator whose ``random()`` returns a fixed value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


@pytest.mark.parametrize(
    ("draw", "expected"),
    [(0.0, (1, State.ALERT)), (0.5, (1, State.INFECTED)), (0.25, (1, State.ALERT))],
)
def test_select_event_boundaries_belong_to_the_next_live_rate(
    draw: float, expected: tuple[int, State]
) -> None:
    # Flattened rates [0, 0, 0, 0, 1, 1]: u = draw * 2. Interval [0, 1) is the
    # first live entry and [1, 2) the second; a draw of 0 never picks a zero rate.
    rates = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    assert _select_event(rates, _FixedDraw(draw)) == expected


def test_select_event_never_picks_zero_rates() -> None:
    rates = _rates(PATH, PATH_STATES)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        node, target = select_event(rates, rng)
        assert rates[node, TARGETS.index(target)] > 0


@pytest.mark.parametrize(
    ("rates", "error", "match"),
    [
        (np.zeros((3, 3)), ValueError, "every rate is 0"),
        (np.array([[1.0, -0.1, 0.0]]), ValueError, "non-negative"),
        (np.array([[1.0, math.nan, 0.0]]), ValueError, "finite"),
        (np.ones((3, 2)), ValueError, r"shape \(n, 3\)"),
        (np.ones((3, 3), dtype=np.int64), TypeError, "floating-point"),
        ([[1.0, 0.0, 0.0]], TypeError, "floating-point"),
    ],
    ids=["zero", "negative", "nan", "shape", "int", "list"],
)
def test_select_event_rejects_bad_rates(rates: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        select_event(rates, np.random.default_rng(0))


# --- single steps -------------------------------------------------------------


def _counts(g: nx.Graph, states: np.ndarray) -> np.ndarray:
    return infected_positive_neighbors(g, states)


@pytest.mark.parametrize("code", ["SSSS", "AASS"])
def test_step_absorbed_without_infection(code: str) -> None:
    g = _graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    states = _st(code)
    step = gillespie_step(g, states, _counts(g, states), PARAMS, 0, t_max=5.0)
    assert (step.reason, step.time, step.event) == ("absorbed", 5.0, None)
    assert states.tolist() == _st(code).tolist()


def test_step_absorbed_with_zero_delta_and_no_positive_contact() -> None:
    g = _graph(3, [(0, 1, -1), (1, 2, -1)])
    params = SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.0)
    states = _st("SIA")
    step = gillespie_step(g, states, _counts(g, states), params, 0, t_max=1.0)
    assert step.reason == "absorbed"


def test_step_past_t_max_is_not_applied() -> None:
    g = _graph(2, [(0, 1, 1)])
    states = _st("SI")
    n_pos = _counts(g, states)
    step = gillespie_step(g, states, n_pos, PARAMS, 0, time=1.0, t_max=1.0 + 1e-12)
    assert (step.reason, step.time, step.event) == ("t_max", 1.0 + 1e-12, None)
    assert states.tolist() == [1, -1]
    assert n_pos.tolist() == [1, 0]


def test_step_rejects_bad_clock() -> None:
    g = _graph(2, [(0, 1, 1)])
    states = _st("SI")
    with pytest.raises(ValueError, match="t_max must be >= time"):
        gillespie_step(g, states, _counts(g, states), PARAMS, 0, time=2.0, t_max=1.0)
    with pytest.raises(ValueError, match="time must be finite"):
        gillespie_step(g, states, _counts(g, states), PARAMS, 0, time=-1.0)


def test_every_step_conserves_states_and_tracks_n_exactly() -> None:
    g = generate_erdos_renyi(50, 0.12, 0.6, rng=31)
    states = make_initial_state(50, 0.2, rng=31).states
    n_pos = _counts(g, states)
    rng = np.random.default_rng(31)
    time, applied = 0.0, 0
    for _ in range(1500):
        before = states.copy()
        step = gillespie_step(g, states, n_pos, PARAMS, rng, time=time)
        if step.event is None:
            break
        applied += 1
        assert step.time > time
        time = step.time
        changed = np.flatnonzero(before != states)
        assert changed.tolist() == [step.event.node]
        assert (step.event.old_state, step.event.new_state) in ALLOWED
        assert before[step.event.node] == step.event.old_state
        assert set(np.unique(states).tolist()) <= {1, 0, -1}
        assert sum(int(np.sum(states == s)) for s in (1, 0, -1)) == 50
        assert np.array_equal(n_pos, _counts(g, states))
    assert applied == 1500


# --- full runs ----------------------------------------------------------------


def test_run_reports_absorbed_and_jumps_to_t_max() -> None:
    g = _graph(3, [(0, 1, 1), (1, 2, 1)])
    result = run_gillespie(g, _st("SSA"), PARAMS, t_max=4.0, rng=0)
    assert result.termination_reason == "absorbed"
    assert (result.time, result.n_events) == (4.0, 0)


def test_run_absorbs_after_recovery_of_isolated_node() -> None:
    result = run_gillespie(_graph(1, []), _st("I"), PARAMS, t_max=1e9, rng=0)
    assert result.termination_reason == "absorbed"
    assert (result.time, result.n_events) == (1e9, 1)
    assert result.states.tolist() == [1]


def test_run_stops_at_t_max() -> None:
    g = generate_erdos_renyi(30, 0.2, 0.7, rng=2)
    states = make_initial_state(30, 0.3, rng=2).states
    result = run_gillespie(g, states, PARAMS, t_max=0.5, rng=2, record_events=True)
    assert result.termination_reason == "t_max"
    assert result.time == 0.5
    assert result.events is not None
    assert len(result.events) == result.n_events > 0
    assert all(e.time <= 0.5 for e in result.events)


def test_run_t_max_zero_applies_nothing() -> None:
    g = _graph(2, [(0, 1, 1)])
    result = run_gillespie(g, _st("SI"), PARAMS, t_max=0.0, rng=0)
    assert (result.termination_reason, result.time, result.n_events) == (
        "t_max",
        0.0,
        0,
    )


@pytest.mark.parametrize("cap", [0, 1, 7])
def test_run_stops_at_max_events(cap: int) -> None:
    g = generate_erdos_renyi(30, 0.3, 0.8, rng=3)
    states = make_initial_state(30, 0.5, rng=3).states
    result = run_gillespie(
        g, states, PARAMS, t_max=1e6, rng=3, max_events=cap, record_events=True
    )
    assert result.termination_reason == "max_events"
    assert result.n_events == cap
    assert result.events is not None
    assert result.time == (result.events[-1].time if cap else 0.0)


def test_run_does_not_mutate_inputs_and_events_off_by_default() -> None:
    g = generate_erdos_renyi(20, 0.3, 0.5, rng=5)
    signs = sorted(g.edges(data="sign"))
    states = make_initial_state(20, 0.4, rng=5).states
    original = states.copy()
    result = run_gillespie(g, states, PARAMS, t_max=2.0, rng=5)
    assert np.array_equal(states, original)
    assert sorted(g.edges(data="sign")) == signs
    assert result.events is None


def test_run_event_log_replays_to_final_state() -> None:
    g = generate_erdos_renyi(40, 0.15, 0.6, rng=8)
    states = make_initial_state(40, 0.25, rng=8).states
    result = run_gillespie(g, states, PARAMS, t_max=3.0, rng=8, record_events=True)
    assert result.events is not None and result.n_events > 50
    replay = states.copy()
    last = 0.0
    for event in result.events:
        assert event.time > last
        last = event.time
        assert replay[event.node] == event.old_state
        assert (event.old_state, event.new_state) in ALLOWED
        replay[event.node] = event.new_state
        assert int(np.isin(replay, (1, 0, -1)).sum()) == 40
    assert np.array_equal(replay, result.states)


def test_run_seeding() -> None:
    g = generate_erdos_renyi(40, 0.15, 0.6, rng=8)
    states = make_initial_state(40, 0.25, rng=8).states

    def run(rng: object) -> tuple[object, ...]:
        r = run_gillespie(g, states, PARAMS, t_max=3.0, rng=rng, record_events=True)
        return tuple(r.events or ())

    assert run(17) == run(17)
    assert run(17) == run(np.random.default_rng(17))
    assert run(17) != run(18)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"t_max": -1.0}, ValueError),
        ({"t_max": math.inf}, ValueError),
        ({"t_max": math.nan}, ValueError),
        ({"t_max": "1"}, TypeError),
        ({"max_events": -1}, ValueError),
        ({"max_events": 1.0}, TypeError),
        ({"max_events": True}, TypeError),
        ({"record_events": 1}, TypeError),
        ({"params": (2.0, 0.5, 3.0, 0.7)}, TypeError),
        ({"rng": None}, TypeError),
    ],
)
def test_run_rejects_bad_input(kwargs: dict[str, object], error: type) -> None:
    args: dict[str, object] = {"params": PARAMS, "t_max": 1.0, "rng": 0}
    args.update(kwargs)
    with pytest.raises(error):
        run_gillespie(_graph(2, [(0, 1, 1)]), _st("SI"), **args)


# --- statistics (fixed seeds, 5-sigma bounds) -----------------------------------


def test_waiting_time_mean_and_variance() -> None:
    rate, samples = 2.5, 200_000
    rng = np.random.default_rng(101)
    x = np.array([sample_waiting_time(rate, rng) for _ in range(samples)])
    # Exp(R): mean 1/R, sd 1/R; Var of the sample variance ~ (mu4 - sigma^4)/N
    # with mu4 = 9/R^4, so its sd is sqrt(8/N)/R^2.
    assert abs(x.mean() - 1 / rate) <= 5 * (1 / rate) / math.sqrt(samples)
    assert abs(x.var() - 1 / rate**2) <= 5 * math.sqrt(8 / samples) / rate**2


def test_event_frequencies_match_rate_shares() -> None:
    rates = _rates(PATH, PATH_STATES)
    total, draws = float(rates.sum()), 60_000
    rng = np.random.default_rng(202)
    counts = np.zeros_like(rates)
    for _ in range(draws):
        node, target = select_event(rates, rng)
        counts[node, TARGETS.index(target)] += 1
    p = rates / total
    sd = np.sqrt(draws * p * (1 - p))
    assert np.all(np.abs(counts - draws * p) <= 5 * sd)
    assert np.all(counts[rates == 0] == 0)


def test_isolated_recovery_time_has_mean_one_over_delta() -> None:
    params = SAISParams(beta=1.0, beta_a=0.0, kappa=0.0, delta=2.0)
    g, trials = _graph(1, []), 20_000
    rng = np.random.default_rng(303)
    times = []
    for _ in range(trials):
        states, n_pos = _st("I"), np.zeros(1, dtype=np.int64)
        step = gillespie_step(g, states, n_pos, params, rng)
        assert step.event is not None and states.tolist() == [1]
        times.append(step.time)
    mean = 1 / params.delta
    assert abs(float(np.mean(times)) - mean) <= 5 * mean / math.sqrt(trials)
