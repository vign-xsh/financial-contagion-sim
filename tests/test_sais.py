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
    compute_rates,
    infected_positive_neighbors,
    make_initial_state,
    refresh_after_edge_flip,
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
