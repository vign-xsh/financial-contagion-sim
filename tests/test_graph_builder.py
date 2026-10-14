"""Tests for signed financial network construction (``graph_builder``)."""

import math
from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.graph_builder import (
    _assign_signs,
    _corr_distance,
    generate_barabasi_albert,
    generate_erdos_renyi,
    graph_from_correlation,
    validate_signed_graph,
)

GraphFactory = Callable[[float], nx.Graph]
SeededFactory = Callable[[object], nx.Graph]


def _signed(n: int, edges: list[tuple[int, int, object]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def _signed_edges(g: nx.Graph) -> list[tuple[int, int, int]]:
    return sorted((min(u, v), max(u, v), s) for u, v, s in g.edges(data="sign"))


def _positives(g: nx.Graph) -> int:
    return sum(1 for _, _, s in g.edges(data="sign") if s == 1)


# --- validate_signed_graph ---------------------------------------------------


def test_valid_signed_graph_passes() -> None:
    validate_signed_graph(_signed(4, [(0, 1, 1), (1, 2, -1), (2, 3, 1)]))


def test_isolated_nodes_and_no_edges_pass() -> None:
    validate_signed_graph(_signed(3, []))


def test_signs_are_mutable_in_place() -> None:
    g = _signed(2, [(0, 1, 1)])
    g.edges[0, 1]["sign"] = -g.edges[0, 1]["sign"]
    validate_signed_graph(g)
    assert g.edges[0, 1]["sign"] == -1


@pytest.mark.parametrize(
    "g", [{}, nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()], ids=repr
)
def test_rejects_non_simple_undirected_graph(g: object) -> None:
    with pytest.raises(TypeError, match="undirected simple networkx.Graph"):
        validate_signed_graph(g)


def test_rejects_self_loop() -> None:
    g = _signed(2, [(0, 1, 1), (1, 1, 1)])
    with pytest.raises(ValueError, match="self-loops"):
        validate_signed_graph(g)


def test_rejects_non_contiguous_nodes() -> None:
    g = nx.Graph()
    g.add_nodes_from([0, 2])
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        validate_signed_graph(g)


@pytest.mark.parametrize("node", ["0", 0.0, np.int64(0), True], ids=repr)
def test_rejects_non_int_nodes(node: object) -> None:
    g = nx.Graph()
    g.add_node(node)
    with pytest.raises(TypeError, match="nodes must be Python ints"):
        validate_signed_graph(g)


def test_rejects_missing_sign() -> None:
    g = nx.Graph()
    g.add_edge(0, 1)
    with pytest.raises(ValueError, match="no 'sign' attribute"):
        validate_signed_graph(g)


@pytest.mark.parametrize("sign", [0, 2, -2], ids=repr)
def test_rejects_out_of_range_sign(sign: int) -> None:
    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        validate_signed_graph(_signed(2, [(0, 1, sign)]))


@pytest.mark.parametrize(
    "sign", [1.0, -1.0, True, False, "1", np.int64(1), None], ids=repr
)
def test_rejects_non_int_sign(sign: object) -> None:
    with pytest.raises(TypeError, match="sign must be a Python int"):
        validate_signed_graph(_signed(2, [(0, 1, sign)]))


# --- generators: structure ---------------------------------------------------


@pytest.mark.parametrize(("n", "m"), [(2, 1), (6, 1), (10, 3), (50, 4), (30, 29)])
def test_ba_edge_count_is_m_times_n_minus_m(n: int, m: int) -> None:
    g = generate_barabasi_albert(n, m, 0.5, rng=0)
    assert g.number_of_edges() == m * (n - m)


@pytest.mark.parametrize(
    "g",
    [
        generate_barabasi_albert(40, 3, 0.3, rng=1),
        generate_erdos_renyi(40, 0.2, 0.3, rng=1),
    ],
    ids=["ba", "er"],
)
def test_generated_graph_satisfies_contract(g: nx.Graph) -> None:
    assert type(g) is nx.Graph
    assert sorted(g.nodes) == list(range(40))
    assert all(type(s) is int and s in (1, -1) for _, _, s in g.edges(data="sign"))
    validate_signed_graph(g)


def test_er_single_node() -> None:
    g = generate_erdos_renyi(1, 1.0, 0.5, rng=0)
    assert list(g.nodes) == [0]
    assert g.number_of_edges() == 0


def test_er_p0_has_no_edges_and_no_realized_r0() -> None:
    g = generate_erdos_renyi(10, 0.0, 0.7, rng=0)
    assert g.number_of_edges() == 0
    assert g.graph["r0_requested"] == 0.7
    assert g.graph["r0_realized"] is None


def test_er_p1_is_complete() -> None:
    g = generate_erdos_renyi(7, 1.0, 0.5, rng=0)
    assert g.number_of_edges() == 7 * 6 // 2


# --- generators: r0 exactness ------------------------------------------------


@pytest.mark.parametrize(
    ("make", "m", "r0", "expected_pos"),
    [
        (lambda r0: generate_barabasi_albert(6, 1, r0, rng=3), 5, 0.5, 2),
        (lambda r0: generate_erdos_renyi(3, 1.0, r0, rng=3), 3, 0.5, 2),
        (lambda r0: generate_erdos_renyi(2, 1.0, r0, rng=3), 1, 0.5, 0),
        (lambda r0: generate_erdos_renyi(5, 1.0, r0, rng=3), 10, 0.25, 2),
        (lambda r0: generate_erdos_renyi(4, 1.0, r0, rng=3), 6, 0.75, 4),
        (lambda r0: generate_barabasi_albert(50, 4, r0, rng=3), 184, 0.3, 55),
        (lambda r0: generate_barabasi_albert(10, 3, r0, rng=3), 21, 0.0, 0),
        (lambda r0: generate_barabasi_albert(10, 3, r0, rng=3), 21, 1.0, 21),
    ],
    ids=["m5-half", "m3-half", "m1-half", "m10-quarter", "m6", "m184", "r0=0", "r0=1"],
)
def test_r0_is_exact_round_half_even(
    make: GraphFactory, m: int, r0: float, expected_pos: int
) -> None:
    g = make(r0)
    assert g.number_of_edges() == m
    assert expected_pos == round(r0 * m)
    assert _positives(g) == expected_pos
    assert g.number_of_edges() - _positives(g) == m - expected_pos
    assert g.graph["r0_requested"] == r0
    assert g.graph["r0_realized"] == expected_pos / m


# --- generators: seeding -----------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda rng: generate_barabasi_albert(60, 3, 0.4, rng),
        lambda rng: generate_erdos_renyi(60, 0.1, 0.4, rng),
    ],
    ids=["ba", "er"],
)
def test_seeding(make: SeededFactory) -> None:
    same_a, same_b = make(42), make(42)
    assert sorted(same_a.nodes) == sorted(same_b.nodes)
    assert _signed_edges(same_a) == _signed_edges(same_b)

    from_generator = make(np.random.default_rng(42))
    assert _signed_edges(from_generator) == _signed_edges(same_a)

    assert _signed_edges(make(43)) != _signed_edges(same_a)


def test_signs_depend_on_edge_set_not_insertion_order() -> None:
    edges = [(u, v) for u in range(12) for v in range(u + 1, 12) if (u * v) % 3]
    forward, backward = nx.Graph(), nx.Graph()
    forward.add_nodes_from(range(12))
    backward.add_nodes_from(range(12))
    forward.add_edges_from(edges)
    backward.add_edges_from((v, u) for u, v in reversed(edges))
    _assign_signs(forward, 0.5, np.random.default_rng(7))
    _assign_signs(backward, 0.5, np.random.default_rng(7))
    assert _signed_edges(forward) == _signed_edges(backward)


def test_same_topology_different_seed_changes_signs() -> None:
    a = generate_erdos_renyi(30, 1.0, 0.5, rng=1)
    b = generate_erdos_renyi(30, 1.0, 0.5, rng=2)
    assert sorted(a.edges) == sorted(b.edges)
    assert _signed_edges(a) != _signed_edges(b)
    assert _positives(a) == _positives(b) == round(0.5 * 435)


# --- generators: parameter validation ---------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n": 1, "m": 1}, ValueError, "m must satisfy 1 <= m < n"),
        ({"n": 5, "m": 5}, ValueError, "m must satisfy 1 <= m < n, got m=5, n=5"),
        ({"n": 5, "m": 0}, ValueError, "m must be >= 1, got 0"),
        ({"n": 0, "m": 1}, ValueError, "n must be >= 1, got 0"),
        ({"n": 5.0, "m": 1}, TypeError, "n must be an int"),
        ({"n": True, "m": 1}, TypeError, "n must be an int"),
        ({"n": "5", "m": 1}, TypeError, "n must be an int"),
        ({"n": 5, "m": 1.0}, TypeError, "m must be an int"),
        ({"n": 5, "m": False}, TypeError, "m must be an int"),
    ],
)
def test_ba_rejects_bad_n_m(kwargs: dict[str, object], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        generate_barabasi_albert(r0=0.5, rng=0, **kwargs)


def test_ba_accepts_m_equal_n_minus_1() -> None:
    assert generate_barabasi_albert(5, 4, 0.5, rng=0).number_of_edges() == 4


@pytest.mark.parametrize(
    ("p", "error", "match"),
    [
        (-0.01, ValueError, r"p must be in \[0, 1\], got -0.01"),
        (1.01, ValueError, r"p must be in \[0, 1\], got 1.01"),
        (float("nan"), ValueError, r"p must be in \[0, 1\], got nan"),
        ("0.5", TypeError, "p must be a real number"),
        (True, TypeError, "p must be a real number"),
        (None, TypeError, "p must be a real number"),
    ],
)
def test_er_rejects_bad_p(p: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        generate_erdos_renyi(5, p, 0.5, rng=0)


def test_er_rejects_bad_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 1, got 0"):
        generate_erdos_renyi(0, 0.5, 0.5, rng=0)
    with pytest.raises(TypeError, match="n must be an int"):
        generate_erdos_renyi(np.int64(5), 0.5, 0.5, rng=0)


@pytest.mark.parametrize(
    ("r0", "error", "match"),
    [
        (-0.1, ValueError, r"r0 must be in \[0, 1\]"),
        (1.1, ValueError, r"r0 must be in \[0, 1\]"),
        (float("nan"), ValueError, r"r0 must be in \[0, 1\], got nan"),
        (float("inf"), ValueError, r"r0 must be in \[0, 1\]"),
        ("0.5", TypeError, "r0 must be a real number"),
        (True, TypeError, "r0 must be a real number"),
    ],
)
def test_generators_reject_bad_r0(r0: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        generate_barabasi_albert(10, 2, r0, rng=0)
    with pytest.raises(error, match=match):
        generate_erdos_renyi(10, 0.5, r0, rng=0)


@pytest.mark.parametrize(
    ("rng", "error", "match"),
    [
        (None, TypeError, "rng must be a numpy.random.Generator"),
        (1.5, TypeError, "rng must be a numpy.random.Generator"),
        (True, TypeError, "rng must be a numpy.random.Generator"),
        ("42", TypeError, "rng must be a numpy.random.Generator"),
        (np.random.RandomState(0), TypeError, "rng must be a numpy.random.Generator"),
        (-1, ValueError, "rng seed must be a non-negative int, got -1"),
    ],
    ids=repr,
)
def test_generators_reject_bad_rng(rng: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        generate_barabasi_albert(10, 2, 0.5, rng)
    with pytest.raises(error, match=match):
        generate_erdos_renyi(10, 0.5, 0.5, rng)


# --- graph_from_correlation --------------------------------------------------

TOL = 1e-8


def _corr(n: int, entries: dict[tuple[int, int], float]) -> np.ndarray:
    a = np.eye(n)
    for (i, j), value in entries.items():
        a[i, j] = a[j, i] = value
    return a


def test_correlation_hand_built_example() -> None:
    corr = _corr(4, {(0, 1): 0.8, (0, 2): -0.6, (1, 2): 0.0, (1, 3): 0.3})
    g = graph_from_correlation(corr, tau=0.5)
    validate_signed_graph(g)
    assert sorted(g.nodes) == [0, 1, 2, 3]
    assert _signed_edges(g) == [(0, 1, 1), (0, 2, -1)]
    assert g.edges[0, 1]["distance"] == pytest.approx(math.sqrt(0.4))
    assert g.edges[0, 2]["distance"] == pytest.approx(math.sqrt(3.2))
    assert g.graph["r0_realized"] == 0.5
    assert "r0_requested" not in g.graph


def test_correlation_distance_formula_special_values() -> None:
    assert _corr_distance(0.0) == math.sqrt(2.0)
    assert _corr_distance(1.0) == 0.0
    assert _corr_distance(-1.0) == 2.0
    assert _corr_distance(0.5) == pytest.approx(1.0)


def test_correlation_extreme_values_through_public_api() -> None:
    g = graph_from_correlation(_corr(3, {(0, 1): 1.0, (0, 2): -1.0}), tau=0.9)
    assert g.edges[0, 1]["distance"] == 0.0
    assert g.edges[0, 1]["sign"] == 1
    assert g.edges[0, 2]["distance"] == 2.0
    assert g.edges[0, 2]["sign"] == -1


def test_correlation_threshold_is_inclusive_and_symmetric_in_sign() -> None:
    corr = _corr(4, {(0, 1): 0.5, (0, 2): -0.5, (0, 3): 0.49999})
    assert _signed_edges(graph_from_correlation(corr, tau=0.5)) == [
        (0, 1, 1),
        (0, 2, -1),
    ]


def test_correlation_zero_excluded_even_at_tau_zero() -> None:
    corr = _corr(3, {(0, 1): 0.0, (0, 2): -0.01, (1, 2): 0.01})
    g = graph_from_correlation(corr, tau=0.0)
    assert _signed_edges(g) == [(0, 2, -1), (1, 2, 1)]


def test_correlation_tau_one_keeps_only_perfect_correlation() -> None:
    corr = _corr(3, {(0, 1): 1.0, (0, 2): 0.999, (1, 2): -1.0})
    assert _signed_edges(graph_from_correlation(corr, tau=1.0)) == [
        (0, 1, 1),
        (1, 2, -1),
    ]


def test_correlation_single_asset_and_no_edges() -> None:
    g = graph_from_correlation(np.array([[1.0]]), tau=0.5)
    assert list(g.nodes) == [0]
    assert g.number_of_edges() == 0
    assert g.graph["r0_realized"] is None


def test_correlation_integer_identity_is_accepted() -> None:
    g = graph_from_correlation(np.eye(3, dtype=int), tau=0.1)
    assert sorted(g.nodes) == [0, 1, 2]
    assert g.number_of_edges() == 0


def test_correlation_signs_are_python_ints() -> None:
    g = graph_from_correlation(_corr(3, {(0, 1): 0.9, (1, 2): -0.9}), tau=0.5)
    assert {type(s) for _, _, s in g.edges(data="sign")} == {int}
    assert {type(d) for _, _, d in g.edges(data="distance")} == {float}


@pytest.mark.parametrize("value", [1.0 + TOL / 2, 1.0 + TOL], ids=["half", "at"])
def test_correlation_clamps_just_above_one(value: float) -> None:
    g = graph_from_correlation(_corr(2, {(0, 1): value}), tau=0.5)
    assert g.edges[0, 1]["distance"] == 0.0


@pytest.mark.parametrize("value", [-1.0 - TOL / 2, -1.0 - TOL], ids=["half", "at"])
def test_correlation_clamps_just_below_minus_one(value: float) -> None:
    g = graph_from_correlation(_corr(2, {(0, 1): value}), tau=0.5)
    assert g.edges[0, 1]["distance"] == 2.0


@pytest.mark.parametrize("value", [1.0 + 2 * TOL, -1.0 - 2 * TOL, 1.5, -3.0])
def test_correlation_rejects_out_of_range_beyond_tolerance(value: float) -> None:
    with pytest.raises(ValueError, match=r"entries must be in \[-1, 1\]"):
        graph_from_correlation(_corr(2, {(0, 1): value}), tau=0.5)


def test_correlation_symmetry_tolerance_uses_upper_triangle() -> None:
    corr = _corr(2, {(0, 1): 0.6})
    corr[1, 0] = 0.6 + TOL / 2
    g = graph_from_correlation(corr, tau=0.6)
    assert g.edges[0, 1]["distance"] == _corr_distance(0.6)


def test_correlation_rejects_asymmetry_beyond_tolerance() -> None:
    corr = _corr(2, {(0, 1): 0.6})
    corr[1, 0] = 0.6 + 2 * TOL
    with pytest.raises(ValueError, match="symmetric within"):
        graph_from_correlation(corr, tau=0.5)


def test_correlation_diagonal_tolerance() -> None:
    ok = np.eye(2)
    ok[0, 0] = 1.0 - TOL / 2
    graph_from_correlation(ok, tau=0.5)
    bad = np.eye(2)
    bad[1, 1] = 1.0 + 2 * TOL
    with pytest.raises(ValueError, match="diagonal must be 1"):
        graph_from_correlation(bad, tau=0.5)
    with pytest.raises(ValueError, match="diagonal must be 1"):
        graph_from_correlation(np.zeros((2, 2)), tau=0.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=repr)
def test_correlation_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        graph_from_correlation(_corr(3, {(0, 1): bad}), tau=0.5)


@pytest.mark.parametrize(
    "corr",
    [np.ones(3), np.ones((2, 3)), np.ones((2, 2, 2)), np.ones((1, 0))],
    ids=["1d", "2x3", "3d", "1x0"],
)
def test_correlation_rejects_non_square(corr: np.ndarray) -> None:
    with pytest.raises(ValueError, match="square 2-D array"):
        graph_from_correlation(corr, tau=0.5)


def test_correlation_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least 1x1"):
        graph_from_correlation(np.zeros((0, 0)), tau=0.5)


@pytest.mark.parametrize(
    "corr",
    [[[1.0, 0.5], [0.5, 1.0]], ((1.0,),), "corr", None],
    ids=["list", "tuple", "str", "none"],
)
def test_correlation_rejects_non_ndarray(corr: object) -> None:
    with pytest.raises(TypeError, match=r"corr must be a numpy\.ndarray"):
        graph_from_correlation(corr, tau=0.5)


@pytest.mark.parametrize(
    "dtype", [bool, complex, object, "U1"], ids=["bool", "complex", "object", "str"]
)
def test_correlation_rejects_non_real_dtype(dtype: object) -> None:
    corr = np.eye(2).astype(dtype)
    with pytest.raises(TypeError, match="real numeric dtype"):
        graph_from_correlation(corr, tau=0.5)


@pytest.mark.parametrize(
    ("tau", "error", "match"),
    [
        (-0.1, ValueError, r"tau must be in \[0, 1\]"),
        (1.1, ValueError, r"tau must be in \[0, 1\]"),
        (float("nan"), ValueError, r"tau must be in \[0, 1\], got nan"),
        ("0.5", TypeError, "tau must be a real number"),
        (None, TypeError, "tau must be a real number"),
        (True, TypeError, "tau must be a real number"),
    ],
)
def test_correlation_rejects_bad_tau(tau: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        graph_from_correlation(np.eye(2), tau)
