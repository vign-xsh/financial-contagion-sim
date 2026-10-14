"""Validate fast local energy differences against exact matrix recomputation.

Spec §4, for ``energy_calculator``; the energy definitions are spec §3.2 and
the local flip rule is spec §3.3.
"""

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.energy_calculator import (
    adjacency_matrix,
    exact_structural_energy,
    spreading_energy,
    structural_energy,
    total_energy,
)
from financial_contagion_sim.graph_builder import (
    generate_barabasi_albert,
    generate_erdos_renyi,
)


def _graph(n: int, edges: list[tuple[int, int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


def _complete(n: int, sign: object) -> nx.Graph:
    """Complete graph; ``sign`` is a constant or a function of the pair."""
    edges = [
        (u, v, sign(u, v) if callable(sign) else sign)
        for u, v in itertools.combinations(range(n), 2)
    ]
    return _graph(n, edges)


def _oracle_triad_sum(g: nx.Graph) -> int:
    """Brute-force sum over all i < j < k (test oracle only)."""

    def a(i: int, j: int) -> int:
        return g.edges[i, j]["sign"] if g.has_edge(i, j) else 0

    return sum(
        a(i, j) * a(j, k) * a(k, i)
        for i, j, k in itertools.combinations(range(g.number_of_nodes()), 3)
    )


def _oracle_structural(g: nx.Graph) -> Fraction:
    n = g.number_of_nodes()
    return Fraction(-_oracle_triad_sum(g), math.comb(n, 3)) if n >= 3 else Fraction(0)


def _random_signed(seed: int, n: int, p: float, r0: float) -> nx.Graph:
    return generate_erdos_renyi(n, p, r0, rng=seed)


Engine = Callable[[nx.Graph], float]
ENGINES: list[Engine] = [exact_structural_energy, structural_energy]


# --- adjacency matrix ---------------------------------------------------------


def test_adjacency_matrix_is_signed_symmetric_int64() -> None:
    g = _graph(4, [(0, 1, 1), (1, 2, -1), (0, 3, -1)])
    a = adjacency_matrix(g)
    assert a.dtype == np.int64
    assert a.tolist() == [
        [0, 1, 0, -1],
        [1, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
    ]


def test_adjacency_matrix_validates_graph() -> None:
    with pytest.raises(TypeError):
        adjacency_matrix(nx.DiGraph())


# --- structural energy: spec table and hand-computed values -------------------


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
@pytest.mark.parametrize(
    ("signs", "expected"),
    [((1, 1, 1), -1.0), ((1, -1, -1), -1.0), ((1, 1, -1), 1.0), ((-1, -1, -1), 1.0)],
    ids=["+++", "+--", "++-", "---"],
)
def test_triad_sign_table(
    engine: Engine, signs: tuple[int, int, int], expected: float
) -> None:
    for perm in set(itertools.permutations(signs)):
        g = _graph(3, [(0, 1, perm[0]), (1, 2, perm[1]), (0, 2, perm[2])])
        assert engine(g) == expected


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
def test_hand_computed_k4(engine: Engine) -> None:
    assert engine(_complete(4, 1)) == -1.0
    # Factions {0, 1} and {2, 3}: positive inside, negative across -> balanced.
    assert engine(_complete(4, lambda u, v: 1 if (u < 2) == (v < 2) else -1)) == -1.0
    # Negative (0, 1) and (2, 3): every one of the 4 triads has product -1.
    assert (
        engine(_complete(4, lambda u, v: -1 if (u, v) in {(0, 1), (2, 3)} else 1))
        == 1.0
    )
    # Negative (0, 1) only: triads 012 and 013 are -1, 023 and 123 are +1.
    assert engine(_complete(4, lambda u, v: -1 if (u, v) == (0, 1) else 1)) == 0.0


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
def test_no_triangles_gives_zero(engine: Engine) -> None:
    assert engine(_graph(3, [(0, 1, 1), (1, 2, 1)])) == 0.0
    assert engine(_graph(5, [(0, k, 1) for k in range(1, 5)])) == 0.0
    assert engine(_graph(4, [(0, 1, -1), (1, 2, 1), (2, 3, -1)])) == 0.0


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
@pytest.mark.parametrize("n", [1, 2])
def test_fewer_than_three_nodes_gives_zero(engine: Engine, n: int) -> None:
    edges = [(0, 1, -1)] if n == 2 else []
    assert engine(_graph(n, edges)) == 0.0


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
@pytest.mark.parametrize(("n", "c3"), [(3, 1), (4, 4), (5, 10)])
def test_normalization_by_c_n_3(engine: Engine, n: int, c3: int) -> None:
    one_triangle = _graph(n, [(0, 1, 1), (1, 2, 1), (0, 2, 1)])
    assert engine(one_triangle) == -1 / c3
    assert engine(_complete(n, 1)) == -1.0


@pytest.mark.parametrize("engine", ENGINES, ids=["exact", "sparse"])
def test_sparse_graphs_have_small_energy(engine: Engine) -> None:
    g = generate_barabasi_albert(200, 2, 1.0, rng=0)
    assert 0 < -engine(g) < 0.01
    assert engine(_complete(12, 1)) == -1.0


# --- engines agree with the brute-force oracle ---------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_engines_match_oracle_on_random_graphs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 13))
    g = _random_signed(seed, n, float(rng.uniform(0.2, 1.0)), float(rng.uniform()))
    expected = float(_oracle_structural(g))
    assert exact_structural_energy(g) == expected
    assert structural_energy(g) == expected


@pytest.mark.parametrize("seed", range(5))
def test_engines_agree_on_larger_graphs(seed: int) -> None:
    g = _random_signed(seed, 80, 0.3, 0.4)
    assert exact_structural_energy(g) == structural_energy(g)


def test_engines_do_not_mutate_graph() -> None:
    g = _random_signed(3, 20, 0.4, 0.5)
    before = sorted(g.edges(data="sign"))
    exact_structural_energy(g)
    structural_energy(g)
    adjacency_matrix(g)
    assert sorted(g.edges(data="sign")) == before


# --- spreading energy -----------------------------------------------------------

CODES = {"S": 1, "A": 0, "I": -1}


def _st(code: str) -> np.ndarray:
    return np.array([CODES[c] for c in code], dtype=np.int8)


# Spec §3.2 table, written out pair by pair: (state_u, state_v, sign) -> E_ij^p.
SPEC_PAIR_TABLE = {
    ("S", "I", 1): 1,
    ("I", "S", 1): 1,
    ("A", "I", 1): 1,
    ("I", "A", 1): 1,
    ("S", "I", -1): -1,
    ("I", "S", -1): -1,
    ("A", "I", -1): -1,
    ("I", "A", -1): -1,
    ("S", "S", 1): 0,
    ("S", "S", -1): 0,
    ("I", "I", 1): 0,
    ("I", "I", -1): 0,
    ("A", "A", 1): 0,
    ("A", "A", -1): 0,
    ("S", "A", 1): 0,
    ("S", "A", -1): 0,
    ("A", "S", 1): 0,
    ("A", "S", -1): 0,
}


def test_pair_table_covers_every_combination() -> None:
    combos = {(a, b, s) for a in "SAI" for b in "SAI" for s in (1, -1)}
    assert set(SPEC_PAIR_TABLE) == combos


@pytest.mark.parametrize(("pair", "expected"), sorted(SPEC_PAIR_TABLE.items()))
def test_spreading_pair_table(pair: tuple[str, str, int], expected: int) -> None:
    a, b, sign = pair
    # n = 2, so C(2, 2) = 1 and E_p equals the single pair term.
    assert spreading_energy(_graph(2, [(0, 1, sign)]), _st(a + b)) == expected


def test_non_edge_contributes_nothing() -> None:
    assert spreading_energy(_graph(2, []), _st("SI")) == 0.0
    assert spreading_energy(_graph(3, [(0, 1, 1)]), _st("SSI")) == 0.0


def test_spreading_hand_computed_four_nodes() -> None:
    g = _graph(4, [(0, 1, 1), (1, 2, -1), (2, 3, 1), (0, 3, 1), (0, 2, 1)])
    # States S I A I: 01 S+I=+1, 12 I-A=-1, 23 A+I=+1, 03 S+I=+1, 02 S+A=0.
    assert spreading_energy(g, _st("SIAI")) == 2 / 6


def test_spreading_single_node_is_zero() -> None:
    assert spreading_energy(_graph(1, []), _st("I")) == 0.0


def test_spreading_extremes_on_complete_graph() -> None:
    # Star of one I node to n-1 S nodes on a complete graph: n-1 active pairs.
    n = 6
    assert spreading_energy(_complete(n, 1), _st("I" + "S" * 5)) == 5 / 15
    assert spreading_energy(_complete(n, -1), _st("IIISSS")) == -9 / 15


@pytest.mark.parametrize("seed", range(20))
def test_spreading_bounded_on_random_graphs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    g = _random_signed(seed, n, float(rng.uniform()), float(rng.uniform()))
    states = rng.choice(np.array([1, 0, -1], dtype=np.int8), size=n)
    assert -1.0 <= spreading_energy(g, states) <= 1.0


def test_spreading_validates_states() -> None:
    g = _graph(2, [(0, 1, 1)])
    with pytest.raises(TypeError, match="dtype int8"):
        spreading_energy(g, np.array([1, -1]))
    with pytest.raises(ValueError, match="shape"):
        spreading_energy(g, _st("SIS"))
    with pytest.raises(ValueError, match="found"):
        spreading_energy(g, np.array([1, 5], dtype=np.int8))


# --- total energy ------------------------------------------------------------------


def test_total_energy_endpoints_and_mix() -> None:
    g = _random_signed(4, 25, 0.4, 0.5)
    states = np.random.default_rng(4).choice(np.array([1, 0, -1], dtype=np.int8), 25)
    e_delta, e_p = structural_energy(g), spreading_energy(g, states)
    assert total_energy(g, states, 1.0) == e_delta
    assert total_energy(g, states, 0.0) == e_p
    assert total_energy(g, states, 0.3) == pytest.approx(0.3 * e_delta + 0.7 * e_p)


@pytest.mark.parametrize("seed", range(10))
def test_total_energy_bounded(seed: int) -> None:
    rng = np.random.default_rng(seed)
    g = _random_signed(seed, 15, float(rng.uniform()), float(rng.uniform()))
    states = rng.choice(np.array([1, 0, -1], dtype=np.int8), 15)
    assert -1.0 <= total_energy(g, states, float(rng.uniform())) <= 1.0


@pytest.mark.parametrize(
    ("alpha", "error"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (math.nan, ValueError),
        ("0.5", TypeError),
        (True, TypeError),
    ],
)
def test_total_energy_rejects_bad_alpha(alpha: object, error: type) -> None:
    with pytest.raises(error):
        total_energy(_graph(2, [(0, 1, 1)]), _st("SI"), alpha)
