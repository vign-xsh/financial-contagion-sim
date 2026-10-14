"""Tests for signed financial network construction (``graph_builder``)."""

import networkx as nx
import numpy as np
import pytest

from financial_contagion_sim.graph_builder import validate_signed_graph


def _signed(n: int, edges: list[tuple[int, int, object]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, sign in edges:
        g.add_edge(u, v, sign=sign)
    return g


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
