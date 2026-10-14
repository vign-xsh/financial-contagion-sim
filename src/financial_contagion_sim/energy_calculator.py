"""Network energy (spec §4; formulas in §3.2 and §3.3).

Responsibility: an exact O(n^3) energy engine that recomputes the energy over
all triads, and a fast local energy-difference engine that evaluates the
change caused by flipping a single edge sign.

Signed adjacency (spec §3.2): ``A_ij`` is the edge's ``"sign"`` (+1 or -1) and
0 for a non-edge, so only closed triangles contribute to triad sums.

Structural-balance energy (spec §3.2)::

    E_delta = -1 / C(n, 3) * sum over i < j < k of A_ij * A_jk * A_ki

A triad is balanced when its product is +1 (+++ or +--) and unbalanced when it
is -1 (++- or ---). The spec's statement that a perfectly balanced network
gives ``E_delta = -1`` (and a fully frustrated one +1) holds for complete
graphs. On sparse graphs most triples are not triangles, so ``|E_delta|`` is
far below 1; ``E_delta = 0.0`` for ``n < 3``.

Two engines compute it. :func:`exact_structural_energy` is the dense O(n^3)
reference: the integer sum is ``trace(A^3) / 6``, divided once at the end.
:func:`structural_energy` is the sparse engine: each triangle is visited once
through common neighbours, costing O(sum over edges of min(deg u, deg v)).
Both return the same float, because both divide the same exact integer.
"""

import math

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim.graph_builder import validate_signed_graph

AdjacencyMatrix = npt.NDArray[np.int64]


def adjacency_matrix(g: "nx.Graph[int]") -> AdjacencyMatrix:
    """Return the dense signed adjacency matrix ``A`` (spec §3.2).

    Args:
        g: Signed graph satisfying the Phase 1 contract.

    Returns:
        Symmetric ``(n, n)`` ``numpy.int64`` matrix with entries in
        ``{-1, 0, +1}`` and a zero diagonal. ``int64`` (not ``int8``) so that
        ``A @ A @ A`` cannot overflow.

    Raises:
        TypeError: ``g`` violates the Phase 1 contract's types.
        ValueError: ``g`` violates the Phase 1 contract's values.
    """
    validate_signed_graph(g)
    n = g.number_of_nodes()
    a = np.zeros((n, n), dtype=np.int64)
    for u, v, data in g.edges(data=True):
        a[u, v] = a[v, u] = data["sign"]
    return a


def exact_structural_energy(g: "nx.Graph[int]") -> float:
    """Exact structural-balance energy E_delta via the dense matrix (spec §3.2).

    The triad sum over ``i < j < k`` is ``trace(A^3) / 6``: every triangle
    appears 6 times among ordered triples, and the zero diagonal removes
    repeated indices. Integer arithmetic throughout; one division at the end.
    Cost O(n^3) time and O(n^2) memory.

    Args:
        g: Signed graph satisfying the Phase 1 contract.

    Returns:
        ``E_delta`` in ``[-1, 1]``; ``0.0`` when ``n < 3``.

    Raises:
        TypeError: ``g`` violates the Phase 1 contract's types.
        ValueError: ``g`` violates the Phase 1 contract's values.
    """
    a = adjacency_matrix(g)
    n = a.shape[0]
    if n < 3:
        return 0.0
    triad_sum = int(np.trace(a @ a @ a)) // 6
    return -triad_sum / math.comb(n, 3)


def structural_energy(g: "nx.Graph[int]") -> float:
    """Structural-balance energy E_delta via common neighbours (spec §3.2).

    Same value as :func:`exact_structural_energy`, validated against it in the
    tests. Each triangle ``{u, v, w}`` is counted once, from its edge ``(u, v)``
    with ``w > max(u, v)``. Cost O(sum over edges of min(deg u, deg v)).

    Args:
        g: Signed graph satisfying the Phase 1 contract.

    Returns:
        ``E_delta`` in ``[-1, 1]``; ``0.0`` when ``n < 3``.

    Raises:
        TypeError: ``g`` violates the Phase 1 contract's types.
        ValueError: ``g`` violates the Phase 1 contract's values.
    """
    validate_signed_graph(g)
    n = g.number_of_nodes()
    if n < 3:
        return 0.0
    adj = g.adj
    triad_sum = 0
    for u, v, data in g.edges(data=True):
        top = max(u, v)
        for w in _common_neighbors(g, u, v):
            if w > top:
                triad_sum += data["sign"] * adj[u][w]["sign"] * adj[v][w]["sign"]
    return -triad_sum / math.comb(n, 3)


def _common_neighbors(g: "nx.Graph[int]", u: int, v: int) -> list[int]:
    """Neighbours shared by ``u`` and ``v``, in O(min(deg u, deg v))."""
    small, large = sorted((g.adj[u], g.adj[v]), key=len)
    return [w for w in small if w in large]
