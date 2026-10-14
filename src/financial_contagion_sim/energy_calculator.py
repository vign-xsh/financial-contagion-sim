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

Pairwise spreading energy (spec §3.2). For an edge ``(i, j)``, ``E_ij^p`` is
``+1`` when an S or A node meets an I node over a positive edge (S+I, A+I),
``-1`` over a negative edge (S-I, A-I), and 0 otherwise (S±S, I±I, A±A, S±A);
non-edges are 0. That is ``E_ij^p = A_ij`` when exactly one endpoint is I. The
spec does not say how the pair terms are aggregated; the approved rule is::

    E_p = (sum over edges of E_ij^p) / C(n, 2)

a fixed denominator like E_delta's ``C(n, 3)``, so ``E_p`` lies in ``[-1, 1]``
and the local flip difference is exact. ``E_p = 0.0`` for ``n < 2``.

Total energy (spec §3.2): ``E = alpha * E_delta + (1 - alpha) * E_p`` with
``alpha`` in ``[0, 1]``, so ``E`` also lies in ``[-1, 1]``.
"""

import math

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim.graph_builder import (
    _check_unit_interval,
    validate_signed_graph,
)
from financial_contagion_sim.sais_engine import StateArray, _check_states

AdjacencyMatrix = npt.NDArray[np.int64]
_INFECTED = -1


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


def spreading_energy(g: "nx.Graph[int]", states: StateArray) -> float:
    """Pairwise spreading energy E_p (spec §3.2; aggregation approved in Phase 2).

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: ``numpy.int8`` state array of length ``n`` (see
            :mod:`financial_contagion_sim.sais_engine`).

    Returns:
        ``E_p = (sum over edges of E_ij^p) / C(n, 2)``, in ``[-1, 1]``;
        ``0.0`` when ``n < 2``. Cost O(n + m).

    Raises:
        TypeError: ``g`` or ``states`` has the wrong type.
        ValueError: ``g`` or ``states`` is invalid.
    """
    validate_signed_graph(g)
    n = g.number_of_nodes()
    _check_states(states, n)
    if n < 2:
        return 0.0
    pair_sum = 0
    for u, v, data in g.edges(data=True):
        if (states[u] == _INFECTED) != (states[v] == _INFECTED):
            pair_sum += data["sign"]
    return pair_sum / math.comb(n, 2)


def total_energy(g: "nx.Graph[int]", states: StateArray, alpha: float) -> float:
    """Total network energy ``E = alpha * E_delta + (1 - alpha) * E_p`` (spec §3.2).

    Uses the sparse :func:`structural_energy` engine.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: ``numpy.int8`` state array of length ``n``.
        alpha: Weight of structural balance against epidemic control, in
            ``[0, 1]``.

    Returns:
        ``E`` in ``[-1, 1]``. ``alpha = 1`` gives ``E_delta`` and ``alpha = 0``
        gives ``E_p``.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    _check_unit_interval("alpha", alpha)
    return alpha * structural_energy(g) + (1 - alpha) * spreading_energy(g, states)


def _common_neighbors(g: "nx.Graph[int]", u: int, v: int) -> list[int]:
    """Neighbours shared by ``u`` and ``v``, in O(min(deg u, deg v))."""
    small, large = sorted((g.adj[u], g.adj[v]), key=len)
    return [w for w in small if w in large]
