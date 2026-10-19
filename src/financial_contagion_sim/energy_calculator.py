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

Local flip differences (spec §3.3). Flipping an existing edge ``A_ij -> -A_ij``
changes only the triads ``{i, j, k}``; each product goes from
``A_ij A_jk A_ki`` to ``-A_ij A_jk A_ki``, so::

    dE_delta = -1/C(n,3) * sum_k (-2 A_ij_old A_ik A_jk)
             = +2 A_ij_old / C(n,3) * sum_k A_ik A_jk
             = -2 A_ij_new / C(n,3) * sum_k A_ik A_jk

The spec's ``-2 A_ij(t'') / C(n,3) * sum A_ik(t') A_jk(t')`` is the last line,
reading ``t''`` as after the flip and ``t'`` as before it (the other edges do
not change). Only common neighbours ``k`` of ``i`` and ``j`` contribute.
With node states fixed, only the pair ``(i, j)`` changes in ``E_p``::

    dE_p = -2 * c_ij * A_ij_old / C(n, 2)

where ``c_ij = 1`` iff exactly one endpoint is I. The total is
``dE = alpha * dE_delta + (1 - alpha) * dE_p``, which is what acceptance uses.

Cost: ``dE_delta`` is O(|N(i) ∩ N(j)|) to sum, plus O(min(deg i, deg j)) to
find the common neighbours, which is O(n) on dense graphs; ``dE_p`` is O(1).
The total is therefore not O(1) in general, despite the title of spec §3.3:
O(1) needs bounded degree or a maintained common-neighbour cache, which is not
implemented.

Ties (spec §3.3 accepts ``dE == 0`` with probability 0.5). Both numerators are
integers and a float ``alpha`` converts exactly to a ``Fraction``, so
``delta_total_energy(..., exact=True)`` returns the exact ``Fraction`` and its
zero test needs no tolerance, at any ``n``. The float result may round a true
zero to a tiny non-zero value; pass the exact value to :func:`accept_flip`
when ties matter.
"""

import math
from fractions import Fraction
from typing import Literal, overload

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim.graph_builder import (
    _as_generator,
    _check_int,
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


def delta_structural_energy(g: "nx.Graph[int]", i: int, j: int) -> float:
    """Change in E_delta if the existing edge ``(i, j)`` flips sign (spec §3.3).

    ``dE_delta = +2 * A_ij_old / C(n, 3) * sum over common neighbours k of
    A_ik * A_jk``; see the module docstring for the derivation. The graph is
    not modified, and only ``(i, j)`` is checked, not the whole graph.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        i: One endpoint of an existing edge.
        j: The other endpoint.

    Returns:
        ``E_delta(after) - E_delta(before)``; ``0.0`` when ``n < 3``. Cost
        O(min(deg i, deg j)).

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: ``(i, j)`` is not an edge of ``g``.
    """
    a_old = _edge_sign(g, i, j)
    n = g.number_of_nodes()
    if n < 3:
        return 0.0
    return _triad_flip_numerator(g, i, j, a_old) / math.comb(n, 3)


def delta_spreading_energy(
    g: "nx.Graph[int]", states: StateArray, i: int, j: int
) -> float:
    """Change in E_p if the existing edge ``(i, j)`` flips sign (states fixed).

    ``dE_p = -2 * c_ij * A_ij_old / C(n, 2)``, with ``c_ij = 1`` iff exactly
    one endpoint is infected. Only ``states[i]`` and ``states[j]`` are checked.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: ``numpy.int8`` state array of length ``n``.
        i: One endpoint of an existing edge.
        j: The other endpoint.

    Returns:
        ``E_p(after) - E_p(before)``. Cost O(1).

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    a_old = _edge_sign(g, i, j)
    n = g.number_of_nodes()
    _check_endpoint_states(states, n, i, j)
    return _spread_flip_numerator(states, i, j, a_old) / math.comb(n, 2)


@overload
def delta_total_energy(
    g: "nx.Graph[int]",
    states: StateArray,
    i: int,
    j: int,
    alpha: float,
    *,
    exact: Literal[False] = False,
) -> float: ...


@overload
def delta_total_energy(
    g: "nx.Graph[int]",
    states: StateArray,
    i: int,
    j: int,
    alpha: float,
    *,
    exact: Literal[True],
) -> Fraction: ...


def delta_total_energy(
    g: "nx.Graph[int]",
    states: StateArray,
    i: int,
    j: int,
    alpha: float,
    *,
    exact: bool = False,
) -> float | Fraction:
    """Change in ``E = alpha * E_delta + (1 - alpha) * E_p`` for flipping ``(i, j)``.

    This is the energy difference used for acceptance (spec §3.3, applied to
    the full energy of §3.2).

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: ``numpy.int8`` state array of length ``n``.
        i: One endpoint of an existing edge.
        j: The other endpoint.
        alpha: Weight in ``[0, 1]``.
        exact: If true, return the exact ``Fraction`` (its ``== 0`` test is
            exact at any ``n``); otherwise a float.

    Returns:
        ``E(after) - E(before)``. Cost O(min(deg i, deg j)).

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    _check_unit_interval("alpha", alpha)
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {exact!r}")
    a_old = _edge_sign(g, i, j)
    n = g.number_of_nodes()
    _check_endpoint_states(states, n, i, j)
    k_delta = _triad_flip_numerator(g, i, j, a_old)
    k_p = _spread_flip_numerator(states, i, j, a_old)
    c3, c2 = math.comb(n, 3), math.comb(n, 2)
    if exact:
        weight = Fraction(alpha)
        structural = Fraction(k_delta, c3) if c3 else Fraction(0)
        return weight * structural + (1 - weight) * Fraction(k_p, c2)
    structural_float = k_delta / c3 if c3 else 0.0
    return alpha * structural_float + (1 - alpha) * (k_p / c2)


def accept_flip(delta_energy: float | Fraction, rng: np.random.Generator | int) -> bool:
    """Accept or reject an edge-sign flip (spec §3.3).

    ``dE < 0`` is always accepted, ``dE > 0`` always rejected, and ``dE == 0``
    accepted with probability 0.5. There is no temperature. ``rng`` is used
    only for ties. For a reliable tie test pass the ``Fraction`` from
    ``delta_total_energy(..., exact=True)``; a float is compared exactly.

    Args:
        delta_energy: The flip's energy change, a real number or ``Fraction``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Returns:
        Whether the flip is accepted.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: ``delta_energy`` is NaN.
    """
    if isinstance(delta_energy, bool) or not isinstance(
        delta_energy, (int, float, Fraction)
    ):
        raise TypeError(
            "delta_energy must be a real number or Fraction, "
            f"got {delta_energy!r} ({type(delta_energy).__name__})"
        )
    if isinstance(delta_energy, float) and math.isnan(delta_energy):
        raise ValueError("delta_energy must not be NaN")
    generator = _as_generator(rng)
    if delta_energy < 0:
        return True
    if delta_energy > 0:
        return False
    return bool(generator.random() < 0.5)


def _edge_sign(g: "nx.Graph[int]", i: int, j: int) -> int:
    """Validate that ``(i, j)`` is an edge of a simple graph; return its sign."""
    if not isinstance(g, nx.Graph) or g.is_directed() or g.is_multigraph():
        raise TypeError(
            f"g must be an undirected simple networkx.Graph, got {type(g).__name__}"
        )
    _check_int("i", i, minimum=0)
    _check_int("j", j, minimum=0)
    if i == j:
        raise ValueError(f"i and j must differ, got {i}")
    if not g.has_edge(i, j):
        raise ValueError(f"({i}, {j}) is not an edge; only existing edges flip")
    sign = g.adj[i][j]["sign"]
    if type(sign) is not int or sign not in (1, -1):
        raise ValueError(f"edge ({i}, {j}) has invalid sign {sign!r}")
    return sign


def _check_endpoint_states(states: object, n: int, i: int, j: int) -> None:
    if not isinstance(states, np.ndarray):
        raise TypeError(f"states must be a numpy.ndarray, got {type(states).__name__}")
    if states.dtype != np.int8:
        raise TypeError(f"states must have dtype int8, got {states.dtype}")
    if states.shape != (n,):
        raise ValueError(f"states must have shape ({n},), got {states.shape}")
    for node in (i, j):
        if int(states[node]) not in (1, 0, -1):
            raise ValueError(f"states[{node}] must be in {{1, 0, -1}}")


def _triad_flip_numerator(g: "nx.Graph[int]", i: int, j: int, a_old: int) -> int:
    """``2 * A_ij_old * sum_k A_ik * A_jk``: ``dE_delta`` times ``C(n, 3)``."""
    adj = g.adj
    total: int = sum(
        adj[i][k]["sign"] * adj[j][k]["sign"] for k in _common_neighbors(g, i, j)
    )
    return 2 * a_old * total


def _spread_flip_numerator(states: StateArray, i: int, j: int, a_old: int) -> int:
    """``-2 * c_ij * A_ij_old``: ``dE_p`` times ``C(n, 2)``."""
    active = (states[i] == _INFECTED) != (states[j] == _INFECTED)
    return -2 * a_old if active else 0


def _common_neighbors(g: "nx.Graph[int]", u: int, v: int) -> list[int]:
    """Neighbours shared by ``u`` and ``v``, in O(min(deg u, deg v))."""
    small, large = sorted((g.adj[u], g.adj[v]), key=len)
    return [w for w in small if w in large]
