"""Signed financial network construction (spec §4; inputs described in §5, step 1).

Responsibility: scale-free (Barabási–Albert), Erdős–Rényi, and
correlation-matrix graph loaders that produce the signed financial networks
every other module operates on.

Graph contract (relied on by later phases):

- A signed network is an undirected, simple ``networkx.Graph``: no
  ``DiGraph``, no multigraph, no self-loops.
- Nodes are exactly the Python ints ``0 .. n-1``, so later phases can use them
  directly as array indices. Isolated nodes are allowed.
- Every edge carries ``"sign"``, a Python ``int`` equal to ``+1`` (direct
  exposure) or ``-1`` (hedge or short position), per spec §1–§2. Signs may be
  changed in place; call :func:`validate_signed_graph` to recheck afterwards.
- Non-edges are simply absent. Treating them as ``A_ij = 0`` is the energy
  layer's concern (spec §3.2).
- Errors: ``TypeError`` for a value of the wrong type, ``ValueError`` for a
  value of the right type that is out of range.

Positive-link ratio r0 (spec §5, step 1):

- ``r0 = positive edges / m``, where ``m`` is the total number of edges. This is
  the same quantity as the friendly link density r(t) tracked in spec §5,
  step 2.
- Signs are assigned exactly: ``round(r0 * m)`` edges are positive (Python
  ``round``, i.e. round-half-even on the float product; ``m = 5, r0 = 0.5``
  gives 2). The positive edges are a seeded random subset of the edges sorted
  by ``(u, v)`` with ``u < v``.
- Generated graphs record ``g.graph["r0_requested"]`` and
  ``g.graph["r0_realized"]``. With no edges, ``r0`` cannot be realized and
  ``r0_realized`` is ``None``.

Randomness: generators take ``rng``, a ``numpy.random.Generator`` or a
non-negative ``int`` seed (passed to ``numpy.random.default_rng``). There is no
global seeding. One integer seed is drawn from ``rng`` for the networkx
generator, then the same ``rng`` draws the sign assignment, so the same seed
always gives the same graph, including its signs.
"""

import math

import networkx as nx
import numpy as np

_SIGN = "sign"
_VALID_SIGNS = (1, -1)
_NX_SEED_HIGH = 2**31 - 1


def validate_signed_graph(g: object) -> None:
    """Check that ``g`` satisfies the signed-graph contract; raise if not.

    Raises:
        TypeError: ``g`` is not an undirected simple ``networkx.Graph``, or a
            node or sign is not a Python ``int`` (``bool`` and NumPy integers
            are rejected).
        ValueError: the nodes are not exactly ``0 .. n-1``, there is a
            self-loop, or an edge has no sign or a sign other than ``+1``/``-1``.
    """
    if not isinstance(g, nx.Graph) or g.is_directed() or g.is_multigraph():
        raise TypeError(
            f"g must be an undirected simple networkx.Graph, got {type(g).__name__}"
        )

    for node in g.nodes:
        if type(node) is not int:
            raise TypeError(
                f"nodes must be Python ints, got {node!r} ({type(node).__name__})"
            )
    n = g.number_of_nodes()
    missing = sorted(set(range(n)) - set(g.nodes))
    if missing:
        raise ValueError(f"nodes must be exactly 0..{n - 1}; missing {missing}")

    loops = sorted(u for u, _ in nx.selfloop_edges(g))
    if loops:
        raise ValueError(f"self-loops are not allowed; found at nodes {loops}")

    for u, v, data in g.edges(data=True):
        if _SIGN not in data:
            raise ValueError(f"edge ({u}, {v}) has no {_SIGN!r} attribute")
        sign = data[_SIGN]
        if type(sign) is not int:
            raise TypeError(
                f"edge ({u}, {v}) sign must be a Python int, "
                f"got {sign!r} ({type(sign).__name__})"
            )
        if sign not in _VALID_SIGNS:
            raise ValueError(f"edge ({u}, {v}) sign must be +1 or -1, got {sign}")


def generate_barabasi_albert(
    n: int, m: int, r0: float, rng: np.random.Generator | int
) -> "nx.Graph[int]":
    """Generate a signed scale-free (Barabási–Albert) network (spec §5, step 1).

    Args:
        n: Number of nodes, ``n >= 2``.
        m: Edges attached from each new node to existing nodes, ``1 <= m < n``.
            Not to be confused with the total edge count used to define
            ``r0``; the networkx model yields ``m * (n - m)`` edges in total.
        r0: Requested positive-link ratio in ``[0, 1]``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Raises:
        TypeError: a parameter has the wrong type.
        ValueError: a parameter is out of range.
    """
    _check_int("n", n, minimum=1)
    _check_int("m", m, minimum=1)
    if m >= n:
        raise ValueError(f"m must satisfy 1 <= m < n, got m={m}, n={n}")
    _check_unit_interval("r0", r0)
    generator = _as_generator(rng)
    g = nx.barabasi_albert_graph(n, m, seed=_nx_seed(generator))
    _assign_signs(g, r0, generator)
    validate_signed_graph(g)
    return g


def generate_erdos_renyi(
    n: int, p: float, r0: float, rng: np.random.Generator | int
) -> "nx.Graph[int]":
    """Generate a signed Erdős–Rényi G(n, p) network (spec §5, step 1).

    Uses ``networkx.gnp_random_graph``.

    Args:
        n: Number of nodes, ``n >= 1``.
        p: Probability of each edge, in ``[0, 1]``.
        r0: Requested positive-link ratio in ``[0, 1]``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Raises:
        TypeError: a parameter has the wrong type.
        ValueError: a parameter is out of range.
    """
    _check_int("n", n, minimum=1)
    _check_unit_interval("p", p)
    _check_unit_interval("r0", r0)
    generator = _as_generator(rng)
    g: nx.Graph[int] = nx.gnp_random_graph(n, p, seed=_nx_seed(generator))
    _assign_signs(g, r0, generator)
    validate_signed_graph(g)
    return g


def _assign_signs(g: "nx.Graph[int]", r0: float, rng: np.random.Generator) -> None:
    """Make exactly ``round(r0 * m)`` edges positive and the rest negative."""
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges)
    m = len(edges)
    g.graph["r0_requested"] = r0
    if m == 0:
        g.graph["r0_realized"] = None
        return
    n_pos = round(r0 * m)
    positive = set(rng.permutation(m)[:n_pos].tolist())
    for index, (u, v) in enumerate(edges):
        g.edges[u, v][_SIGN] = 1 if index in positive else -1
    g.graph["r0_realized"] = n_pos / m


def _nx_seed(rng: np.random.Generator) -> int:
    """Draw the integer seed handed to a networkx generator."""
    return int(rng.integers(0, _NX_SEED_HIGH))


def _as_generator(rng: object) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, int) and not isinstance(rng, bool):
        if rng < 0:
            raise ValueError(f"rng seed must be a non-negative int, got {rng}")
        return np.random.default_rng(rng)
    raise TypeError(
        "rng must be a numpy.random.Generator or a non-negative int seed, "
        f"got {rng!r} ({type(rng).__name__})"
    )


def _check_int(name: str, value: object, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be an int, got {value!r} ({type(value).__name__})"
        )
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _check_unit_interval(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number, got {value!r} ({type(value).__name__})"
        )
    if math.isnan(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
