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
"""

import networkx as nx

_SIGN = "sign"
_VALID_SIGNS = (1, -1)


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
