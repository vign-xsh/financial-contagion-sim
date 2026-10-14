"""SAIS contagion dynamics (spec §4; governing rates in §3.1).

Responsibility: CTMC transition rates and Gillespie state-jump logic for the
SAIS (Susceptible, Alert, Infected) process on a signed financial network
built by :mod:`financial_contagion_sim.graph_builder`.

States (spec §3.1) are stored as a ``numpy.int8`` array ``states`` indexed by
node, which works because Phase 1 nodes are exactly ``0 .. n-1``:
``State.SUSCEPTIBLE = 1``, ``State.ALERT = 0``, ``State.INFECTED = -1``.

Transitions (spec §3.1). There are exactly four; every rate is in 1/time:

- S -> I (default cascade): ``beta * N_i``
- A -> I (attenuated default): ``beta_a * N_i``, with ``0 <= beta_a < beta``
- S -> A (precautionary hedging): ``kappa * N_i``
- I -> S (recapitalization / bailout): ``delta``

``N_i`` is the number of infected neighbours of ``i`` joined to it by a
positive (+1) edge; negative edges contribute nothing. The "Recapitalized"
arrow in the §3.1 diagram is read as the I -> S transition above, so there is
no A -> S or I -> A transition.

Rate arrays have shape ``(n, 3)``: column ``k`` is the rate at which each node
jumps to state ``TARGETS[k]``.
"""

import enum
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim.graph_builder import (
    _as_generator,
    _check_int,
    _check_unit_interval,
    validate_signed_graph,
)


class State(enum.IntEnum):
    """Node state codes (spec §3.1)."""

    SUSCEPTIBLE = 1
    ALERT = 0
    INFECTED = -1


TARGETS = (State.SUSCEPTIBLE, State.ALERT, State.INFECTED)
"""Column ``k`` of a rate array is the rate of jumping to ``TARGETS[k]``."""

_S, _A, _I = int(State.SUSCEPTIBLE), int(State.ALERT), int(State.INFECTED)
_TO_S, _TO_A, _TO_I = 0, 1, 2

StateArray = npt.NDArray[np.int8]
CountArray = npt.NDArray[np.int64]
RateArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SAISParams:
    """SAIS rate parameters (spec §3.1), each in 1/time.

    Attributes:
        beta: S -> I rate per infected positive neighbour; ``beta > 0``.
        beta_a: A -> I rate per infected positive neighbour;
            ``0 <= beta_a < beta``.
        kappa: S -> A rate per infected positive neighbour; ``kappa >= 0``.
        delta: I -> S recapitalization rate; ``delta >= 0``.

    Raises:
        TypeError: a parameter is not a real number (``bool`` is rejected).
        ValueError: a parameter is not finite or violates its bound.
    """

    beta: float
    beta_a: float
    kappa: float
    delta: float

    def __post_init__(self) -> None:
        for name in ("beta", "beta_a", "kappa", "delta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a real number, "
                    f"got {value!r} ({type(value).__name__})"
                )
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta!r}")
        if not 0 <= self.beta_a < self.beta:
            raise ValueError(
                "beta_a must satisfy 0 <= beta_a < beta, "
                f"got beta_a={self.beta_a!r}, beta={self.beta!r}"
            )
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa!r}")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta!r}")


@dataclass(frozen=True)
class InitialState:
    """Result of :func:`make_initial_state`.

    Attributes:
        states: ``numpy.int8`` state array of length ``n``.
        requested_infected_fraction: the ``initial_infected_fraction`` asked for.
        realized_infected_fraction: the fraction of nodes actually infected.
    """

    states: StateArray
    requested_infected_fraction: float
    realized_infected_fraction: float


def make_initial_state(
    n: int, initial_infected_fraction: float, rng: np.random.Generator | int
) -> InitialState:
    """Build an initial state array from the initial infected fraction (rho_0, §5).

    Exactly ``round(initial_infected_fraction * n)`` nodes are infected (Python
    ``round`` on the float product, as for r0 in Phase 1), chosen at random by
    ``rng``. Every other node is susceptible; no node starts in A, because the
    spec does not give an initial A fraction.

    Args:
        n: Number of nodes, ``n >= 1``.
        initial_infected_fraction: Requested fraction of infected nodes, in
            ``[0, 1]``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Returns:
        The state array with the requested and realized infected fractions.

    Raises:
        TypeError: a parameter has the wrong type.
        ValueError: a parameter is out of range.
    """
    _check_int("n", n, minimum=1)
    _check_unit_interval("initial_infected_fraction", initial_infected_fraction)
    generator = _as_generator(rng)
    n_infected = round(initial_infected_fraction * n)
    states = np.full(n, _S, dtype=np.int8)
    states[generator.permutation(n)[:n_infected]] = _I
    return InitialState(states, initial_infected_fraction, n_infected / n)


def infected_positive_neighbors(g: "nx.Graph[int]", states: StateArray) -> CountArray:
    """Count, for every node, its infected neighbours on positive edges (``N_i``).

    This is the full O(n + m) recomputation. :func:`run_gillespie` builds it
    once and then updates it incrementally.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: State array of length ``n``.

    Returns:
        ``numpy.int64`` array of length ``n``.

    Raises:
        TypeError: ``g`` or ``states`` has the wrong type.
        ValueError: ``g`` or ``states`` is invalid.
    """
    validate_signed_graph(g)
    _check_states(states, g.number_of_nodes())
    return _count_infected_positive_neighbors(g, states)


def compute_rates(
    states: StateArray, n_pos: CountArray, params: SAISParams
) -> RateArray:
    """Return the per-node transition rates (spec §3.1), in 1/time.

    Args:
        states: State array of length ``n``.
        n_pos: ``N_i`` for every node, as from :func:`infected_positive_neighbors`.
        params: SAIS rate parameters.

    Returns:
        ``(n, 3)`` float array; column ``k`` is the rate of jumping to
        ``TARGETS[k]``. Entries are non-negative rates, not probabilities, and
        are not bounded by 1. Cost O(n).

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    if not isinstance(states, np.ndarray):
        raise TypeError(f"states must be a numpy.ndarray, got {type(states).__name__}")
    n = states.shape[0] if states.ndim == 1 else -1
    _check_states(states, n)
    _check_counts(n_pos, n)
    _check_params(params)
    return _compute_rates(states, n_pos, params)


def refresh_after_edge_flip(
    n_pos: CountArray, states: StateArray, i: int, j: int, new_sign: int
) -> None:
    """Update ``n_pos`` in place after edge ``(i, j)`` changed sign to ``new_sign``.

    Hook for the co-evolution phase: call it after flipping the edge's sign in
    the graph. Node states are unchanged. If ``j`` is infected, ``N_i`` changes
    by +1 when the edge becomes positive and by -1 when it becomes negative, and
    symmetrically for ``N_j``. Cost O(1).

    Args:
        n_pos: ``N_i`` array to update in place.
        states: State array of length ``n``.
        i: One endpoint of the flipped edge.
        j: The other endpoint, ``j != i``.
        new_sign: The edge's sign after the flip, ``+1`` or ``-1``.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is out of range.
    """
    if type(new_sign) is not int:
        raise TypeError(
            f"new_sign must be a Python int, got {new_sign!r} "
            f"({type(new_sign).__name__})"
        )
    if new_sign not in (1, -1):
        raise ValueError(f"new_sign must be +1 or -1, got {new_sign}")
    n = len(n_pos)
    if len(states) != n:
        raise ValueError(
            f"states and n_pos must have the same length, got {len(states)} and {n}"
        )
    for name, node in (("i", i), ("j", j)):
        _check_int(name, node, minimum=0)
        if node >= n:
            raise ValueError(f"{name} must be < n={n}, got {node}")
    if i == j:
        raise ValueError(f"i and j must differ, got {i}")
    change = 1 if new_sign == 1 else -1
    if states[j] == _I:
        n_pos[i] += change
    if states[i] == _I:
        n_pos[j] += change


def _count_infected_positive_neighbors(
    g: "nx.Graph[int]", states: StateArray
) -> CountArray:
    counts = np.zeros(g.number_of_nodes(), dtype=np.int64)
    for u, v, data in g.edges(data=True):
        if data["sign"] == 1:
            if states[u] == _I:
                counts[v] += 1
            if states[v] == _I:
                counts[u] += 1
    return counts


def _compute_rates(
    states: StateArray, n_pos: CountArray, params: SAISParams
) -> RateArray:
    rates = np.zeros((states.shape[0], 3), dtype=np.float64)
    neighbours = n_pos.astype(np.float64)
    susceptible = states == _S
    alert = states == _A
    rates[susceptible, _TO_I] = params.beta * neighbours[susceptible]
    rates[susceptible, _TO_A] = params.kappa * neighbours[susceptible]
    rates[alert, _TO_I] = params.beta_a * neighbours[alert]
    rates[states == _I, _TO_S] = params.delta
    return rates


def _check_states(states: object, n: int) -> None:
    if not isinstance(states, np.ndarray):
        raise TypeError(f"states must be a numpy.ndarray, got {type(states).__name__}")
    if states.dtype != np.int8:
        raise TypeError(f"states must have dtype int8, got {states.dtype}")
    if states.shape != (n,):
        raise ValueError(f"states must have shape ({n},), got {states.shape}")
    bad = sorted({int(x) for x in np.unique(states)} - {_S, _A, _I})
    if bad:
        raise ValueError(f"states must be in {{1, 0, -1}}, found {bad}")


def _check_counts(n_pos: object, n: int) -> None:
    if not isinstance(n_pos, np.ndarray):
        raise TypeError(f"n_pos must be a numpy.ndarray, got {type(n_pos).__name__}")
    if n_pos.dtype != np.int64:
        raise TypeError(f"n_pos must have dtype int64, got {n_pos.dtype}")
    if n_pos.shape != (n,):
        raise ValueError(f"n_pos must have shape ({n},), got {n_pos.shape}")
    if n_pos.size and int(n_pos.min()) < 0:
        raise ValueError("n_pos must be non-negative")


def _check_params(params: object) -> None:
    if not isinstance(params, SAISParams):
        raise TypeError(f"params must be SAISParams, got {type(params).__name__}")
