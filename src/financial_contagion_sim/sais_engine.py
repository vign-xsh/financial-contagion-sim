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

Gillespie simulation (direct method). With total rate ``R``, the waiting time
is ``dt ~ Exp(R)`` (mean ``1/R``) and the event is chosen with probability
``rate_k / R``. If ``t + dt`` would pass ``t_max``, the clock stops at
``t_max`` and the event is not applied; by memorylessness that is exact. If
``R == 0`` (no transition possible) the chain is absorbed and the clock jumps
to ``t_max``. No fixed time step is used anywhere.

Cost per event: O(n) to rebuild the rates and scan them for the event, plus
O(deg(v)) to update ``N`` around the node ``v`` that changed state. ``N`` is
never recomputed globally inside a run. A Fenwick tree over the rates would
make selection O(log n); it is not implemented.
"""

import enum
import math
from dataclasses import dataclass
from typing import Literal

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
_ALLOWED_TRANSITIONS = frozenset({(_S, _I), (_A, _I), (_S, _A), (_I, _S)})

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


@dataclass(frozen=True)
class Event:
    """One applied state jump.

    Attributes:
        time: Time at which the jump happened.
        node: The node that changed state.
        old_state: Its state before the jump.
        new_state: Its state after the jump.
    """

    time: float
    node: int
    old_state: State
    new_state: State


@dataclass(frozen=True)
class StepResult:
    """Outcome of :func:`gillespie_step`.

    Attributes:
        reason: ``"event"`` if a jump was applied, ``"t_max"`` if the next jump
            would fall after ``t_max``, or ``"absorbed"`` if the total rate is 0.
        time: Clock after the step: the event time, or ``t_max`` otherwise.
        event: The applied jump, or ``None``.
    """

    reason: Literal["event", "t_max", "absorbed"]
    time: float
    event: Event | None


@dataclass(frozen=True)
class SAISResult:
    """Outcome of :func:`run_gillespie`.

    Attributes:
        states: Final state array (a copy; the input is not modified).
        time: Final clock: ``t_max`` for ``"t_max"`` and ``"absorbed"``, the
            last event time for ``"max_events"``.
        n_events: Number of jumps applied.
        termination_reason: ``"t_max"``, ``"absorbed"`` or ``"max_events"``.
        events: The applied jumps in order if ``record_events`` was set,
            otherwise ``None``.
    """

    states: StateArray
    time: float
    n_events: int
    termination_reason: Literal["t_max", "absorbed", "max_events"]
    events: tuple[Event, ...] | None


def sample_waiting_time(total_rate: float, rng: np.random.Generator | int) -> float:
    """Draw a waiting time ``dt ~ Exp(total_rate)``, mean ``1 / total_rate``.

    Args:
        total_rate: Total transition rate ``R`` in 1/time; finite and ``> 0``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Returns:
        The waiting time, in time units.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: ``total_rate`` is not finite and positive.
    """
    if not isinstance(total_rate, (int, float)) or isinstance(total_rate, bool):
        raise TypeError(
            f"total_rate must be a real number, got {total_rate!r} "
            f"({type(total_rate).__name__})"
        )
    if not (math.isfinite(total_rate) and total_rate > 0):
        raise ValueError(f"total_rate must be finite and > 0, got {total_rate!r}")
    return _sample_waiting_time(float(total_rate), _as_generator(rng))


def select_event(rates: RateArray, rng: np.random.Generator | int) -> tuple[int, State]:
    """Choose the next jump with probability ``rate_k / R`` (direct method).

    Args:
        rates: ``(n, 3)`` rate array, as from :func:`compute_rates`.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.

    Returns:
        ``(node, new_state)``. A zero-rate jump is never chosen. Cost O(n).

    Raises:
        TypeError: ``rates`` is not a float array.
        ValueError: ``rates`` has the wrong shape, a negative or non-finite
            entry, or no positive entry.
    """
    if not isinstance(rates, np.ndarray) or not np.issubdtype(rates.dtype, np.floating):
        raise TypeError("rates must be a floating-point numpy.ndarray")
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(f"rates must have shape (n, 3), got {rates.shape}")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rates must be finite and non-negative")
    if not np.any(rates > 0):
        raise ValueError("no transition is possible: every rate is 0")
    return _select_event(rates, _as_generator(rng))


def gillespie_step(
    g: "nx.Graph[int]",
    states: StateArray,
    n_pos: CountArray,
    params: SAISParams,
    rng: np.random.Generator | int,
    *,
    time: float = 0.0,
    t_max: float = math.inf,
) -> StepResult:
    """Advance the chain by at most one jump, updating ``states`` and ``n_pos``.

    Draws ``dt ~ Exp(R)``. If ``time + dt > t_max`` nothing is applied and the
    result reports ``"t_max"``; if ``R == 0`` it reports ``"absorbed"``. Cost
    O(n) plus O(deg(v)) for the node ``v`` that changes. The graph is not
    validated here (it is O(m)); :func:`run_gillespie` validates it once.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: State array, updated in place.
        n_pos: ``N_i`` array consistent with ``g`` and ``states``, updated in
            place.
        params: SAIS rate parameters.
        rng: ``numpy.random.Generator`` (or ``int`` seed; an int re-seeds on
            every call).
        time: Current clock, finite and ``>= 0``.
        t_max: Time horizon, ``>= time``; may be ``math.inf``.

    Returns:
        The step outcome.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    n = g.number_of_nodes()
    _check_states(states, n)
    _check_counts(n_pos, n)
    _check_params(params)
    for name, value in (("time", time), ("t_max", t_max)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number, got {value!r}")
    if not (math.isfinite(time) and time >= 0):
        raise ValueError(f"time must be finite and >= 0, got {time!r}")
    if math.isnan(t_max) or t_max < time:
        raise ValueError(f"t_max must be >= time, got t_max={t_max!r}, time={time!r}")
    return _gillespie_step(
        g, states, n_pos, params, _as_generator(rng), float(time), float(t_max)
    )


def run_gillespie(
    g: "nx.Graph[int]",
    states: StateArray,
    params: SAISParams,
    t_max: float,
    rng: np.random.Generator | int,
    *,
    max_events: int | None = None,
    record_events: bool = False,
) -> SAISResult:
    """Simulate the SAIS chain on a fixed graph from time 0 to ``t_max``.

    The graph's signs are held fixed; edge dynamics belong to the co-evolution
    phase. ``N`` is built once in O(n + m) and then updated per event.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: Initial state array; it is copied, not modified.
        params: SAIS rate parameters.
        t_max: Time horizon, finite and ``>= 0``.
        rng: ``numpy.random.Generator`` or non-negative ``int`` seed.
        max_events: Optional cap on the number of jumps, ``>= 0``.
        record_events: If true, keep every applied jump in ``result.events``.

    Returns:
        The final state, clock, event count and termination reason.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
    """
    validate_signed_graph(g)
    n = g.number_of_nodes()
    _check_states(states, n)
    _check_params(params)
    if not isinstance(t_max, (int, float)) or isinstance(t_max, bool):
        raise TypeError(f"t_max must be a real number, got {t_max!r}")
    if not (math.isfinite(t_max) and t_max >= 0):
        raise ValueError(f"t_max must be finite and >= 0, got {t_max!r}")
    if max_events is not None:
        _check_int("max_events", max_events, minimum=0)
    if not isinstance(record_events, bool):
        raise TypeError(f"record_events must be a bool, got {record_events!r}")
    generator = _as_generator(rng)

    current = states.copy()
    n_pos = _count_infected_positive_neighbors(g, current)
    events: list[Event] | None = [] if record_events else None
    time, n_events = 0.0, 0
    while True:
        if max_events is not None and n_events >= max_events:
            reason: Literal["t_max", "absorbed", "max_events"] = "max_events"
            break
        step = _gillespie_step(g, current, n_pos, params, generator, time, float(t_max))
        time = step.time
        if step.event is None:
            reason = "absorbed" if step.reason == "absorbed" else "t_max"
            break
        n_events += 1
        if events is not None:
            events.append(step.event)
    return SAISResult(
        current, time, n_events, reason, None if events is None else tuple(events)
    )


def apply_sais_event(
    g: "nx.Graph[int]",
    states: StateArray,
    n_pos: CountArray,
    node: int,
    new_state: int,
) -> State:
    """Apply one SAIS jump in place and update ``n_pos`` around ``node``.

    This is the update :func:`gillespie_step` performs after choosing an
    event, exposed for callers that run their own clock (the co-evolution
    simulator). Only the four spec §3.1 transitions are allowed. Cost
    O(deg(node)); the graph itself is not validated.

    Args:
        g: Signed graph satisfying the Phase 1 contract.
        states: State array, updated in place.
        n_pos: ``N_i`` array consistent with ``g`` and ``states``, updated in
            place.
        node: The node that changes state.
        new_state: Its new state; ``(old, new)`` must be S->I, A->I, S->A or
            I->S.

    Returns:
        The node's previous state.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid or the transition is not allowed.
    """
    n = g.number_of_nodes()
    for name, array, dtype in (("states", states, np.int8), ("n_pos", n_pos, np.int64)):
        if not isinstance(array, np.ndarray) or array.dtype != dtype:
            raise TypeError(f"{name} must be a numpy.ndarray of dtype {dtype.__name__}")
        if array.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    _check_int("node", node, minimum=0)
    if node >= n:
        raise ValueError(f"node must be < n={n}, got {node}")
    if not isinstance(new_state, int) or isinstance(new_state, bool):
        raise TypeError(f"new_state must be an int, got {new_state!r}")
    old_state = int(states[node])
    if (old_state, int(new_state)) not in _ALLOWED_TRANSITIONS:
        raise ValueError(
            f"transition {old_state} -> {int(new_state)} is not one of the four "
            "SAIS transitions (S->I, A->I, S->A, I->S)"
        )
    _apply_event(g, states, n_pos, node, State(int(new_state)))
    return State(old_state)


def _sample_waiting_time(total_rate: float, rng: np.random.Generator) -> float:
    return float(rng.exponential(1.0 / total_rate))


def _select_event(rates: RateArray, rng: np.random.Generator) -> tuple[int, State]:
    flat = rates.ravel()
    cumulative = np.cumsum(flat)
    u = rng.random() * cumulative[-1]
    index = int(np.searchsorted(cumulative, u, side="right"))
    if index >= flat.size:  # u rounded up to the total; take the last live entry
        index = int(np.flatnonzero(flat)[-1])
    node, column = divmod(index, 3)
    return node, TARGETS[column]


def _gillespie_step(
    g: "nx.Graph[int]",
    states: StateArray,
    n_pos: CountArray,
    params: SAISParams,
    rng: np.random.Generator,
    time: float,
    t_max: float,
) -> StepResult:
    rates = _compute_rates(states, n_pos, params)
    total = float(rates.sum())
    if total <= 0.0:  # sums of products of finite non-negatives: exact test
        return StepResult("absorbed", t_max, None)
    dt = _sample_waiting_time(total, rng)
    if time + dt > t_max:
        return StepResult("t_max", t_max, None)
    node, new_state = _select_event(rates, rng)
    old_state = State(int(states[node]))
    _apply_event(g, states, n_pos, node, new_state)
    event = Event(time + dt, node, old_state, new_state)
    return StepResult("event", event.time, event)


def _apply_event(
    g: "nx.Graph[int]",
    states: StateArray,
    n_pos: CountArray,
    node: int,
    new_state: State,
) -> None:
    was_infected = states[node] == _I
    states[node] = new_state
    is_infected = new_state == _I
    if was_infected != is_infected:
        change = 1 if is_infected else -1
        for neighbour, data in g.adj[node].items():
            if data["sign"] == 1:
                n_pos[neighbour] += change


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
