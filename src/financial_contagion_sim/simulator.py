"""Co-evolutionary Monte Carlo simulation (spec §4; pipeline in §5, step 2).

Responsibility: the co-evolutionary Monte Carlo simulation loop, in which
contagion state and network structure evolve together (spec §1), combining
the graph builder, the SAIS engine and the energy calculator.

Configuration (Phase 3 decisions A1-A6): a frozen :class:`SimulationConfig`
holds every physical parameter with no defaults, plus control options with
defaults. Values are validated by the Phase 1-2 validators where they exist.

Randomness (Phase 3 E1-E3): the seed (a non-negative ``int`` or a
``numpy.random.SeedSequence``) is turned into five independent child streams,
always in this order: ``graph``, ``initial_state``, ``clock`` (waiting times and
channel choice), ``sais_selection`` and ``edge_updates`` (edge choice and flip
acceptance). Each component draws only from its own stream, so changing one
component's settings does not shift another's draws. A caller's
``SeedSequence`` is never consumed: a fresh copy is spawned from. Results are
reproducible for the same configuration and seed on the same machine and
package versions; the versions are recorded in every result.

Co-evolution mechanics (Phase 3 B1-B9; the spec gives no rate or schedule):

- Only the signs of existing edges change (+1 <-> -1, spec §3.3); edges are
  never created, deleted or rewired.
- One joint continuous-time Markov chain with two channels: the SAIS jumps
  (Phase 2 rates, total ``R_sais``) and flip proposals, where every edge
  proposes at rate ``gamma`` regardless of node states, so ``R_net = gamma*m``.
  With ``R = R_sais + R_net``, the waiting time is ``dt ~ Exp(R)`` and the
  channel is SAIS with probability ``R_sais / R``. No fixed time step is used.
- A proposal picks an edge uniformly from the edges sorted by ``(u, v)``. Its
  exact energy change ``dE`` (full energy E, spec §3.2) is computed from the
  current node states; it is accepted if ``dE < 0``, with probability 0.5 if
  ``dE == 0`` (tested exactly with ``Fraction``), and rejected if ``dE > 0``
  (spec §3.3). A rejected proposal is still an event: it takes time and counts
  toward ``max_events``.
- After an accepted flip, the maintained sums and ``N_i`` are updated through
  the Phase 2 hooks, so the next SAIS rates already see the new sign.
- Modelling note: on sparse graphs most proposals have ``dE == 0`` exactly
  (no common neighbours and no infected endpoint), so those edges flip with
  probability 0.5 and their signs random-walk towards ``r = 0.5``.
- If the next event would fall after ``t_max`` it is not applied
  (memorylessness). ``R == 0`` (only possible when ``gamma * m == 0`` and no
  SAIS transition is possible) ends the run as ``"absorbed"``.

Exact bookkeeping (Phase 3 C2): the energy is kept as two integers, the triad
sum ``S3`` and the pair sum ``P``, with ``E_delta = -S3 / C(n, 3)`` and
``E_p = P / C(n, 2)``. They change by exact integers on every flip and every
SAIS jump, so there is no floating-point drift. ``N_i`` is updated in O(1) per
flip and O(deg v) per jump, never recomputed globally.

Recording (Phase 3 C3-C4): ``record="grid"`` samples the exact state at
``num_points`` evenly spaced times in ``[0, t_max]`` (right-continuous: the
value at a grid time includes events at that time). ``record="events"``
records t = 0 and the state after every event, including rejected proposals.
Recording never draws random numbers, so both modes produce the same
trajectory. If the run ends at ``t_max`` or is absorbed, the record extends to
``t_max``; if it stops on ``max_events`` or ``"sais_absorbed"``, the record ends
at the stop time.

Cost per event: a flip proposal costs O(min(deg u, deg v)) to evaluate, plus
O(1) to apply. A SAIS jump costs O(deg v) for the updates plus O(n) to rebuild
the cached rate array before the next event. Rates are rebuilt only after an
event that can change them (a jump, or an accepted flip touching an infected
node), so long stretches of proposals do not pay O(n) each.
"""

import math
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, get_args

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim import __version__
from financial_contagion_sim.energy_calculator import (
    accept_flip,
    energy_from_sums,
    flip_energy_numerators,
    spreading_pair_sum,
    state_change_pair_delta,
    structural_triad_sum,
)
from financial_contagion_sim.graph_builder import (
    _check_int,
    _check_unit_interval,
    generate_barabasi_albert,
    generate_erdos_renyi,
    validate_signed_graph,
)
from financial_contagion_sim.sais_engine import (
    SAISParams,
    State,
    StateArray,
    apply_sais_event,
    compute_rates,
    infected_positive_neighbors,
    make_initial_state,
    refresh_after_edge_flip,
    sample_waiting_time,
    select_event,
)

SCHEMA_VERSION = 1
"""Version of the :class:`SimulationResult` layout."""

Topology = Literal["barabasi_albert", "erdos_renyi"]
RecordMode = Literal["grid", "events"]
TerminationReason = Literal["t_max", "max_events", "absorbed", "sais_absorbed"]
Seed = int | np.random.SeedSequence

STREAM_NAMES = ("graph", "initial_state", "clock", "sais_selection", "edge_updates")
"""Order in which the five child random streams are spawned from the seed."""

TRANSITION_NAMES = ("S->I", "A->I", "S->A", "I->S")
"""Keys of :attr:`SimulationResult.n_sais_events`."""


@dataclass(frozen=True)
class SimulationConfig:
    """Full configuration of a synthetic-network run (spec §5).

    Physical parameters have no defaults. Exactly one of ``ba_m`` (for
    ``topology="barabasi_albert"``) or ``er_p`` (for ``"erdos_renyi"``) is set.

    Attributes:
        n: Number of institutions (nodes), ``>= 1``.
        topology: ``"barabasi_albert"`` or ``"erdos_renyi"`` (spec §5, step 1).
        sais_params: SAIS rates, in 1/time (spec §3.1).
        alpha: Energy weight in ``[0, 1]`` (spec §3.2).
        r0: Requested positive-link ratio in ``[0, 1]`` (spec §5).
        initial_infected_fraction: rho_0, the initial infected fraction in
            ``[0, 1]`` (spec §5).
        edge_update_rate: gamma, the flip-proposal rate per edge, in 1/time;
            finite and ``>= 0``.
        t_max: Time horizon; finite and ``>= 0``.
        seed: Non-negative ``int`` or ``numpy.random.SeedSequence``.
        ba_m: Edges attached per new node for Barabási–Albert, ``1 <= ba_m < n``.
        er_p: Edge probability for Erdős–Rényi, in ``[0, 1]``.
        max_events: Optional cap on all events (SAIS jumps and flip
            proposals), ``>= 0``.
        stop_on_sais_absorbed: Stop as soon as no SAIS transition is possible.
        record: ``"grid"`` (``num_points`` evenly spaced times) or ``"events"``
            (a record after every event).
        num_points: Grid size for ``record="grid"``, ``>= 2``.
        verify_every: If set (``>= 1``), recompute every maintained quantity
            from scratch every ``verify_every`` events and raise
            ``RuntimeError`` on any mismatch.

    Raises:
        TypeError: a field has the wrong type.
        ValueError: a field is out of range or the topology fields disagree.
    """

    n: int
    topology: Topology
    sais_params: SAISParams
    alpha: float
    r0: float
    initial_infected_fraction: float
    edge_update_rate: float
    t_max: float
    seed: Seed
    ba_m: int | None = None
    er_p: float | None = None
    max_events: int | None = None
    stop_on_sais_absorbed: bool = False
    record: RecordMode = "grid"
    num_points: int = 501
    verify_every: int | None = None

    def __post_init__(self) -> None:
        _check_int("n", self.n, minimum=1)
        if not isinstance(self.topology, str):
            raise TypeError(f"topology must be a str, got {self.topology!r}")
        if self.topology not in get_args(Topology):
            raise ValueError(
                f"topology must be one of {get_args(Topology)}, got {self.topology!r}"
            )
        if self.topology == "barabasi_albert":
            if self.ba_m is None or self.er_p is not None:
                raise ValueError(
                    "topology 'barabasi_albert' needs ba_m and no er_p, "
                    f"got ba_m={self.ba_m!r}, er_p={self.er_p!r}"
                )
            _check_int("ba_m", self.ba_m, minimum=1)
            if self.ba_m >= self.n:
                raise ValueError(
                    f"ba_m must satisfy 1 <= ba_m < n, got ba_m={self.ba_m}, n={self.n}"
                )
        else:
            if self.er_p is None or self.ba_m is not None:
                raise ValueError(
                    "topology 'erdos_renyi' needs er_p and no ba_m, "
                    f"got ba_m={self.ba_m!r}, er_p={self.er_p!r}"
                )
            _check_unit_interval("er_p", self.er_p)
        _check_unit_interval("r0", self.r0)
        _check_unit_interval(
            "initial_infected_fraction", self.initial_infected_fraction
        )
        _check_model_options(self.sais_params, self.alpha, self.edge_update_rate)
        _check_run_options(
            t_max=self.t_max,
            seed=self.seed,
            max_events=self.max_events,
            stop_on_sais_absorbed=self.stop_on_sais_absorbed,
            record=self.record,
            num_points=self.num_points,
            verify_every=self.verify_every,
        )


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of :func:`run_simulation` or :func:`simulate` (spec §5, step 2).

    All series have one entry per record (see ``record`` in
    :class:`SimulationConfig`) and are piecewise-constant samples of the
    exact event process: the value at ``times[k]`` is the state after every
    event at or before that time.

    Attributes:
        times: Record times, ``float64``.
        S: Susceptible counts, ``int64``.
        A: Alert counts, ``int64``.
        I: Infected counts, ``int64``.
        r: Friendly-link density r(t) = positive edges / m, ``float64``; NaN
            throughout when m = 0.
        E: Total energy E(G_t), ``float64`` (spec §3.2).
        E_delta: Structural energy component, ``float64``.
        E_p: Spreading energy component, ``float64``.
        infected_fraction: ``I / n``, ``float64``.
        n_events: All events: SAIS jumps plus flip proposals.
        n_sais_events: SAIS jumps by transition (keys ``TRANSITION_NAMES``).
        n_proposed: Flip proposals.
        n_accepted: Accepted flips.
        n_rejected: Rejected flips.
        n_zero_accepted: Accepted flips whose energy change was exactly 0.
        termination_reason: ``"t_max"``, ``"max_events"``, ``"absorbed"`` or
            ``"sais_absorbed"``.
        final_time: Clock at the end of the run.
        initial_states: Initial node states, ``int8``.
        final_states: Final node states, ``int8``.
        edges: ``(m, 2)`` ``int64`` array of edges ``(u, v)`` with ``u < v``,
            sorted.
        initial_signs: Edge signs at time 0, ``int8``, aligned with ``edges``.
        final_signs: Edge signs at the end, ``int8``, aligned with ``edges``.
        requested_r0: The configured r0, or ``None`` for :func:`simulate`.
        realized_r0: Positive edges / m at time 0, or ``None`` if m = 0.
        requested_initial_infected_fraction: The configured rho_0, or ``None``
            for :func:`simulate`.
        realized_initial_infected_fraction: Infected fraction at time 0.
        config: The :class:`SimulationConfig`, or ``None`` for :func:`simulate`.
        seed_entropy: ``entropy`` of the seed's ``SeedSequence``.
        schema_version: :data:`SCHEMA_VERSION`.
        versions: Package, Python, NumPy and networkx versions.
    """

    times: npt.NDArray[np.float64]
    S: npt.NDArray[np.int64]
    A: npt.NDArray[np.int64]
    I: npt.NDArray[np.int64]  # noqa: E741 - spec notation
    r: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    E_delta: npt.NDArray[np.float64]
    E_p: npt.NDArray[np.float64]
    infected_fraction: npt.NDArray[np.float64]
    n_events: int
    n_sais_events: dict[str, int]
    n_proposed: int
    n_accepted: int
    n_rejected: int
    n_zero_accepted: int
    termination_reason: TerminationReason
    final_time: float
    initial_states: npt.NDArray[np.int8]
    final_states: npt.NDArray[np.int8]
    edges: npt.NDArray[np.int64]
    initial_signs: npt.NDArray[np.int8]
    final_signs: npt.NDArray[np.int8]
    requested_r0: float | None
    realized_r0: float | None
    requested_initial_infected_fraction: float | None
    realized_initial_infected_fraction: float
    config: SimulationConfig | None
    seed_entropy: object
    schema_version: int
    versions: dict[str, str]


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Build a synthetic network and initial state, then run the co-evolution.

    The graph is drawn from the ``graph`` stream with the Phase 1 generator for
    ``config.topology``; the initial state is drawn from the ``initial_state``
    stream with ``make_initial_state`` (``round(rho_0 * n)`` infected nodes,
    the rest susceptible, none alert).

    Args:
        config: A validated :class:`SimulationConfig`.

    Returns:
        The :class:`SimulationResult`.

    Raises:
        TypeError: ``config`` is not a :class:`SimulationConfig`.
    """
    if not isinstance(config, SimulationConfig):
        raise TypeError(f"config must be SimulationConfig, got {type(config).__name__}")
    streams = _spawn_streams(config.seed)
    if config.topology == "barabasi_albert":
        ba_m = config.ba_m
        assert ba_m is not None  # guaranteed by SimulationConfig
        graph = generate_barabasi_albert(config.n, ba_m, config.r0, streams["graph"])
    else:
        er_p = config.er_p
        assert er_p is not None  # guaranteed by SimulationConfig
        graph = generate_erdos_renyi(config.n, er_p, config.r0, streams["graph"])
    initial = make_initial_state(
        config.n, config.initial_infected_fraction, streams["initial_state"]
    )
    return _simulate(
        graph,
        initial.states,
        config.sais_params,
        config.alpha,
        config.edge_update_rate,
        config.t_max,
        streams,
        seed=config.seed,
        max_events=config.max_events,
        stop_on_sais_absorbed=config.stop_on_sais_absorbed,
        record=config.record,
        num_points=config.num_points,
        verify_every=config.verify_every,
        requested_r0=config.r0,
        requested_initial_infected_fraction=config.initial_infected_fraction,
        config=config,
    )


def simulate(
    graph: "nx.Graph[int]",
    initial_states: StateArray,
    sais_params: SAISParams,
    alpha: float,
    edge_update_rate: float,
    t_max: float,
    *,
    seed: Seed,
    max_events: int | None = None,
    stop_on_sais_absorbed: bool = False,
    record: RecordMode = "grid",
    num_points: int = 501,
    verify_every: int | None = None,
) -> SimulationResult:
    """Run the co-evolution on a caller-supplied graph and initial state.

    Use this for graphs not built by :func:`run_simulation`, such as
    correlation-matrix graphs (spec §5, step 1). The graph and state array are
    copied and never modified. The ``graph`` and ``initial_state`` streams are
    still spawned (and left unused) so the other streams match
    :func:`run_simulation` for the same seed.

    Args:
        graph: Signed graph satisfying the Phase 1 contract.
        initial_states: ``numpy.int8`` state array of length ``n``.
        sais_params: SAIS rates, in 1/time.
        alpha: Energy weight in ``[0, 1]``.
        edge_update_rate: gamma, the per-edge flip-proposal rate, in 1/time.
        t_max: Time horizon, finite and ``>= 0``.
        seed: Non-negative ``int`` or ``numpy.random.SeedSequence``.
        max_events: See :class:`SimulationConfig`.
        stop_on_sais_absorbed: See :class:`SimulationConfig`.
        record: See :class:`SimulationConfig`.
        num_points: See :class:`SimulationConfig`.
        verify_every: See :class:`SimulationConfig`.

    Returns:
        The :class:`SimulationResult` (``config``, ``requested_r0`` and
        ``requested_initial_infected_fraction`` are ``None``).

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an argument is invalid.
        RuntimeError: ``verify_every`` is set and a check fails.
    """
    validate_signed_graph(graph)
    _check_model_options(sais_params, alpha, edge_update_rate)
    _check_run_options(
        t_max=t_max,
        seed=seed,
        max_events=max_events,
        stop_on_sais_absorbed=stop_on_sais_absorbed,
        record=record,
        num_points=num_points,
        verify_every=verify_every,
    )
    infected_positive_neighbors(graph, initial_states)  # validates the states
    return _simulate(
        graph.copy(),
        initial_states.copy(),
        sais_params,
        alpha,
        edge_update_rate,
        t_max,
        _spawn_streams(seed),
        seed=seed,
        max_events=max_events,
        stop_on_sais_absorbed=stop_on_sais_absorbed,
        record=record,
        num_points=num_points,
        verify_every=verify_every,
        requested_r0=None,
        requested_initial_infected_fraction=None,
        config=None,
    )


_INFECTED = int(State.INFECTED)
_TRANSITION_KEYS = {(1, -1): "S->I", (0, -1): "A->I", (1, 0): "S->A", (-1, 1): "I->S"}

_Row = tuple[float, int, int, int, float, float, float, float]


@dataclass
class _RunState:
    """Mutable state of one run; private to this module."""

    graph: "nx.Graph[int]"
    states: StateArray
    n_pos: npt.NDArray[np.int64]
    params: SAISParams
    alpha: float
    s3: int
    p: int
    n_positive: int
    counts: dict[int, int]
    rates: npt.NDArray[np.float64] = field(init=False)
    r_sais: float = field(init=False)
    rates_stale: bool = True

    def refresh_rates(self) -> None:
        if self.rates_stale:
            self.rates = compute_rates(self.states, self.n_pos, self.params)
            self.r_sais = float(self.rates.sum())
            self.rates_stale = False

    def row(self, time: float, m: int) -> _Row:
        n = self.graph.number_of_nodes()
        r = self.n_positive / m if m else math.nan
        return (
            time,
            self.counts[1],
            self.counts[0],
            self.counts[_INFECTED],
            r,
            energy_from_sums(self.s3, self.p, n, self.alpha),
            energy_from_sums(self.s3, self.p, n, 1.0),
            energy_from_sums(self.s3, self.p, n, 0.0),
        )


class _Recorder:
    """Collects records without drawing random numbers (Phase 3 C3-C4)."""

    def __init__(self, mode: RecordMode, t_max: float, num_points: int) -> None:
        self.mode = mode
        self.grid = np.linspace(0.0, t_max, num_points) if mode == "grid" else None
        self.next_index = 0
        self.rows: list[_Row] = []

    def start(self, row: _Row) -> None:
        if self.mode == "events":
            self.rows.append(row)

    def before_event(self, event_time: float, run: _RunState, m: int) -> None:
        """Emit grid points strictly before ``event_time`` from the current state."""
        self._emit_grid(lambda t: t < event_time, run, m)

    def after_event(self, row: _Row) -> None:
        if self.mode == "events":
            self.rows.append(row)

    def finish(self, end_time: float, run: _RunState, m: int) -> None:
        self._emit_grid(lambda t: t <= end_time, run, m)
        if not self.rows or self.rows[-1][0] < end_time:
            self.rows.append(run.row(end_time, m))

    def _emit_grid(self, keep: Callable[[float], bool], run: _RunState, m: int) -> None:
        if self.grid is None:
            return
        while self.next_index < len(self.grid) and keep(
            float(self.grid[self.next_index])
        ):
            self.rows.append(run.row(float(self.grid[self.next_index]), m))
            self.next_index += 1


def _simulate(
    graph: "nx.Graph[int]",
    states: StateArray,
    params: SAISParams,
    alpha: float,
    edge_update_rate: float,
    t_max: float,
    streams: dict[str, np.random.Generator],
    *,
    seed: Seed,
    max_events: int | None,
    stop_on_sais_absorbed: bool,
    record: RecordMode,
    num_points: int,
    verify_every: int | None,
    requested_r0: float | None,
    requested_initial_infected_fraction: float | None,
    config: SimulationConfig | None,
) -> SimulationResult:
    """Run the joint CTMC on ``graph`` and ``states``, which it owns and mutates."""
    n = graph.number_of_nodes()
    edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges)
    m = len(edges)
    initial_signs = np.array([graph.adj[u][v]["sign"] for u, v in edges], dtype=np.int8)
    initial_states = states.copy()
    run = _RunState(
        graph=graph,
        states=states,
        n_pos=infected_positive_neighbors(graph, states),
        params=params,
        alpha=float(alpha),
        s3=structural_triad_sum(graph),
        p=spreading_pair_sum(graph, states),
        n_positive=int(np.count_nonzero(initial_signs == 1)),
        counts={s: int(np.count_nonzero(states == s)) for s in (1, 0, _INFECTED)},
    )
    clock, sais_rng, edge_rng = (
        streams["clock"],
        streams["sais_selection"],
        streams["edge_updates"],
    )
    r_net = float(edge_update_rate) * m
    recorder = _Recorder(record, float(t_max), num_points)
    recorder.start(run.row(0.0, m))

    time = 0.0
    n_events = 0
    sais_counts = dict.fromkeys(TRANSITION_NAMES, 0)
    proposed = accepted = rejected = zero_accepted = 0
    reason: TerminationReason
    while True:
        run.refresh_rates()
        if stop_on_sais_absorbed and run.r_sais == 0.0:
            reason = "sais_absorbed"
            break
        if max_events is not None and n_events >= max_events:
            reason = "max_events"
            break
        total = run.r_sais + r_net
        if total <= 0.0:  # sums of non-negative finite rates: exact test
            reason, time = "absorbed", float(t_max)
            break
        dt = sample_waiting_time(total, clock)
        if time + dt > t_max:
            reason, time = "t_max", float(t_max)
            break
        recorder.before_event(time + dt, run, m)
        time += dt

        if clock.random() * total < run.r_sais:
            node, new_state = select_event(run.rates, sais_rng)
            target = int(new_state)
            run.p += state_change_pair_delta(graph, states, node, target)
            old_state = int(apply_sais_event(graph, states, run.n_pos, node, target))
            run.counts[old_state] -= 1
            run.counts[target] += 1
            sais_counts[_TRANSITION_KEYS[(old_state, target)]] += 1
            run.rates_stale = True
        else:
            proposed += 1
            u, v = edges[int(edge_rng.integers(m))]
            d_s3, d_p = flip_energy_numerators(graph, states, u, v)
            delta = energy_from_sums(d_s3, d_p, n, run.alpha, exact=True)
            if accept_flip(delta, edge_rng):
                new_sign = -graph.adj[u][v]["sign"]
                graph.adj[u][v]["sign"] = new_sign
                run.s3 += d_s3
                run.p += d_p
                run.n_positive += new_sign
                refresh_after_edge_flip(run.n_pos, states, u, v, new_sign)
                if states[u] == _INFECTED or states[v] == _INFECTED:
                    run.rates_stale = True
                accepted += 1
                zero_accepted += delta == 0
            else:
                rejected += 1
        n_events += 1
        recorder.after_event(run.row(time, m))
        if verify_every is not None and n_events % verify_every == 0:
            _verify(run, n_events)

    recorder.finish(time, run, m)
    columns = list(zip(*recorder.rows, strict=True))
    i_series = np.array(columns[3], dtype=np.int64)
    return SimulationResult(
        times=np.array(columns[0], dtype=np.float64),
        S=np.array(columns[1], dtype=np.int64),
        A=np.array(columns[2], dtype=np.int64),
        I=i_series,
        r=np.array(columns[4], dtype=np.float64),
        E=np.array(columns[5], dtype=np.float64),
        E_delta=np.array(columns[6], dtype=np.float64),
        E_p=np.array(columns[7], dtype=np.float64),
        infected_fraction=i_series / n,
        n_events=n_events,
        n_sais_events=sais_counts,
        n_proposed=proposed,
        n_accepted=accepted,
        n_rejected=rejected,
        n_zero_accepted=zero_accepted,
        termination_reason=reason,
        final_time=time,
        initial_states=initial_states,
        final_states=states.copy(),
        edges=np.array(edges, dtype=np.int64).reshape(m, 2),
        initial_signs=initial_signs,
        final_signs=np.array(
            [graph.adj[u][v]["sign"] for u, v in edges], dtype=np.int8
        ),
        requested_r0=requested_r0,
        realized_r0=int(np.count_nonzero(initial_signs == 1)) / m if m else None,
        requested_initial_infected_fraction=requested_initial_infected_fraction,
        realized_initial_infected_fraction=(
            int(np.count_nonzero(initial_states == _INFECTED)) / n
        ),
        config=config,
        seed_entropy=_seed_sequence(seed).entropy,
        schema_version=SCHEMA_VERSION,
        versions=_environment_versions(),
    )


def _verify(run: _RunState, n_events: int) -> None:
    """Recompute every maintained quantity from scratch; raise on any mismatch."""
    g, states = run.graph, run.states
    signs: list[int] = [data["sign"] for _, _, data in g.edges(data=True)]
    expected: dict[str, object] = {
        "n_pos": infected_positive_neighbors(g, states).tolist(),
        "S3": structural_triad_sum(g),
        "P": spreading_pair_sum(g, states),
        "positive edges": sum(1 for s in signs if s == 1),
        "state counts": {s: int(np.count_nonzero(states == s)) for s in (1, 0, -1)},
    }
    actual: dict[str, object] = {
        "n_pos": run.n_pos.tolist(),
        "S3": run.s3,
        "P": run.p,
        "positive edges": run.n_positive,
        "state counts": run.counts,
    }
    if not run.rates_stale:
        fresh = compute_rates(states, run.n_pos, run.params)
        expected["rates"] = fresh.tolist()
        actual["rates"] = run.rates.tolist()
        expected["R_sais"] = float(fresh.sum())
        actual["R_sais"] = run.r_sais
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(
                f"verification failed after {n_events} events: maintained {key} "
                f"{actual[key]!r} != recomputed {value!r}"
            )


def _environment_versions() -> dict[str, str]:
    """Versions recorded in every result (reproducibility scope, Phase 3 E2)."""
    return {
        "financial_contagion_sim": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
    }


def _spawn_streams(seed: Seed) -> dict[str, np.random.Generator]:
    """Spawn the five child generators named in :data:`STREAM_NAMES`.

    A ``SeedSequence`` argument is copied first, so the caller's object is not
    consumed and the same seed always yields the same streams.

    Raises:
        TypeError: ``seed`` is not a non-negative ``int`` or ``SeedSequence``.
        ValueError: ``seed`` is a negative ``int``.
    """
    root = _seed_sequence(seed)
    children = root.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }


def _seed_sequence(seed: object) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(
            entropy=seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    if isinstance(seed, int) and not isinstance(seed, bool):
        if seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed}")
        return np.random.SeedSequence(seed)
    raise TypeError(
        "seed must be a non-negative int or numpy.random.SeedSequence, "
        f"got {seed!r} ({type(seed).__name__})"
    )


def _check_model_options(
    sais_params: object, alpha: object, edge_update_rate: object
) -> None:
    if not isinstance(sais_params, SAISParams):
        raise TypeError(
            f"sais_params must be SAISParams, got {type(sais_params).__name__}"
        )
    _check_unit_interval("alpha", alpha)
    _check_finite_non_negative("edge_update_rate", edge_update_rate)


def _check_run_options(
    *,
    t_max: object,
    seed: object,
    max_events: object,
    stop_on_sais_absorbed: object,
    record: object,
    num_points: object,
    verify_every: object,
) -> None:
    _check_finite_non_negative("t_max", t_max)
    _seed_sequence(seed)
    if max_events is not None:
        _check_int("max_events", max_events, minimum=0)
    if not isinstance(stop_on_sais_absorbed, bool):
        raise TypeError(
            f"stop_on_sais_absorbed must be a bool, got {stop_on_sais_absorbed!r}"
        )
    if not isinstance(record, str):
        raise TypeError(f"record must be a str, got {record!r}")
    if record not in get_args(RecordMode):
        raise ValueError(
            f"record must be one of {get_args(RecordMode)}, got {record!r}"
        )
    _check_int("num_points", num_points, minimum=2)
    if verify_every is not None:
        _check_int("verify_every", verify_every, minimum=1)


def _check_finite_non_negative(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number, got {value!r} ({type(value).__name__})"
        )
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
