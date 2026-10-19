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
"""

import math
import platform
from dataclasses import dataclass
from typing import Literal, get_args

import networkx as nx
import numpy as np
import numpy.typing as npt

from financial_contagion_sim import __version__
from financial_contagion_sim.graph_builder import _check_int, _check_unit_interval
from financial_contagion_sim.sais_engine import SAISParams

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
