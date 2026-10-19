"""Shared test helpers."""

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import (
    SCHEMA_VERSION,
    TRANSITION_NAMES,
    SimulationConfig,
    SimulationResult,
)

FIXTURE_CONFIG = SimulationConfig(
    n=10,
    topology="erdos_renyi",
    er_p=0.3,
    sais_params=SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7),
    alpha=0.5,
    r0=0.6,
    initial_infected_fraction=0.2,
    edge_update_rate=1.0,
    t_max=4.0,
    seed=3,
)


def build_result(
    times: Sequence[float],
    S: Sequence[int],
    A: Sequence[int],
    I: Sequence[int],  # noqa: E741 - spec notation
    r: Sequence[float],
    E: Sequence[float],
    *,
    n: int = 10,
    m: int = 4,
    reason: str = "t_max",
    config: SimulationConfig | None = FIXTURE_CONFIG,
) -> SimulationResult:
    """A hand-built SimulationResult; the arrays are used exactly as given."""
    i_series = np.asarray(I, dtype=np.int64)
    energy = np.asarray(E, dtype=np.float64)
    return SimulationResult(
        times=np.asarray(times, dtype=np.float64),
        S=np.asarray(S, dtype=np.int64),
        A=np.asarray(A, dtype=np.int64),
        I=i_series,
        r=np.asarray(r, dtype=np.float64),
        E=energy,
        E_delta=energy * 0.5,
        E_p=energy * 0.25,
        infected_fraction=i_series / n,
        n_events=len(times) - 1,
        n_sais_events=dict.fromkeys(TRANSITION_NAMES, 0),
        n_proposed=0,
        n_accepted=0,
        n_rejected=0,
        n_zero_accepted=0,
        termination_reason=reason,  # type: ignore[arg-type]
        final_time=float(times[-1]),
        initial_states=np.ones(n, dtype=np.int8),
        final_states=np.ones(n, dtype=np.int8),
        edges=np.array([[0, k] for k in range(1, m + 1)], dtype=np.int64).reshape(m, 2),
        initial_signs=np.ones(m, dtype=np.int8),
        final_signs=np.ones(m, dtype=np.int8),
        requested_r0=0.6,
        realized_r0=1.0 if m else None,
        requested_initial_infected_fraction=0.2,
        realized_initial_infected_fraction=float(i_series[0]) / n,
        config=config,
        seed_entropy=3,
        schema_version=SCHEMA_VERSION,
        versions={"financial_contagion_sim": "test"},
    )


@pytest.fixture
def make_result() -> Callable[..., SimulationResult]:
    return build_result
