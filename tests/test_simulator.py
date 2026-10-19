"""Tests for the co-evolutionary simulation engine (``simulator``).

Spec §1, §3.3 and §5, step 2; Phase 3 decisions in DECISIONS.md.
"""

import dataclasses
import math

import numpy as np
import pytest

from financial_contagion_sim.sais_engine import SAISParams
from financial_contagion_sim.simulator import (
    SCHEMA_VERSION,
    STREAM_NAMES,
    SimulationConfig,
    _spawn_streams,
)

PARAMS = SAISParams(beta=2.0, beta_a=0.5, kappa=3.0, delta=0.7)


def _cfg(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "n": 30,
        "topology": "erdos_renyi",
        "er_p": 0.2,
        "sais_params": PARAMS,
        "alpha": 0.5,
        "r0": 0.6,
        "initial_infected_fraction": 0.2,
        "edge_update_rate": 1.0,
        "t_max": 2.0,
        "seed": 7,
    }
    values.update(overrides)
    return SimulationConfig(**values)


# --- configuration ------------------------------------------------------------------


def test_valid_configs() -> None:
    er = _cfg()
    assert (er.max_events, er.stop_on_sais_absorbed, er.record) == (None, False, "grid")
    assert (er.num_points, er.verify_every) == (501, None)
    _cfg(topology="barabasi_albert", er_p=None, ba_m=2)
    _cfg(edge_update_rate=0.0, t_max=0.0, r0=0.0, initial_infected_fraction=1.0)
    _cfg(seed=np.random.SeedSequence(3), record="events", num_points=2)


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cfg().n = 5


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"topology": "watts_strogatz"}, ValueError, "topology must be one of"),
        ({"topology": 1}, TypeError, "topology must be a str"),
        ({"er_p": None}, ValueError, "needs er_p and no ba_m"),
        ({"ba_m": 2}, ValueError, "needs er_p and no ba_m"),
        ({"topology": "barabasi_albert"}, ValueError, "needs ba_m and no er_p"),
        ({"topology": "barabasi_albert", "er_p": None}, ValueError, "needs ba_m"),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 30},
            ValueError,
            "1 <= ba_m < n",
        ),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 0},
            ValueError,
            "ba_m must be >= 1",
        ),
        (
            {"topology": "barabasi_albert", "er_p": None, "ba_m": 2.0},
            TypeError,
            "ba_m must be an int",
        ),
        ({"er_p": 1.5}, ValueError, r"er_p must be in \[0, 1\]"),
        ({"n": 0}, ValueError, "n must be >= 1"),
        ({"n": 30.0}, TypeError, "n must be an int"),
        ({"n": True}, TypeError, "n must be an int"),
        ({"alpha": 1.5}, ValueError, r"alpha must be in \[0, 1\]"),
        ({"alpha": "0.5"}, TypeError, "alpha must be a real number"),
        ({"r0": math.nan}, ValueError, r"r0 must be in \[0, 1\]"),
        ({"initial_infected_fraction": 2.0}, ValueError, "initial_infected_fraction"),
        ({"edge_update_rate": -1.0}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": math.nan}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": math.inf}, ValueError, "edge_update_rate must be finite"),
        ({"edge_update_rate": True}, TypeError, "edge_update_rate must be a real"),
        ({"t_max": -1.0}, ValueError, "t_max must be finite and >= 0"),
        ({"t_max": math.inf}, ValueError, "t_max must be finite and >= 0"),
        ({"t_max": math.nan}, ValueError, "t_max must be finite and >= 0"),
        ({"seed": -1}, ValueError, "seed must be a non-negative int"),
        ({"seed": 1.5}, TypeError, "seed must be a non-negative int or"),
        ({"seed": True}, TypeError, "seed must be a non-negative int or"),
        ({"seed": None}, TypeError, "seed must be a non-negative int or"),
        ({"max_events": -1}, ValueError, "max_events must be >= 0"),
        ({"max_events": 1.0}, TypeError, "max_events must be an int"),
        (
            {"stop_on_sais_absorbed": 1},
            TypeError,
            "stop_on_sais_absorbed must be a bool",
        ),
        ({"record": "snapshots"}, ValueError, "record must be one of"),
        ({"record": None}, TypeError, "record must be a str"),
        ({"num_points": 1}, ValueError, "num_points must be >= 2"),
        ({"num_points": True}, TypeError, "num_points must be an int"),
        ({"verify_every": 0}, ValueError, "verify_every must be >= 1"),
        ({"sais_params": (2.0, 0.5, 3.0, 0.7)}, TypeError, "sais_params must be"),
    ],
)
def test_config_rejects_invalid(
    overrides: dict[str, object], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        _cfg(**overrides)


def test_beta_a_equal_beta_is_rejected_by_phase2_params() -> None:
    with pytest.raises(ValueError, match="0 <= beta_a < beta"):
        SAISParams(beta=1.0, beta_a=1.0, kappa=0.0, delta=0.0)


# --- random streams -------------------------------------------------------------------


def test_streams_follow_documented_spawn_order() -> None:
    assert STREAM_NAMES == (
        "graph",
        "initial_state",
        "clock",
        "sais_selection",
        "edge_updates",
    )
    streams = _spawn_streams(11)
    children = np.random.SeedSequence(11).spawn(5)
    for name, child in zip(STREAM_NAMES, children, strict=True):
        expected = np.random.default_rng(child).random(4)
        assert np.array_equal(streams[name].random(4), expected)


def test_seed_sequence_argument_is_not_consumed() -> None:
    seq = np.random.SeedSequence(11)
    first = _spawn_streams(seq)["clock"].random(3)
    second = _spawn_streams(seq)["clock"].random(3)
    assert np.array_equal(first, second)
    assert seq.n_children_spawned == 0
    assert np.array_equal(_spawn_streams(11)["clock"].random(3), first)


def test_schema_version_is_int() -> None:
    assert type(SCHEMA_VERSION) is int and SCHEMA_VERSION >= 1
