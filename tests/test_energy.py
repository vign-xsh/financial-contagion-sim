"""Validate fast local energy differences against exact matrix recomputation.

Spec §4, for ``energy_calculator``. Skipped: nothing is implemented yet, so
nothing is verified here until Phase 1+.
"""

import pytest


@pytest.mark.skip(reason="Phase 1")
def test_structural_balance_energy() -> None:
    """Energy of balanced and unbalanced triads matches hand-computed values."""
