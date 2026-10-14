"""Verify CTMC rate/probability bounds and state conservation.

Spec §4, for ``sais_engine``. Skipped: nothing is implemented yet, so nothing
is verified here until Phase 1+.
"""

import pytest


@pytest.mark.skip(reason="Phase 1")
def test_sais_state_transitions() -> None:
    """State transitions follow the SAIS rates on a small known network."""
