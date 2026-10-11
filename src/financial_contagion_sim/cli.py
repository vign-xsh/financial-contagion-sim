"""Command-line interface.

Responsibility: parse command-line arguments and dispatch to the simulator.

Phase 0 provides ``--help`` and ``--version`` only. Running a simulation is
not implemented (Phase 1+).
"""

import argparse
from collections.abc import Sequence

from financial_contagion_sim import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="financial-contagion-sim",
        description=(
            "Co-evolutionary financial contagion & systemic risk simulator. "
            "No simulation commands are implemented yet (Phase 1+)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and exit; there is nothing to run before Phase 1."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
