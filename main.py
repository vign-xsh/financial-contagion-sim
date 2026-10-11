"""Repository-root entry point; delegates to ``financial_contagion_sim.cli``."""

from financial_contagion_sim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
