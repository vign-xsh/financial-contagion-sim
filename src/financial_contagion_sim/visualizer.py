"""Visualization of simulation results (spec §4; outputs in §5, step 3).

Responsibility: 2D state-trajectory plots (Matplotlib) and 3D parameter
surfaces (Plotly). This module only draws results it is given; it never runs a
simulation and never validates model parameters.

2D time series (Phase 4 A1-A8): one figure with three stacked panels sharing
the time axis: S, A, I counts; the friendly-link density r(t); and the total
energy E(G_t) (spec §5, step 2). Lines are drawn as steps (``steps-post``)
because the recorded state is piecewise constant and right-continuous. If the
graph has no edges, r(t) is NaN and appears as a gap, never as 0. Figures are
built with :class:`matplotlib.figure.Figure` directly (no pyplot), so there is
no global figure state and nothing is shown on screen.

Interpretation notes, carried from Phase 3 rather than hidden:

- With the spec's rule that a flip with zero energy change is accepted with
  probability 1/2 (§3.3), edges with no triangles and no infected endpoint
  random-walk towards r = 0.5, so r(t) drifts regardless of contagion.
- At alpha = 1 the acceptance energy ignores node states, so topology evolves
  independently of infection; at alpha = 0 only the spreading energy counts.

The spec asks for charts "illustrating natural immunization and clustering
behavior" (§5, step 3) but defines no metric for either; none is computed or
claimed here (Phase 4 A9, Open). Every figure carries the caption
:data:`SYNTHETIC_CAPTION`.
"""

from pathlib import Path

from matplotlib.figure import Figure

from financial_contagion_sim.simulator import SimulationResult

SYNTHETIC_CAPTION = "Synthetic simulation output; not empirical market data."
PNG_DPI = 150
TIMESERIES_FIGSIZE = (8.0, 9.0)


def plot_timeseries(result: SimulationResult, *, normalize: bool = False) -> Figure:
    """Draw S/A/I, r(t) and E(G_t) from a simulation result (spec §5, step 2).

    Args:
        result: A :class:`~financial_contagion_sim.simulator.SimulationResult`.
        normalize: If true, plot S, A, I as fractions of ``n`` instead of
            counts.

    Returns:
        A new Matplotlib ``Figure`` with three axes sharing the time axis.

    Raises:
        TypeError: ``result`` is not a ``SimulationResult`` or ``normalize`` is
            not a bool.
    """
    if not isinstance(result, SimulationResult):
        raise TypeError(f"result must be SimulationResult, got {type(result).__name__}")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, got {normalize!r}")
    n = len(result.initial_states)
    m = len(result.edges)
    fig = Figure(figsize=TIMESERIES_FIGSIZE)
    states_ax, r_ax, energy_ax = fig.subplots(3, 1, sharex=True)
    t = result.times

    for label, series in (
        ("S (susceptible)", result.S),
        ("A (alert)", result.A),
        ("I (infected)", result.I),
    ):
        values = series / n if normalize else series
        states_ax.step(t, values, where="post", label=label)
    states_ax.set_ylabel(
        "fraction of institutions" if normalize else "number of institutions"
    )
    states_ax.set_title("Node states")
    states_ax.legend(loc="best")

    r_ax.step(t, result.r, where="post", label="r(t)")
    r_ax.set_ylabel("r(t) = positive edges / total edges")
    r_ax.set_ylim(-0.02, 1.02)
    r_ax.set_title("Friendly-link density")
    r_ax.legend(loc="best")
    if m == 0:
        r_ax.text(
            0.5,
            0.5,
            "No edges: r(t) is undefined (NaN) and not drawn",
            transform=r_ax.transAxes,
            ha="center",
            va="center",
        )

    energy_ax.step(t, result.E, where="post", label="E(G_t)")
    energy_ax.set_ylabel("E(G_t)")
    energy_ax.set_title("Network energy")
    energy_ax.set_xlabel("time (model units)")
    energy_ax.legend(loc="best")

    fig.suptitle(_title(result))
    fig.text(0.5, 0.005, SYNTHETIC_CAPTION, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    return fig


def save_figure(fig: Figure, path: str | Path) -> None:
    """Save a Matplotlib figure as PNG at :data:`PNG_DPI` dpi.

    Args:
        fig: The figure to save.
        path: Destination file; its suffix must be ``.png``.

    Raises:
        TypeError: ``fig`` is not a Matplotlib ``Figure``.
        ValueError: ``path`` does not end in ``.png``.
    """
    if not isinstance(fig, Figure):
        raise TypeError(f"fig must be a matplotlib Figure, got {type(fig).__name__}")
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError(f"path must end in .png, got {str(target)!r}")
    fig.savefig(target, dpi=PNG_DPI, format="png")


def _title(result: SimulationResult) -> str:
    """Key parameters, taken from the result's own config echo."""
    n, m = len(result.initial_states), len(result.edges)
    config = result.config
    if config is None:
        return (
            f"Simulation on a supplied graph: n={n}, m={m}, "
            f"seed entropy={result.seed_entropy}"
        )
    params = config.sais_params
    topology = (
        f"Barabási–Albert m={config.ba_m}"
        if config.topology == "barabasi_albert"
        else f"Erdős–Rényi p={config.er_p}"
    )
    seed = config.seed if isinstance(config.seed, int) else result.seed_entropy
    return (
        f"{topology}, n={n}, m={m}, seed={seed}\n"
        f"α={config.alpha}, r0={config.r0}, ρ0={config.initial_infected_fraction}, "
        f"γ={config.edge_update_rate}, β={params.beta}, β_a={params.beta_a}, "
        f"κ={params.kappa}, δ={params.delta}"
    )
