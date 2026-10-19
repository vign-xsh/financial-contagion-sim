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

3D surfaces (Phase 4 B1, B4, B9, B10): :func:`plot_surface` draws one Plotly
surface per rho_0 value over x = alpha and y = r0, with a dropdown to switch
between them, for either rho_inf or the steady-state energy. Plotly indexes
``z[y][x]``, so ``z[r0_index][alpha_index]``. NaN marks a grid point without
enough converged repetitions and appears as a hole. :func:`write_surface_html`
writes a self-contained HTML file with plotly.js embedded (no CDN).

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

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure

from financial_contagion_sim.simulator import SimulationResult
from financial_contagion_sim.sweep import MetricSurface, SweepGrid, SweepResult

SurfaceMetric = Literal["rho_inf", "energy"]

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


_SURFACE_LABELS: dict[str, tuple[str, str]] = {
    "rho_inf": ("Steady-state infected fraction rho_inf", "rho_inf"),
    "energy": ("Steady-state network energy E(G)", "E_ss"),
}


def plot_surface(sweep: SweepResult, metric: SurfaceMetric) -> Any:
    """Interactive 3D surface of a sweep metric (spec §5, step 3; Phase 4 B).

    One ``go.Surface`` per rho_0 value, with a dropdown to switch between
    them. Plotly indexes ``z[y][x]``, so ``z[r0_index][alpha_index]`` is the
    mean at ``(alpha[alpha_index], r0[r0_index])`` (B10). Non-converged grid
    points are NaN and appear as holes. Hover text shows the grid values, the
    mean, the sample standard deviation, the standard error and the number
    of converged repetitions.

    Args:
        sweep: A :class:`~financial_contagion_sim.sweep.SweepResult`.
        metric: ``"rho_inf"`` or ``"energy"``.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Raises:
        TypeError: ``sweep`` is not a ``SweepResult``.
        ValueError: ``metric`` is unknown.
    """
    if not isinstance(sweep, SweepResult):
        raise TypeError(f"sweep must be SweepResult, got {type(sweep).__name__}")
    if metric not in _SURFACE_LABELS:
        raise ValueError(f"metric must be 'rho_inf' or 'energy', got {metric!r}")
    surface: MetricSurface = sweep.rho_inf if metric == "rho_inf" else sweep.energy
    title, z_label = _SURFACE_LABELS[metric]
    grid = sweep.grid
    traces = []
    for p, rho0 in enumerate(grid.rho0):
        text = [
            [
                _hover(grid, surface, p, r, a, sweep.repetitions)
                for a in range(len(grid.alpha))
            ]
            for r in range(len(grid.r0))
        ]
        traces.append(
            go.Surface(
                x=list(grid.alpha),
                y=list(grid.r0),
                z=surface.mean[p],
                # A 2D object array: plotly turns each row of a nested list
                # into one string, which would break per-cell hover text.
                text=np.array(text, dtype=object),
                hovertemplate="%{text}<extra></extra>",
                name=f"rho_0 = {rho0:g}",
                visible=p == 0,
                colorbar={"title": {"text": z_label}},
            )
        )
    buttons = [
        {
            "label": f"rho_0 = {rho0:g}",
            "method": "update",
            "args": [{"visible": [q == p for q in range(len(grid.rho0))]}],
        }
        for p, rho0 in enumerate(grid.rho0)
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": f"{title} over alpha and r0 (synthetic simulation output)"},
        scene={
            "xaxis": {"title": {"text": "alpha (structural weight)"}},
            "yaxis": {"title": {"text": "r0 (initial positive-link ratio)"}},
            "zaxis": {"title": {"text": z_label}},
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.1}],
        annotations=[
            {
                "text": SYNTHETIC_CAPTION,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.05,
            }
        ],
    )
    return fig


def write_surface_html(fig: Any, path: str | Path) -> None:
    """Write a fully self-contained HTML file (plotly.js embedded, no CDN).

    Raises:
        TypeError: ``fig`` is not a plotly ``Figure``.
        ValueError: ``path`` does not end in ``.html``.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"fig must be a plotly Figure, got {type(fig).__name__}")
    target = Path(path)
    if target.suffix.lower() != ".html":
        raise ValueError(f"path must end in .html, got {str(target)!r}")
    fig.write_html(target, include_plotlyjs=True, full_html=True)


def _hover(
    grid: SweepGrid,
    surface: MetricSurface,
    p: int,
    r: int,
    a: int,
    repetitions: int,
) -> str:
    def fmt(value: float) -> str:
        return "NaN" if math.isnan(value) else f"{value:.6g}"

    alpha, r0, rho0 = grid.alpha[a], grid.r0[r], grid.rho0[p]
    return (
        f"alpha={alpha:g}<br>r0={r0:g}<br>rho_0={rho0:g}<br>"
        f"mean={fmt(float(surface.mean[p, r, a]))}<br>"
        f"std={fmt(float(surface.std[p, r, a]))}<br>"
        f"SE={fmt(float(surface.se[p, r, a]))}<br>"
        f"k={int(surface.k_converged[p, r, a])}/{repetitions} converged"
    )


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
