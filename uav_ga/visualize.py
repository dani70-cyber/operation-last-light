"""Plots for Operation Last Light: convergence, evolved routes, coverage heatmaps.

All functions take plain data (env, FleetChromosome, history lists) and save a PNG.
Layers in `env` are indexed [y, x], so every imshow uses origin="lower".

NOTE: coverage here is an *approximate footprint* for drawing (sensor radius =
1 + 3 * power along each leg, return leg at minimum power). The official
coverage number comes from fitness.py, so quote that one in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed, we only save files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from uav_ga.chromosome import POWER_MIN, FleetChromosome

DRONE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
GA_COLOR, RS_COLOR = "#1a9850", "#d73027"


# ---------------------------------------------------------------- helpers
def _pad(histories: list[list[float]]) -> np.ndarray:
    """Runs stop at different generations (early stop): carry the last value forward."""
    n = max(len(h) for h in histories)
    return np.array([list(h) + [h[-1]] * (n - len(h)) for h in histories], dtype=float)


def _layer(ax, mask: np.ndarray, color: str, alpha: float = 1.0) -> None:
    """Paint the True cells of `mask` in a single colour."""
    shown = np.ma.masked_where(~mask, np.ones(mask.shape))
    ax.imshow(shown, cmap=ListedColormap([color]), origin="lower", alpha=alpha,
              vmin=0, vmax=1, interpolation="nearest")


def coverage_mask(env, fleet: FleetChromosome) -> np.ndarray:
    """Boolean grid of cells inside any sensor footprint along any route."""
    h, w = env.priority.shape
    yy, xx = np.mgrid[0:h, 0:w]
    covered = np.zeros((h, w), dtype=bool)
    for plan in fleet.plans:
        pts = [env.depot] + plan.waypoints() + [env.depot]
        powers = [g.sensor_power for g in plan.genes] + [POWER_MIN]  # return leg = min power
        for (a, b), power in zip(zip(pts[:-1], pts[1:]), powers):
            radius = 1.0 + 3.0 * power
            steps = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])) + 1)
            for t in np.linspace(0.0, 1.0, steps):
                px = a[0] + t * (b[0] - a[0])
                py = a[1] + t * (b[1] - a[1])
                covered |= (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    return covered


def _base_map(ax, env, show_hazard: bool = True) -> None:
    if show_hazard:
        ax.imshow(env.hazard, cmap="Oranges", origin="lower", vmin=0, vmax=1, alpha=0.55)
    _layer(ax, env.no_fly.astype(bool), "#222222")
    ax.plot(*env.depot, marker="*", color="gold", markeredgecolor="black",
            markersize=16, linestyle="none", zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------- plots
def plot_convergence(ga_hist: list[list[float]], rs_hist: list[list[float]], out: str | Path) -> None:
    """Best-so-far fitness: GA vs random search, mean +/- std over seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for hist, color, label in ((ga_hist, GA_COLOR, "AI Commander (GA)"),
                               (rs_hist, RS_COLOR, "Rookie Survivor (random search)")):
        arr = _pad(hist)
        x = np.arange(arr.shape[1])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        for row in arr:                                   # faint individual seeds
            ax.plot(x, row, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(x, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("Generation (random search: every pop_size evaluations)")
    ax.set_ylabel("Best fitness so far")
    ax.set_title("Convergence, same evaluation budget")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def draw_routes(ax, env, fleet: FleetChromosome, title: str = "") -> None:
    _base_map(ax, env)
    pr = env.priority > 0
    ys, xs = np.nonzero(pr)
    ax.scatter(xs, ys, s=6, color="#2b8cbe", alpha=0.6, zorder=3)  # survivor camps
    callsigns = ("Falcon-1", "Shaheen-2", "Chai-Runner-3", "Raven-4", "Kite-5")
    for i, plan in enumerate(fleet.plans):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        pts = [env.depot] + plan.waypoints() + [env.depot]
        px, py = zip(*pts)
        ax.plot(px, py, "-", color=color, linewidth=2, zorder=4,
                label=callsigns[i] if i < len(callsigns) else f"Drone-{i + 1}")
        ax.plot(px[1:-1], py[1:-1], "o", color=color, markeredgecolor="white",
                markersize=6, zorder=6)
    ax.set_title(title, fontsize=10)


def plot_routes(env, fleet: FleetChromosome, out: str | Path, title: str = "Evolved fleet routes") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    draw_routes(ax, env, fleet, title)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.text(0.5, 0.01, "star = depot, black = no-fly, orange = hazard, blue dots = survivor camps",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_coverage_snapshots(env, snapshots: list[tuple[int, FleetChromosome]],
                            out: str | Path, n_panels: int = 4) -> None:
    """Rescued (green) vs unreached (red) survivor camps at several generations."""
    idx = sorted(set(np.linspace(0, len(snapshots) - 1, n_panels).round().astype(int)))
    fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.6))
    axes = np.atleast_1d(axes)
    total = float(env.priority.sum())
    pr = env.priority > 0
    for ax, i in zip(axes, idx):
        gen, fleet = snapshots[i]
        cov = coverage_mask(env, fleet)
        _base_map(ax, env, show_hazard=False)
        _layer(ax, pr & ~cov, "#d73027")      # camps not reached
        _layer(ax, pr & cov, "#1a9850")       # camps covered
        share = float(env.priority[cov].sum()) / total if total > 0 else 0.0
        ax.set_title(f"Generation {gen}\nfitness {fleet.fitness:.3f} | ~{share:.0%} priority covered",
                     fontsize=10)
    fig.suptitle("Coverage heatmap: green = covered camps, red = unreached", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out, dpi=200)
    plt.close(fig)