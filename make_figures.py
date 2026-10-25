"""Run GA + random search for several seeds and export figures + a results table.

Usage (from repo root):
    python make_figures.py --seeds 5 --generations 150 --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uav_ga import visualize as viz
from uav_ga.baseline import run_random_search
from uav_ga.environment import generate_map
from uav_ga.ga import Hyperparameters, run_ga


def build_env(map_seed: int = 7, depot: tuple[int, int] = (25, 25), max_range: float = 120.0):
    """Build the SAME environment main.py uses (same defaults)."""
    return generate_map(seed=map_seed, depot=depot, max_range=max_range)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    env = build_env()

    rows, ga_hist, rs_hist = [], [], []
    best_res, best_fit = None, -np.inf
    for seed in range(args.seeds):
        hp = Hyperparameters(generations=args.generations, seed=seed)
        res = run_ga(env, hp, verbose=False)
        rs_h, rs_best = run_random_search(
            env, res.evaluations, hp.n_drones, hp.n_waypoints, seed=seed, chunk=hp.pop_size
        )
        ga_hist.append(res.best_history)
        rs_hist.append(rs_h)
        rows.append({
            "seed": seed,
            "ga_fitness": res.best.fitness,
            "random_fitness": rs_best.fitness,
            "evaluations": res.evaluations,
            "generations_run": res.generations_run,
        })
        print(f"seed {seed}: GA {res.best.fitness:.3f} | random {rs_best.fitness:.3f} "
              f"| gens {res.generations_run}")
        if res.best.fitness > best_fit:
            best_fit, best_res = res.best.fitness, res

    df = pd.DataFrame(rows)
    df.to_csv(out / "ga_vs_random.csv", index=False)
    print(f"\nGA     : {df.ga_fitness.mean():.3f} +/- {df.ga_fitness.std(ddof=0):.3f}")
    print(f"Random : {df.random_fitness.mean():.3f} +/- {df.random_fitness.std(ddof=0):.3f}")

    viz.plot_convergence(ga_hist, rs_hist, out / "convergence.png")
    viz.plot_routes(env, best_res.best, out / "best_routes.png",
                    title=f"Best evolved fleet (fitness {best_res.best.fitness:.3f})")
    viz.plot_coverage_snapshots(env, best_res.snapshots, out / "coverage_heatmaps.png")
    print(f"\nSaved convergence.png, best_routes.png, coverage_heatmaps.png, ga_vs_random.csv in {out}/")


if __name__ == "__main__":
    main()