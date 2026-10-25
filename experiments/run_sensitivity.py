"""Hyperparameter sensitivity sweep for Operation Last Light (one-factor-at-a-time).

Varies population size, mutation rate and tournament size (3 values each),
holding the other parameters at their defaults. Each config runs on N seeds.

Run from the repo root:
    python experiments/run_sensitivity.py --seeds 3 --generations 150 --workers 4
Outputs (in results/):
    sensitivity_raw.csv      one row per (parameter, value, seed)
    sensitivity_summary.csv  mean/std per (parameter, value)
    sensitivity.png          errorbar plot, one panel per parameter

Needs build_env() in make_figures.py to be filled in.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from uav_ga.ga import Hyperparameters, run_ga
from make_figures import build_env

GRID = {
    "pop_size": [40, 80, 160],
    "mutation_rate": [0.05, 0.12, 0.25],
    "tournament_k": [2, 3, 5],
}


def run_one(job):
    """One GA run. Builds its own env so it is safe in a worker process."""
    param, value, seed, generations = job
    env = build_env()
    hp = replace(Hyperparameters(), generations=generations, seed=seed, **{param: value})
    res = run_ga(env, hp, snapshot_every=10_000, verbose=False)
    return {
        "param": param,
        "value": value,
        "seed": seed,
        "best_fitness": res.best.fitness,
        "mean_final_pop": res.mean_history[-1],
        "evaluations": res.evaluations,
        "generations_run": res.generations_run,
    }


def make_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(GRID), figsize=(12, 3.6), sharey=True)
    for ax, param in zip(axes, GRID):
        s = summary[summary.param == param].sort_values("value")
        xs = range(len(s))
        ax.errorbar(xs, s["mean"], yerr=s["std"], fmt="o-", capsize=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{v:g}" for v in s["value"]])
        ax.set_xlabel(param)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("best fitness (mean ± std)")
    fig.suptitle("Hyperparameter sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild sensitivity.png from an existing sensitivity_summary.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.plot_only:
        summary = pd.read_csv(os.path.join(args.out, "sensitivity_summary.csv"))
        make_plot(summary, os.path.join(args.out, "sensitivity.png"))
        print("sensitivity.png rebuilt")
        return

    jobs = [(p, v, s, args.generations)
            for p, vals in GRID.items() for v in vals for s in range(args.seeds)]
    print(f"{len(jobs)} runs, {args.workers} worker(s)")

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(run_one, jobs), 1):
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {row['param']}={row['value']} seed={row['seed']} "
                      f"-> {row['best_fitness']:.3f}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            row = run_one(job)
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['param']}={row['value']} seed={row['seed']} "
                  f"-> {row['best_fitness']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "sensitivity_raw.csv"), index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.out, "sensitivity_raw.csv"), index=False)

    # ddof=0 (population std) to match the 5-seed GA vs random table
    summary = (raw.groupby(["param", "value"])
                  .agg(mean=("best_fitness", "mean"),
                       std=("best_fitness", lambda x: x.std(ddof=0)),
                       best=("best_fitness", "max"),
                       evals=("evaluations", "mean"))
                  .reset_index())
    summary.to_csv(os.path.join(args.out, "sensitivity_summary.csv"), index=False)
    make_plot(summary, os.path.join(args.out, "sensitivity.png"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()