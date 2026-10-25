"""Operation Last Light: AI Commander (GA) vs Rookie Survivor (random search)."""

from __future__ import annotations

import argparse

import numpy as np

from uav_ga.baseline import run_random_search
from uav_ga.chromosome import CALLSIGNS
from uav_ga.environment import generate_map
from uav_ga.fitness import Evaluator
from uav_ga.ga import Hyperparameters, run_ga


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=5, help="independent GA runs")
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--max-range", type=float, default=120.0)
    ap.add_argument("--depot", type=int, nargs=2, default=[25, 25], help="depot x y (default: map centre)")
    args = ap.parse_args()

    env = generate_map(seed=args.map_seed, depot=tuple(args.depot), max_range=args.max_range)
    print(env.describe())
    if env.total_priority <= 0:
        raise SystemExit("Map has no priority cells: coverage would always be 0.")

    ev = Evaluator(env)
    ga_scores, rs_scores = [], []
    best_overall = None
    for s in range(args.seeds):
        hp = Hyperparameters(generations=args.generations, pop_size=args.pop, seed=s)
        print(f"\n=== Run {s} (seed {s}) ===")
        res = run_ga(env, hp, verbose=(s == 0))
        _, rs_best = run_random_search(
            env, res.evaluations, hp.n_drones, hp.n_waypoints, seed=s, chunk=hp.pop_size
        )
        ga_scores.append(res.best.fitness)
        rs_scores.append(rs_best.fitness)
        print(f"  AI Commander {res.best.fitness:.3f} | Rookie Survivor {rs_best.fitness:.3f}"
              f" | evals {res.evaluations}")
        if best_overall is None or res.best.fitness > best_overall.fitness:
            best_overall = res.best

    print("\n=== Summary ===")
    print(f"AI Commander    : {np.mean(ga_scores):.3f} +/- {np.std(ga_scores):.3f}")
    print(f"Rookie Survivor : {np.mean(rs_scores):.3f} +/- {np.std(rs_scores):.3f}")

    print("\nBest fleet breakdown:")
    for k, v in ev.breakdown(best_overall).items():
        print(f"  {k:9s} {v:7.3f}")
    for name, plan in zip(CALLSIGNS, best_overall.plans):
        print(f"  {name}: {plan.waypoints()}")
    if ev.breakdown(best_overall)["coverage"] == 0:
        print("WARNING: coverage is 0 for the best fleet. Check the fitness / map.")


if __name__ == "__main__":
    main()