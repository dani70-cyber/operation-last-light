"""Random-search baseline with the same evaluation budget as the GA."""

from __future__ import annotations

import numpy as np

from uav_ga.chromosome import FleetChromosome, random_chromosome
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator, load_weights
from uav_ga.operators import repair


def run_random_search(
    env: Environment,
    budget: int,
    n_drones: int = 3,
    n_waypoints: int = 5,
    weights: dict[str, float] | None = None,
    seed: int = 0,
    chunk: int = 80,
) -> tuple[list[float], FleetChromosome]:
    """
    Sample `budget` random (repaired) fleets and keep the best.
    Returns (best-so-far history sampled every `chunk` evaluations, best fleet).
    Use chunk = GA pop_size so the curve lines up with GA generations.
    """
    rng = np.random.default_rng(seed + 1000)
    evaluate = Evaluator(env, weights if weights is not None else load_weights())
    best: FleetChromosome | None = None
    history: list[float] = []
    for i in range(1, budget + 1):
        c = repair(env, random_chromosome(env, rng, n_drones, n_waypoints))
        evaluate(c)
        if best is None or c.fitness > best.fitness:
            best = c
        if i % chunk == 0:
            history.append(best.fitness)
    if not history or budget % chunk != 0:
        history.append(best.fitness)
    return history, best