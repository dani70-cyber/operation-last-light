"""Main genetic algorithm loop: init, evolve, terminate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_ga.chromosome import FleetChromosome, random_chromosome
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator, load_weights
from uav_ga.operators import crossover, mutate, repair, tournament_select


@dataclass
class Hyperparameters:
    n_drones: int = 3
    n_waypoints: int = 5
    pop_size: int = 80
    generations: int = 150
    elite: int = 2
    tournament_k: int = 3             # selection pressure
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12       # base rate, adapted upward when stuck
    mutation_step: float = 4.0        # base jitter in cells, adapted upward when stuck
    immigrant_rate: float = 0.05      # fraction of population replaced by fresh randoms
    stagnation_boost: int = 8         # generations without gain before mutation ramps up
    stagnation_stop: int = 60         # generations without gain before early stop
    seed: int = 0


@dataclass
class GAResult:
    best: FleetChromosome
    best_history: list[float] = field(default_factory=list)
    mean_history: list[float] = field(default_factory=list)
    snapshots: list[tuple[int, FleetChromosome]] = field(default_factory=list)  # for GIF/heatmaps
    evaluations: int = 0
    generations_run: int = 0


def run_ga(
    env: Environment,
    hp: Hyperparameters | None = None,
    weights: dict[str, float] | None = None,
    snapshot_every: int = 10,
    verbose: bool = True,
) -> GAResult:
    hp = hp or Hyperparameters()
    rng = np.random.default_rng(hp.seed)
    evaluate = Evaluator(env, weights if weights is not None else load_weights())

    def fresh() -> FleetChromosome:
        c = repair(env, random_chromosome(env, rng, hp.n_drones, hp.n_waypoints))
        evaluate(c)
        return c

    # ---- init
    population = sorted((fresh() for _ in range(hp.pop_size)), key=lambda c: c.fitness, reverse=True)
    best = population[0].copy()
    result = GAResult(best=best)
    result.best_history.append(best.fitness)
    result.mean_history.append(float(np.mean([c.fitness for c in population])))
    result.snapshots.append((0, best.copy()))
    stagnant = 0
    n_imm = int(round(hp.immigrant_rate * hp.pop_size))

    # ---- evolve
    for gen in range(1, hp.generations + 1):
        boost = 1.0 + min(3.0, stagnant / max(1, hp.stagnation_boost))  # adaptive mutation
        rate = min(0.5, hp.mutation_rate * boost)
        step = hp.mutation_step * boost

        next_pop = [c.copy() for c in population[: hp.elite]]  # elitism
        while len(next_pop) < hp.pop_size - n_imm:
            p1 = tournament_select(population, rng, hp.tournament_k)
            p2 = tournament_select(population, rng, hp.tournament_k)
            if rng.random() < hp.crossover_rate:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            for child in (c1, c2):
                mutate(child, env, rng, rate, step)
                repair(env, child)
                evaluate(child)
                next_pop.append(child)
        next_pop = next_pop[: hp.pop_size - n_imm]
        next_pop += [fresh() for _ in range(n_imm)]          # random immigrants keep diversity
        population = sorted(next_pop, key=lambda c: c.fitness, reverse=True)

        if population[0].fitness > best.fitness + 1e-9:
            best = population[0].copy()
            stagnant = 0
        else:
            stagnant += 1

        result.best_history.append(best.fitness)
        result.mean_history.append(float(np.mean([c.fitness for c in population])))
        result.generations_run = gen
        if gen % snapshot_every == 0:
            result.snapshots.append((gen, best.copy()))
        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"  gen {gen:4d} | best {best.fitness:7.3f} | mean {result.mean_history[-1]:7.3f}"
                  f" | mut {rate:.2f}")

        # ---- terminate
        if stagnant >= hp.stagnation_stop:
            if verbose:
                print(f"  early stop at gen {gen} (no improvement for {stagnant} generations)")
            break

    result.best = best
    result.snapshots.append((result.generations_run, best.copy()))
    result.evaluations = evaluate.count
    return result