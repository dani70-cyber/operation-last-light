"""Selection, spatial crossover, mutation, and no-fly / range repair."""

from __future__ import annotations

import numpy as np

from uav_ga.chromosome import (
    MAX_WAYPOINTS,
    MIN_WAYPOINTS,
    POWER_MAX,
    POWER_MIN,
    SPEED_MAX,
    SPEED_MIN,
    DronePlan,
    FleetChromosome,
    Gene,
    random_gene,
)
from uav_ga.environment import Environment


# --------------------------------------------------------------- selection


def tournament_select(
    population: list[FleetChromosome], rng: np.random.Generator, k: int = 3
) -> FleetChromosome:
    """Pick k random individuals, return a copy of the fittest. Larger k = more pressure."""
    idx = rng.integers(0, len(population), size=k)
    best = max((population[i] for i in idx), key=lambda c: c.fitness)
    return best.copy()


# --------------------------------------------------------------- crossover


def crossover(
    a: FleetChromosome,
    b: FleetChromosome,
    rng: np.random.Generator,
    segment_rate: float = 0.5,
    blend_rate: float = 0.3,
) -> tuple[FleetChromosome, FleetChromosome]:
    """
    Spatial crossover on three levels:
      1. drone level   - each drone slot may swap its whole plan between parents
      2. segment swap  - swap waypoint tails after a random cut (variable length)
      3. geometric blend - two waypoints are replaced by points on the line between them
    """
    c1, c2 = a.copy(), b.copy()
    for i in range(min(c1.n_drones, c2.n_drones)):
        if rng.random() < 0.5:
            c1.plans[i], c2.plans[i] = c2.plans[i], c1.plans[i]
        g1, g2 = c1.plans[i].genes, c2.plans[i].genes

        if rng.random() < segment_rate and len(g1) > 1 and len(g2) > 1:
            cut1 = int(rng.integers(1, len(g1)))
            cut2 = int(rng.integers(1, len(g2)))
            g1[cut1:], g2[cut2:] = g2[cut2:], g1[cut1:]
            c1.plans[i].genes = g1[:MAX_WAYPOINTS]
            c2.plans[i].genes = g2[:MAX_WAYPOINTS]
            g1, g2 = c1.plans[i].genes, c2.plans[i].genes

        if rng.random() < blend_rate and g1 and g2:
            k1 = int(rng.integers(0, len(g1)))
            k2 = int(rng.integers(0, len(g2)))
            t = float(rng.random())
            x1, y1, x2, y2 = g1[k1].x, g1[k1].y, g2[k2].x, g2[k2].y
            g1[k1].x, g1[k1].y = int(round(t * x1 + (1 - t) * x2)), int(round(t * y1 + (1 - t) * y2))
            g2[k2].x, g2[k2].y = int(round((1 - t) * x1 + t * x2)), int(round((1 - t) * y1 + t * y2))

    c1.fitness = c2.fitness = None
    return c1, c2


# ---------------------------------------------------------------- mutation


def mutate(
    chrom: FleetChromosome,
    env: Environment,
    rng: np.random.Generator,
    rate: float = 0.15,
    step: float = 4.0,
) -> FleetChromosome:
    """Gaussian jitter on position/speed/power, plus insert / delete / swap waypoints."""
    for plan in chrom.plans:
        for g in plan.genes:
            if rng.random() < rate:
                g.x = int(round(np.clip(g.x + rng.normal(0, step), 0, env.width - 1)))
                g.y = int(round(np.clip(g.y + rng.normal(0, step), 0, env.height - 1)))
            if rng.random() < rate:
                g.speed = float(np.clip(g.speed + rng.normal(0, 0.2), SPEED_MIN, SPEED_MAX))
            if rng.random() < rate:
                g.sensor_power = float(
                    np.clip(g.sensor_power + rng.normal(0, 0.15), POWER_MIN, POWER_MAX)
                )

        if rng.random() < rate * 0.5 and len(plan.genes) < MAX_WAYPOINTS:
            pos = int(rng.integers(0, len(plan.genes) + 1))
            plan.genes.insert(pos, random_gene(env, rng, near_depot=env.max_range / 3.0))
        if rng.random() < rate * 0.5 and len(plan.genes) > MIN_WAYPOINTS:
            plan.genes.pop(int(rng.integers(0, len(plan.genes))))
        if rng.random() < rate * 0.5 and len(plan.genes) > 1:
            i, j = rng.choice(len(plan.genes), size=2, replace=False)
            plan.genes[i], plan.genes[j] = plan.genes[j], plan.genes[i]

    chrom.fitness = None
    return chrom


# ------------------------------------------------------------------ repair


def fallback_gene(env: Environment) -> Gene:
    """Safe gene: nearest flyable cell to the depot that is not the depot."""
    dx, dy = env.depot
    for r in range(1, max(env.width, env.height)):
        for cx in range(dx - r, dx + r + 1):
            for cy in range(dy - r, dy + r + 1):
                if (cx, cy) != (dx, dy) and env.is_flyable(cx, cy):
                    return Gene(cx, cy)
    return Gene(dx, dy)


def _repair_plan(env: Environment, plan: DronePlan) -> None:
    """Make one plan legal: in bounds, flyable, no no-fly crossings, round trip within range."""
    for _ in range(12):
        fixed: list[Gene] = []
        px, py = env.depot
        for g in plan.genes:
            g.x = int(np.clip(g.x, 0, env.width - 1))
            g.y = int(np.clip(g.y, 0, env.height - 1))
            cell = env.nearest_flyable(g.x, g.y)
            if cell is None:
                continue
            g.x, g.y = cell

            # If the straight leg clips a no-fly zone, stop just before it.
            if env.segment_crosses_no_fly(px, py, g.x, g.y):
                last_ok = (px, py)
                for cx, cy in env.cells_along_segment(px, py, g.x, g.y):
                    if not env.is_flyable(cx, cy):
                        break
                    last_ok = (cx, cy)
                g.x, g.y = last_ok
            if (g.x, g.y) == (px, py):
                continue  # zero-length leg: drop the gene
            fixed.append(g)
            px, py = g.x, g.y

        plan.genes = fixed[:MAX_WAYPOINTS]
        if not plan.genes:
            plan.genes = [fallback_gene(env)]

        wps = plan.waypoints()
        lx, ly = wps[-1]
        legal = (
            not env.path_crosses_no_fly(wps)
            and not env.segment_crosses_no_fly(lx, ly, *env.depot)  # return leg
            and env.within_range(wps)
        )
        if legal:
            return

        # Not legal yet: drop the last waypoint, or pull a lone waypoint toward home.
        if len(plan.genes) > MIN_WAYPOINTS:
            plan.genes.pop()
        else:
            g = plan.genes[0]
            g.x = int(round(env.depot[0] + 0.7 * (g.x - env.depot[0])))
            g.y = int(round(env.depot[1] + 0.7 * (g.y - env.depot[1])))

    plan.genes = [fallback_gene(env)]  # guaranteed-legal last resort


def repair(env: Environment, chrom: FleetChromosome) -> FleetChromosome:
    """Repair every drone plan in place and invalidate cached fitness."""
    for plan in chrom.plans:
        _repair_plan(env, plan)
    chrom.fitness = None
    return chrom