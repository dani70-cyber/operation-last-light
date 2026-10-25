"""Chromosome: per-drone waypoint genes (x, y, speed, sensor_power)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_ga.environment import Environment

SPEED_MIN, SPEED_MAX = 0.5, 1.5      # relative cruise speed
POWER_MIN, POWER_MAX = 0.2, 1.0      # thermal scanner power (0.2 = low, 1.0 = high)
MIN_WAYPOINTS, MAX_WAYPOINTS = 1, 10
CALLSIGNS = ("Falcon-1", "Shaheen-2", "Chai-Runner-3", "Raven-4", "Kite-5")


@dataclass
class Gene:
    """One waypoint plus how the drone behaves on the leg that ENDS here."""

    x: int
    y: int
    speed: float = 1.0
    sensor_power: float = 0.6

    @property
    def pos(self) -> tuple[int, int]:
        return self.x, self.y

    def copy(self) -> "Gene":
        return Gene(self.x, self.y, self.speed, self.sensor_power)


@dataclass
class DronePlan:
    """Ordered waypoints for one drone (depot -> genes -> depot). Variable length."""

    genes: list[Gene] = field(default_factory=list)

    def waypoints(self) -> list[tuple[int, int]]:
        return [g.pos for g in self.genes]

    def copy(self) -> "DronePlan":
        return DronePlan([g.copy() for g in self.genes])


@dataclass
class FleetChromosome:
    """A full candidate solution: one DronePlan per drone."""

    plans: list[DronePlan]
    fitness: float | None = None

    @property
    def n_drones(self) -> int:
        return len(self.plans)

    def copy(self) -> "FleetChromosome":
        return FleetChromosome([p.copy() for p in self.plans], self.fitness)


def random_gene(
    env: Environment, rng: np.random.Generator, near_depot: float | None = None
) -> Gene:
    """Random flyable waypoint; if near_depot is set, prefer cells within that radius."""
    x, y = env.random_flyable_cell(rng)
    if near_depot is not None:
        dx, dy = env.depot
        for _ in range(30):
            if np.hypot(x - dx, y - dy) <= near_depot:
                break
            x, y = env.random_flyable_cell(rng)
    return Gene(
        x,
        y,
        float(rng.uniform(SPEED_MIN, SPEED_MAX)),
        float(rng.uniform(POWER_MIN, POWER_MAX)),
    )


def random_chromosome(
    env: Environment,
    rng: np.random.Generator,
    n_drones: int = 3,
    n_waypoints: int = 5,
) -> FleetChromosome:
    """Random fleet; each drone gets 2..n_waypoints genes (not yet repaired)."""
    plans = []
    for _ in range(n_drones):
        n = int(rng.integers(2, max(3, n_waypoints + 1)))
        plans.append(
            DronePlan([random_gene(env, rng, near_depot=env.max_range / 3.0) for _ in range(n)])
        )
    return FleetChromosome(plans)