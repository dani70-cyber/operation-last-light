"""Multi-objective fitness (weighted sum of normalised terms)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from uav_ga.chromosome import POWER_MIN, FleetChromosome
from uav_ga.environment import Environment

DEFAULT_WEIGHTS = {
    "coverage": 0.30,
    "detection": 0.25,
    "energy": 0.15,
    "flight_time": 0.10,
    "hazard_exposure": 0.20,
    "penalty_no_fly": 1.0,
    "penalty_range_overshoot": 0.5,
}
_WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "config" / "default_weights.yaml"


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    """Read weights from YAML; missing keys fall back to DEFAULT_WEIGHTS."""
    p = Path(path) if path else _WEIGHTS_FILE
    weights = dict(DEFAULT_WEIGHTS)
    if p.is_file():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        weights.update({k: float(v) for k, v in data.items() if k in weights})
    return weights


# ---- drone model: how gene values turn into physical effects ----------------


def sensor_radius(power: float) -> float:
    """Thermal scanner radius in cells: low power ~1.6, high power 4.0."""
    return 1.0 + 3.0 * power


def detection_prob(power: float, speed: float) -> float:
    """Chance a seen priority cell is actually detected. Stronger scan helps, speed hurts."""
    return float(np.clip(0.3 + 0.7 * power - 0.15 * (speed - 1.0), 0.05, 1.0))


def energy_rate(speed: float, power: float) -> float:
    """Fuel per cell flown. Speed costs quadratically, scanner costs linearly."""
    return 0.6 * speed**2 + 0.4 * power


_DISK_CACHE: dict[float, np.ndarray] = {}


def _disk_offsets(radius: float) -> np.ndarray:
    key = round(radius * 2) / 2
    if key not in _DISK_CACHE:
        r = int(np.ceil(key))
        d = np.arange(-r, r + 1)
        dy, dx = np.meshgrid(d, d, indexing="ij")
        keep = dx * dx + dy * dy <= key * key
        _DISK_CACHE[key] = np.stack([dx[keep], dy[keep]], axis=1)
    return _DISK_CACHE[key]


def effective_length(env: Environment, plan) -> float:
    """Round-trip distance of a plan (depot -> waypoints -> depot)."""
    return env.path_length(plan.waypoints(), return_to_depot=True)


class Evaluator:
    """Scores fleets. Callable: evaluator(chrom) sets and returns chrom.fitness."""

    def __init__(self, env: Environment, weights: dict[str, float] | None = None):
        self.env = env
        self.w = dict(weights) if weights is not None else load_weights()
        self.count = 0  # number of fitness evaluations (the GA / random-search budget)

    def __call__(self, chrom: FleetChromosome) -> float:
        self.count += 1
        chrom.fitness = self.breakdown(chrom)["fitness"]
        return chrom.fitness

    def breakdown(self, chrom: FleetChromosome) -> dict[str, float]:
        env, w = self.env, self.w
        dx, dy = env.depot
        covered = np.zeros(env.shape, dtype=bool)
        quality = np.zeros(env.shape, dtype=np.float64)  # best detection prob per cell

        energies, times = [], []
        hazard_total = range_excess = 0.0
        legs = nofly_legs = 0

        for plan in chrom.plans:
            px, py = dx, dy
            energy = time = 0.0
            for g in plan.genes:
                legs += 1
                if env.segment_crosses_no_fly(px, py, g.x, g.y):
                    nofly_legs += 1
                length = float(np.hypot(g.x - px, g.y - py))
                energy += length * energy_rate(g.speed, g.sensor_power)
                time += length / g.speed

                # sensor footprint: every cell on the leg, widened by the scanner radius
                cells = np.asarray(env.cells_along_segment(px, py, g.x, g.y))
                pts = (cells[:, None, :] + _disk_offsets(sensor_radius(g.sensor_power))[None]).reshape(-1, 2)
                ok = (pts[:, 0] >= 0) & (pts[:, 0] < env.width) & (pts[:, 1] >= 0) & (pts[:, 1] < env.height)
                xs, ys = pts[ok, 0], pts[ok, 1]
                covered[ys, xs] = True
                q = detection_prob(g.sensor_power, g.speed)
                quality[ys, xs] = np.maximum(quality[ys, xs], q)
                px, py = g.x, g.y

            if plan.genes:  # return leg: scanner idle, last cruise speed
                last = plan.genes[-1]
                legs += 1
                if env.segment_crosses_no_fly(px, py, dx, dy):
                    nofly_legs += 1
                back = float(np.hypot(px - dx, py - dy))
                energy += back * energy_rate(last.speed, POWER_MIN)
                time += back / last.speed

            wps = plan.waypoints()
            hazard_total += env.path_hazard_exposure(wps)
            range_excess += env.range_excess(wps)
            energies.append(energy)
            times.append(time)

        n = max(1, chrom.n_drones)
        total = env.total_priority
        coverage = float(env.priority[covered].sum() / total) if total > 0 else 0.0
        detection = float((env.priority * quality).sum() / total) if total > 0 else 0.0
        energy_n = float(np.mean(energies)) / env.max_range if energies else 0.0
        time_n = float(max(times)) / env.max_range if times else 0.0   # fleet makespan
        hazard_n = min(1.0, hazard_total / (n * 10.0))
        nofly_n = nofly_legs / max(1, legs)
        range_n = range_excess / (n * env.max_range)

        fitness = (
            w["coverage"] * coverage
            + w["detection"] * detection
            - w["energy"] * energy_n
            - w["flight_time"] * time_n
            - w["hazard_exposure"] * hazard_n
            - w["penalty_no_fly"] * nofly_n
            - w["penalty_range_overshoot"] * range_n
        )
        return {
            "coverage": coverage,
            "detection": detection,
            "energy": energy_n,
            "time": time_n,
            "hazard": hazard_n,
            "no_fly": nofly_n,
            "range": range_n,
            "fitness": float(fitness),
        }