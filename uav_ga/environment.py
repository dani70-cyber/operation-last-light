"""Grid world: hazard, priority, no-fly layers and map loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml

# Default mission limits (assignment suggests 50×50 or 100×100 grids).
DEFAULT_GRID_SIZE = 50
DEFAULT_MAX_RANGE = 120.0


class CellKind(IntEnum):
    """Static cell classification for blocked vs traversable space."""

    OPEN = 0
    BLOCKED = 1


@dataclass(frozen=True, slots=True, eq=False)
class Environment:
    """
    Discretized 2D mission space.

    Coordinates use (x, y) with x ∈ [0, width) and y ∈ [0, height).
    Layer arrays are indexed as ``layer[y, x]``.

    ``eq=False`` keeps identity-based equality/hash, because numpy arrays
    cannot be compared or hashed field-by-field.
    """

    width: int
    height: int
    no_fly: np.ndarray
    hazard: np.ndarray
    priority: np.ndarray
    depot: tuple[int, int]
    max_range: float = DEFAULT_MAX_RANGE

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be positive")
        if self.max_range <= 0:
            raise ValueError("max_range must be positive")
        expected = (self.height, self.width)

        # Own private, read-only copies so a GA bug can never edit the map.
        layers = {}
        for name, layer, dtype in (
            ("no_fly", self.no_fly, bool),
            ("hazard", self.hazard, np.float64),
            ("priority", self.priority, np.float64),
        ):
            arr = np.array(layer, dtype=dtype, copy=True)
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != {expected}")
            arr.setflags(write=False)
            layers[name] = arr
        for name, arr in layers.items():
            object.__setattr__(self, name, arr)

        dx, dy = self.depot
        if not self.in_bounds(dx, dy):
            raise ValueError(f"depot {self.depot} out of bounds")
        if not self.is_flyable(dx, dy):
            raise ValueError(f"depot {self.depot} is not flyable")

    # ------------------------------------------------------------------ basics

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_flyable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not bool(self.no_fly[y, x])

    def hazard_at(self, x: int, y: int) -> float:
        if not self.in_bounds(x, y):
            return 0.0
        return float(self.hazard[y, x])

    def priority_at(self, x: int, y: int) -> float:
        if not self.in_bounds(x, y):
            return 0.0
        return float(self.priority[y, x])

    # ---------------------------------------------------------------- segments

    def cells_along_segment(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """Grid cells visited when moving from (x0, y0) to (x1, y1) (inclusive)."""
        return list(_cells(int(x0), int(y0), int(x1), int(y1)))

    def segment_crosses_no_fly(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        for x, y in _cells(int(x0), int(y0), int(x1), int(y1)):
            if not self.is_flyable(x, y):
                return True
        return False

    def path_crosses_no_fly(self, waypoints: Sequence[tuple[int, int]]) -> bool:
        if len(waypoints) < 1:
            return False
        prev_x, prev_y = self.depot
        for x, y in waypoints:
            if not self.in_bounds(x, y):
                return True
            if self.segment_crosses_no_fly(prev_x, prev_y, x, y):
                return True
            prev_x, prev_y = x, y
        return False

    # ----------------------------------------------------------------- metrics

    def path_length(
        self,
        waypoints: Sequence[tuple[int, int]],
        return_to_depot: bool = False,
    ) -> float:
        """
        Euclidean path length from depot through waypoints (cell centers).

        With ``return_to_depot=True`` the leg back home is included, which is
        the distance that actually counts against ``max_range``.
        """
        if not waypoints:
            return 0.0
        total = 0.0
        px, py = self.depot
        for x, y in waypoints:
            total += _euclidean(px, py, x, y)
            px, py = x, y
        if return_to_depot:
            total += _euclidean(px, py, *self.depot)
        return total

    def within_range(self, waypoints: Sequence[tuple[int, int]]) -> bool:
        """True if the full round trip (depot -> waypoints -> depot) fits max_range."""
        return self.path_length(waypoints, return_to_depot=True) <= self.max_range

    def range_excess(self, waypoints: Sequence[tuple[int, int]]) -> float:
        """How far the round trip exceeds max_range (0.0 if it fits). Handy as a penalty."""
        over = self.path_length(waypoints, return_to_depot=True) - self.max_range
        return max(0.0, over)

    def path_hazard_exposure(self, waypoints: Sequence[tuple[int, int]]) -> float:
        """Sum of hazard values on unique cells touched along the flown path."""
        if not waypoints:
            return 0.0
        exposure = 0.0
        px, py = self.depot
        seen: set[tuple[int, int]] = set()
        for x, y in waypoints:
            for cx, cy in _cells(px, py, x, y):
                if (cx, cy) in seen:
                    continue
                seen.add((cx, cy))
                exposure += self.hazard_at(cx, cy)
            px, py = x, y
        return exposure

    # ---------------------------------------------------------------- coverage

    def covered_mask(
        self,
        waypoints: Sequence[tuple[int, int]],
        sensor_radius: float = 0.0,
    ) -> np.ndarray:
        """
        Boolean grid of cells a drone 'sees' while flying depot -> waypoints.

        ``sensor_radius=0`` means only the cells the path passes through.
        A larger radius widens the covered strip around the path.
        Combine drones with ``np.logical_or`` so shared cells are counted once.
        """
        mask = np.zeros(self.shape, dtype=bool)
        px, py = self.depot
        for x, y in waypoints:
            for cx, cy in _cells(px, py, x, y):
                if self.in_bounds(cx, cy):
                    mask[cy, cx] = True
            px, py = x, y

        if sensor_radius <= 0:
            return mask

        r = int(np.ceil(sensor_radius))
        r_sq = sensor_radius**2
        h, w = self.height, self.width
        out = mask.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if (dx == 0 and dy == 0) or dx * dx + dy * dy > r_sq:
                    continue
                src_y = slice(max(0, -dy), h - max(0, dy))
                dst_y = slice(max(0, dy), h - max(0, -dy))
                src_x = slice(max(0, -dx), w - max(0, dx))
                dst_x = slice(max(0, dx), w - max(0, -dx))
                out[dst_y, dst_x] |= mask[src_y, src_x]
        return out

    def priority_covered(
        self,
        waypoints: Sequence[tuple[int, int]],
        sensor_radius: float = 0.0,
    ) -> float:
        """Total priority value of unique cells covered by one drone's route."""
        return float(self.priority[self.covered_mask(waypoints, sensor_radius)].sum())

    def fleet_priority_covered(
        self,
        routes: Sequence[Sequence[tuple[int, int]]],
        sensor_radius: float = 0.0,
    ) -> float:
        """Priority covered by the whole fleet; overlapping cells count only once."""
        if not routes:
            return 0.0
        union = np.zeros(self.shape, dtype=bool)
        for route in routes:
            union |= self.covered_mask(route, sensor_radius)
        return float(self.priority[union].sum())

    @property
    def total_priority(self) -> float:
        """Maximum priority a perfect fleet could collect (for normalising scores)."""
        return float(self.priority.sum())

    # ----------------------------------------------------------------- helpers

    def nearest_flyable(
        self, x: int, y: int, max_radius: int | None = None
    ) -> tuple[int, int] | None:
        """Closest flyable cell by Euclidean distance (used for no-fly repair)."""
        if self.is_flyable(x, y):
            return x, y
        limit = max_radius if max_radius is not None else max(self.width, self.height)
        for r in range(1, limit + 1):
            found = any(self.is_flyable(nx, ny) for nx, ny in _chebyshev_ring(x, y, r))
            if not found:
                continue
            # A cell in a later ring's corner can be closer than one in this ring's
            # corner, so also scan out to ceil(r * sqrt(2)) and take the true minimum.
            scan = min(limit, int(np.ceil(r * np.sqrt(2))))
            best: tuple[int, int] | None = None
            best_d = float("inf")
            for rr in range(r, scan + 1):
                for nx, ny in _chebyshev_ring(x, y, rr):
                    if self.is_flyable(nx, ny):
                        d = _euclidean(x, y, nx, ny)
                        if d < best_d:
                            best, best_d = (nx, ny), d
            return best
        return None

    def random_flyable_cell(self, rng: np.random.Generator) -> tuple[int, int]:
        flyable = np.argwhere(~self.no_fly)
        if flyable.size == 0:
            raise RuntimeError("no flyable cells on map")
        idx = int(rng.integers(0, len(flyable)))
        y, x = flyable[idx]
        return int(x), int(y)

    def describe(self) -> str:
        """Short human-readable map summary for logs and the report."""
        total = self.width * self.height
        return (
            f"Map {self.width}x{self.height} | depot={self.depot} | "
            f"max_range={self.max_range:g}\n"
            f"  no-fly: {100 * self.no_fly.sum() / total:.1f}% of cells\n"
            f"  hazard: peak={self.hazard.max():.2f}, "
            f"{int((self.hazard > 0).sum())} hazardous cells\n"
            f"  priority: {int((self.priority > 0).sum())} cells, "
            f"total value={self.priority.sum():.1f}"
        )


# ------------------------------------------------------------------ geometry


def _euclidean(x0: int, y0: int, x1: int, y1: int) -> float:
    return float(np.hypot(x1 - x0, y1 - y0))


@lru_cache(maxsize=200_000)
def _cells(x0: int, y0: int, x1: int, y1: int) -> tuple[tuple[int, int], ...]:
    """Cached Bresenham cells; the GA re-checks the same segments constantly."""
    return tuple(_bresenham_line(x0, y0, x1, y1))


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterator[tuple[int, int]]:
    """Integer line rasterization (8-connected steps)."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1  # fixed: was `y0 < y0`, which is always False
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _chebyshev_ring(cx: int, cy: int, r: int) -> Iterator[tuple[int, int]]:
    """Cells exactly r steps away in Chebyshev distance (square ring)."""
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if max(abs(dx), abs(dy)) == r:
                yield cx + dx, cy + dy


def _disk_mask(
    height: int, width: int, center_x: int, center_y: int, radius: float
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return dist_sq <= radius**2


# ------------------------------------------------------------- map generation


def generate_map(
    width: int = DEFAULT_GRID_SIZE,
    height: int = DEFAULT_GRID_SIZE,
    *,
    seed: int | None = None,
    depot: tuple[int, int] | None = None,
    max_range: float = DEFAULT_MAX_RANGE,
    no_fly_density: float = 0.12,
    hazard_hotspots: int = 6,
    priority_sites: int = 8,
) -> Environment:
    """
    Build a reproducible random scenario for GA experiments.

    No-fly cells are scattered rectangles; hazard and priority are radial hotspots.
    """
    if width < 5 or height < 5:
        raise ValueError("map must be at least 5×5")
    rng = np.random.default_rng(seed)

    no_fly = np.zeros((height, width), dtype=bool)
    target = int(no_fly_density * width * height)
    while no_fly.sum() < target:
        bw = int(rng.integers(2, max(3, width // 8)))
        bh = int(rng.integers(2, max(3, height // 8)))
        x0 = int(rng.integers(0, max(1, width - bw)))
        y0 = int(rng.integers(0, max(1, height - bh)))
        no_fly[y0 : y0 + bh, x0 : x0 + bw] = True

    if depot is None:
        depot = (0, 0)
    dx, dy = depot
    no_fly[dy, dx] = False
    margin = 3
    no_fly[
        max(0, dy - margin) : min(height, dy + margin + 1),
        max(0, dx - margin) : min(width, dx + margin + 1),
    ] = False

    hazard = np.zeros((height, width), dtype=np.float64)
    for _ in range(hazard_hotspots):
        cx = float(rng.integers(0, width))
        cy = float(rng.integers(0, height))
        radius = float(rng.uniform(3, max(4.0, min(width, height) / 6)))
        peak = float(rng.uniform(0.5, 1.0))
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        hazard = np.maximum(hazard, peak * np.clip(1.0 - dist / (radius + 1e-6), 0.0, 1.0))

    priority = np.zeros((height, width), dtype=np.float64)
    for _ in range(priority_sites):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = float(rng.uniform(2, max(3.0, min(width, height) / 10)))
        weight = float(rng.uniform(0.6, 1.0))
        mask = _disk_mask(height, width, cx, cy, radius)
        priority[mask] = np.maximum(priority[mask], weight)

    hazard[no_fly] = 0.0
    priority[no_fly] = 0.0

    return Environment(
        width=width,
        height=height,
        no_fly=no_fly,
        hazard=hazard,
        priority=priority,
        depot=depot,
        max_range=max_range,
    )


# ---------------------------------------------------------------- map file I/O


def _layers_from_payload(
    width: int, height: int, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (height, width)

    def _layer(key: str, default: float, dtype: type) -> np.ndarray:
        if key not in payload:
            return np.full(shape, default, dtype=dtype)
        raw = payload[key]
        arr = np.asarray(raw, dtype=dtype)
        if arr.shape != shape:
            raise ValueError(f"{key} must be shape {shape}, got {arr.shape}")
        return arr

    no_fly = _layer("no_fly", 0, np.int8).astype(bool)
    hazard = _layer("hazard", 0.0, np.float64)
    priority = _layer("priority", 0.0, np.float64)
    return no_fly, hazard, priority


def _parse_depot(raw: Sequence[int] | tuple[int, int]) -> tuple[int, int]:
    if len(raw) != 2:
        raise ValueError("depot must be [x, y]")
    return int(raw[0]), int(raw[1])


def load_map(path: str | Path) -> Environment:
    """
    Load a map from ``.yaml`` / ``.yml``, ``.json``, or ``.npz``.

    YAML/JSON schema::

        width: 50
        height: 50
        depot: [0, 0]
        max_range: 120
        no_fly: [[0,1,...], ...]   # optional, 0/1 grid
        hazard: [[...], ...]       # optional
        priority: [[...], ...]    # optional
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        width = int(data["width"])
        height = int(data["height"])
        depot = (int(data["depot_x"]), int(data["depot_y"]))
        max_range = float(data["max_range"]) if "max_range" in data else DEFAULT_MAX_RANGE
        no_fly = np.asarray(data["no_fly"], dtype=bool)
        hazard = np.asarray(data["hazard"], dtype=np.float64)
        priority = np.asarray(data["priority"], dtype=np.float64)
        return Environment(
            width=width,
            height=height,
            no_fly=no_fly,
            hazard=hazard,
            priority=priority,
            depot=depot,
            max_range=max_range,
        )

    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise ValueError(f"unsupported map format: {suffix}")

    if not isinstance(payload, dict):
        raise ValueError("map file must contain a mapping at top level")

    width = int(payload["width"])
    height = int(payload["height"])
    depot = _parse_depot(payload.get("depot", [0, 0]))
    max_range = float(payload.get("max_range", DEFAULT_MAX_RANGE))
    no_fly, hazard, priority = _layers_from_payload(width, height, payload)

    return Environment(
        width=width,
        height=height,
        no_fly=no_fly,
        hazard=hazard,
        priority=priority,
        depot=depot,
        max_range=max_range,
    )


def save_map(env: Environment, path: str | Path) -> None:
    """Persist layers to ``.npz`` for fast reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        width=env.width,
        height=env.height,
        depot_x=env.depot[0],
        depot_y=env.depot[1],
        max_range=env.max_range,
        no_fly=env.no_fly.astype(np.uint8),
        hazard=env.hazard,
        priority=env.priority,
    )