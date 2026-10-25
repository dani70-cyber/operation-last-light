"""Tests for the bug fix and new features in environment.py.

Drop this file next to your existing test_environment.py, or paste the tests
into it. Adjust the import below to match your package layout, for example:
    from src.environment import Environment
"""

import numpy as np
import pytest

from environment import Environment, _bresenham_line  # <-- adjust import path


def make_env(width=20, height=20, max_range=30.0, depot=(0, 0), priority=None):
    """Fully open map with optional priority layer."""
    shape = (height, width)
    return Environment(
        width=width,
        height=height,
        no_fly=np.zeros(shape, dtype=bool),
        hazard=np.zeros(shape, dtype=np.float64),
        priority=np.zeros(shape, dtype=np.float64) if priority is None else priority,
        depot=depot,
        max_range=max_range,
    )


# 1. Bresenham bug fix (downward lines used to walk the wrong way)
def test_bresenham_downward_diagonal():
    cells = list(_bresenham_line(0, 0, 3, 3))
    assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_bresenham_all_directions_reach_target():
    for x1, y1 in [(5, 5), (-5, 5), (5, -5), (-5, -5), (0, 7), (7, 0), (0, -7)]:
        cells = list(_bresenham_line(0, 0, x1, y1))
        assert cells[0] == (0, 0)
        assert cells[-1] == (x1, y1)


def test_segment_cells_downward_on_env():
    env = make_env()
    assert env.cells_along_segment(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]


# 2. within_range uses the full round trip
def test_within_range_short_route_true():
    env = make_env(max_range=30.0)
    # depot (0,0) -> (3,4) is 5, and back is 5, so 10 total
    assert env.within_range([(3, 4)])
    assert env.range_excess([(3, 4)]) == 0.0


def test_within_range_long_route_false():
    env = make_env(max_range=30.0)
    route = [(19, 19)]  # about 26.9 each way, so about 53.7 round trip
    assert not env.within_range(route)
    assert env.range_excess(route) == pytest.approx(
        2 * np.hypot(19, 19) - 30.0
    )


def test_path_length_return_to_depot_flag():
    env = make_env()
    assert env.path_length([(3, 4)]) == pytest.approx(5.0)
    assert env.path_length([(3, 4)], return_to_depot=True) == pytest.approx(10.0)


# 3. fleet coverage counts overlapping cells only once
def test_fleet_priority_overlap_counted_once():
    priority = np.zeros((20, 20), dtype=np.float64)
    priority[0, 1:6] = 1.0  # row y=0, x=1..5
    env = make_env(priority=priority)

    route = [(5, 0)]
    single = env.priority_covered(route)
    assert single == pytest.approx(5.0)

    fleet_of_two = env.fleet_priority_covered([route, route])
    assert fleet_of_two == pytest.approx(single)


def test_fleet_priority_different_routes_add_up():
    priority = np.zeros((20, 20), dtype=np.float64)
    priority[0, 1:4] = 1.0  # along the x axis
    priority[1:4, 0] = 2.0  # along the y axis
    env = make_env(priority=priority)

    total = env.fleet_priority_covered([[(3, 0)], [(0, 3)]])
    assert total == pytest.approx(3 * 1.0 + 3 * 2.0)


def test_sensor_radius_widens_coverage():
    priority = np.zeros((20, 20), dtype=np.float64)
    priority[2, 3] = 1.0  # two cells off the flight line
    env = make_env(priority=priority)
    route = [(5, 0)]
    assert env.priority_covered(route, sensor_radius=0) == 0.0
    assert env.priority_covered(route, sensor_radius=3) == pytest.approx(1.0)


# 4. layers are read-only
def test_layers_are_read_only():
    env = make_env()
    with pytest.raises(ValueError):
        env.no_fly[0, 0] = True
    with pytest.raises(ValueError):
        env.hazard[0, 0] = 1.0
    with pytest.raises(ValueError):
        env.priority[0, 0] = 1.0


def test_environment_does_not_modify_callers_array():
    arr = np.zeros((20, 20), dtype=bool)
    Environment(
        width=20,
        height=20,
        no_fly=arr,
        hazard=np.zeros((20, 20)),
        priority=np.zeros((20, 20)),
        depot=(0, 0),
    )
    arr[5, 5] = True  # the caller's own array must stay writable
    assert arr[5, 5]