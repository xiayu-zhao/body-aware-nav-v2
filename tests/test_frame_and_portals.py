"""Building frame and portal detection.

Two fixes forced by the as-built scan, both of which change results rather
than appearance, so both are pinned by tests.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from bodynav.scene_graph.frame import (
    from_building_frame,
    principal_axis,
    rotation,
    to_building_frame,
)
from bodynav.scene_graph.geometric import find_portals


def _rect_building(angle_deg: float, seed: int = 0) -> np.ndarray:
    """A 30 x 12 m rectangular floorplate, rotated by ``angle_deg``."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(0, 30, 20000), rng.uniform(0, 12, 20000),
                             np.zeros(20000)])
    walls = []
    for x0, x1, y0, y1 in ((0, 30, 0, 0), (0, 30, 12, 12), (0, 0, 0, 12), (30, 30, 0, 12)):
        n = 6000
        walls.append(np.column_stack([
            rng.uniform(x0, x1, n) if x1 > x0 else np.full(n, x0),
            rng.uniform(y0, y1, n) if y1 > y0 else np.full(n, y0),
            rng.uniform(0, 2.6, n)]))
    xyz = np.vstack([floor, *walls])
    t = np.radians(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.column_stack([xyz[:, :2] @ rot.T, xyz[:, 2]])


@pytest.mark.parametrize("angle", [0.0, 12.5, 43.0, -30.0, 88.0])
def test_principal_axis_recovers_the_rotation(angle):
    xyz = _rect_building(angle)
    found = principal_axis(xyz[np.abs(xyz[:, 2]) < 0.05][:, :2])
    # the rectangle has four equivalent sides, so compare modulo 90 deg
    err = (found - angle + 45.0) % 90.0 - 45.0
    assert abs(err) < 1.5


def test_building_frame_shrinks_the_axis_aligned_footprint():
    xyz = _rect_building(43.0)
    angle = principal_axis(xyz[np.abs(xyz[:, 2]) < 0.05][:, :2])
    before = np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 1])
    aligned = to_building_frame(xyz, angle)
    after = np.ptp(aligned[:, 0]) * np.ptp(aligned[:, 1])
    assert after < 0.75 * before
    assert sorted(np.round([np.ptp(aligned[:, 0]), np.ptp(aligned[:, 1])])) == [12.0, 30.0]


def test_building_frame_roundtrips_exactly():
    xyz = _rect_building(17.0)
    back = from_building_frame(to_building_frame(xyz, 17.0), 17.0)
    assert np.abs(back - xyz).max() < 1e-9


def test_rotation_is_orthonormal():
    r = rotation(31.7)
    assert np.allclose(r @ r.T, np.eye(2))
    assert np.isclose(np.linalg.det(r), 1.0)


# -- portals ------------------------------------------------------------
@pytest.fixture
def two_rooms():
    """Two 6 x 6 m rooms sharing a wall, with a shut door in the middle.

    The door leaf is 5 cm; the wall is 24 cm. Only the door may be opened.
    """
    rng = np.random.default_rng(1)
    parts = [np.column_stack([rng.uniform(0, 6, 12000), rng.uniform(0, 13, 12000),
                              np.zeros(12000)])]  # floor across both rooms
    # dividing wall at y = 6.0, 0.24 m thick, with a 0.9 m doorway at x in [2.5, 3.4]
    n = 20000
    wx = rng.uniform(0, 6, n)
    wy = rng.uniform(5.88, 6.12, n)
    wz = rng.uniform(0.05, 2.5, n)
    keep = ~((wx > 2.5) & (wx < 3.4))
    parts.append(np.column_stack([wx[keep], wy[keep], wz[keep]]))
    # the door leaf filling that doorway: 5 cm thick
    m = 4000
    parts.append(np.column_stack([rng.uniform(2.5, 3.4, m),
                                  rng.uniform(5.99, 6.04, m),
                                  rng.uniform(0.05, 2.0, m)]))
    return np.vstack(parts)


def _grids(xyz, cell=0.25):
    lo = xyz[:, :2].min(axis=0)
    dims = np.maximum(np.ceil((xyz[:, :2].max(axis=0) - lo) / cell).astype(int), 1)
    def count(sel):
        ij = np.clip(np.floor((xyz[sel, :2] - lo) / cell).astype(int), 0, dims - 1)
        g = np.zeros(tuple(dims), np.int32)
        np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
        return g
    band = (xyz[:, 2] > 0.15) & (xyz[:, 2] < 2.0)
    occ, allr = count(band), count(np.ones(len(xyz), bool))
    fl = count(np.abs(xyz[:, 2]) < 0.10)
    seen = ndimage.binary_closing(allr > 0, np.ones((5, 5)))
    occl = occ / np.maximum(occ + fl, 1)
    free = seen & (occl <= 0.3)
    return free, seen, occl, lo, cell


def _component_areas(free, cell):
    lab, _ = ndimage.label(free)
    s = np.bincount(lab.ravel())
    s[0] = 0
    return np.sort(s)[::-1] * cell * cell


def test_a_shut_door_is_opened_and_reported(two_rooms):
    free, seen, occl, lo, cell = _grids(two_rooms)
    before = _component_areas(free, cell)
    assert before[1] > 5.0, "fixture must start as two separate rooms"

    after_mask, portals = find_portals(two_rooms, (0.0, 2.6), free, seen, occl,
                                       lo, cell, min_area_m2=5.0)
    after = _component_areas(after_mask, cell)
    assert len(portals) == 1
    assert after[0] > before[0] + before[1] - 1.0     # the rooms merged
    assert after[1] < 5.0

    p = portals[0]
    assert p["thickness_m"] < 0.12                    # a door leaf, not masonry
    assert 2.0 < p["centre_xy"][0] < 4.0              # where the doorway is
    assert 5.5 < p["centre_xy"][1] < 6.5


def test_masonry_is_never_opened(two_rooms):
    """Remove the door leaf and thicken the wall: nothing may be bridged."""
    rng = np.random.default_rng(2)
    xyz = two_rooms[~((two_rooms[:, 1] > 5.95) & (two_rooms[:, 1] < 6.08)
                      & (two_rooms[:, 0] > 2.5) & (two_rooms[:, 0] < 3.4))]
    n = 8000
    solid = np.column_stack([rng.uniform(2.5, 3.4, n), rng.uniform(5.88, 6.12, n),
                             rng.uniform(0.05, 2.5, n)])
    xyz = np.vstack([xyz, solid])

    free, seen, occl, lo, cell = _grids(xyz)
    before = _component_areas(free, cell)
    after_mask, portals = find_portals(xyz, (0.0, 2.6), free, seen, occl, lo, cell,
                                       min_area_m2=5.0)
    assert portals == []
    assert np.allclose(_component_areas(after_mask, cell)[:2], before[:2])


def test_a_short_barrier_is_not_a_door(two_rooms):
    """Replace the door leaf with a waist-high rail: thin, but not a door."""
    rng = np.random.default_rng(3)
    xyz = two_rooms[~((two_rooms[:, 1] > 5.95) & (two_rooms[:, 1] < 6.08)
                      & (two_rooms[:, 0] > 2.5) & (two_rooms[:, 0] < 3.4))]
    rail = np.column_stack([rng.uniform(2.5, 3.4, 3000), rng.uniform(5.99, 6.04, 3000),
                            rng.uniform(0.4, 1.1, 3000)])
    xyz = np.vstack([xyz, rail])
    free, seen, occl, lo, cell = _grids(xyz)
    _, portals = find_portals(xyz, (0.0, 2.6), free, seen, occl, lo, cell,
                              min_area_m2=5.0)
    assert portals == []


def test_portals_respect_the_minimum_area(two_rooms):
    """A pocket smaller than min_area is not worth opening a door for."""
    free, seen, occl, lo, cell = _grids(two_rooms)
    _, portals = find_portals(two_rooms, (0.0, 2.6), free, seen, occl, lo, cell,
                              min_area_m2=1000.0)
    assert portals == []
