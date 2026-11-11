"""Voxels to collision geometry: the merge must be exact, or SCR is fiction.

Physics has to see neither more nor less than the map the planner used.
If the box decomposition loses a cell, a robot drives through a wall the
planner routed around; if it adds one, a corridor closes that the planner
was entitled to use. Both turn an executed run into a measurement of the
converter rather than of the plan.
"""
from __future__ import annotations

import numpy as np
import pytest

from bodynav.sim.colliders import ColliderBoxes, boxes_to_mesh, merge_boxes

PITCH = 0.25


def _volume(occupied) -> float:
    return float(occupied.sum()) * PITCH ** 3


def test_a_solid_block_becomes_one_box():
    occ = np.zeros((6, 5, 4), dtype=bool)
    occ[1:5, 1:4, 0:3] = True
    boxes = merge_boxes(occ, np.zeros(3), PITCH)
    assert len(boxes) == 1
    assert boxes.volume == pytest.approx(_volume(occ))


def test_the_union_is_exactly_the_occupied_set():
    rng = np.random.default_rng(0)
    occ = rng.random((12, 10, 6)) < 0.35
    boxes = merge_boxes(occ, np.array([1.0, -2.0, 0.5]), PITCH)
    assert boxes.meta["complete"]
    assert boxes.volume == pytest.approx(_volume(occ))

    # every occupied cell centre is inside some box, and no empty one is
    origin = np.array([1.0, -2.0, 0.5])
    idx = np.argwhere(occ)
    centres = origin + (idx + 0.5) * PITCH
    covered = np.any(np.all((centres[:, None, :] >= boxes.lower[None]) &
                            (centres[:, None, :] <= boxes.upper[None]), axis=2), axis=1)
    assert covered.all()

    empty = np.argwhere(~occ)[:200]
    centres = origin + (empty + 0.5) * PITCH
    covered = np.any(np.all((centres[:, None, :] >= boxes.lower[None]) &
                            (centres[:, None, :] <= boxes.upper[None]), axis=2), axis=1)
    assert not covered.any()


def test_merging_actually_reduces_the_prim_count():
    occ = np.zeros((40, 40, 3), dtype=bool)
    occ[:, 0, :] = True          # a long wall
    occ[:, -1, :] = True
    boxes = merge_boxes(occ, np.zeros(3), PITCH)
    assert len(boxes) <= 4
    assert boxes.meta["reduction"] > 0.9


def test_boxes_are_axis_aligned_and_positive():
    rng = np.random.default_rng(1)
    occ = rng.random((8, 8, 8)) < 0.2
    boxes = merge_boxes(occ, np.zeros(3), PITCH)
    assert np.all(boxes.upper > boxes.lower)
    # every box edge is a whole number of cells
    edges = (boxes.upper - boxes.lower) / PITCH
    assert np.allclose(edges, np.round(edges))


def test_empty_grid_yields_no_boxes():
    boxes = merge_boxes(np.zeros((4, 4, 4), dtype=bool), np.zeros(3), PITCH)
    assert len(boxes) == 0 and boxes.volume == 0.0


def test_mesh_has_six_faces_per_box():
    occ = np.zeros((4, 4, 4), dtype=bool)
    occ[0, 0, 0] = True
    occ[3, 3, 3] = True
    boxes = merge_boxes(occ, np.zeros(3), PITCH)
    points, counts, indices = boxes_to_mesh(boxes)
    assert len(boxes) == 2
    assert points.shape == (16, 3)
    assert len(counts) == 12 and set(counts) == {4}
    assert len(indices) == 48
    assert indices.max() < len(points)


def test_mesh_bounds_match_the_boxes():
    occ = np.zeros((5, 5, 5), dtype=bool)
    occ[1:4, 1:4, 1:4] = True
    origin = np.array([-1.0, 2.0, 0.0])
    boxes = merge_boxes(occ, origin, PITCH)
    points, _, _ = boxes_to_mesh(boxes)
    assert np.allclose(points.min(axis=0), boxes.lower.min(axis=0))
    assert np.allclose(points.max(axis=0), boxes.upper.max(axis=0))


def test_rejects_a_non_3d_grid():
    with pytest.raises(ValueError):
        merge_boxes(np.zeros((4, 4), dtype=bool), np.zeros(3), PITCH)
