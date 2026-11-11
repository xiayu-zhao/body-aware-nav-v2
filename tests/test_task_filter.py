"""Task-guided filtering (Sec. III-B.2): the semantics-to-resolution link.

V_t is where the paper's claim actually bites -- grounded semantics decide
*where* density is preserved, and adaptive voxelization decides how finely.
These tests pin the properties that claim depends on: the corridor widens
near hazards and not elsewhere, hazard volumes enter the mask through their
recorded clearance radius, and the priority merge keeps the inside dense
while thinning the background to a stated fraction.
"""
from __future__ import annotations

import numpy as np
import pytest

from bodynav.abstraction.task_filter import build_corridor, priority_merge, task_mask
from bodynav.scene_graph.attributes import Affordances, HazardParams, NodeAttributes
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints


def _attrs(centre, half=(0.5, 0.5, 0.5), semantic="clutter", hazard=None):
    centre = np.asarray(centre, dtype=float)
    half = np.asarray(half, dtype=float)
    return NodeAttributes(
        centroid=centre, aabb_min=centre - half, aabb_max=centre + half,
        semantic=semantic,
        affordances=Affordances(hazardous=hazard is not None),
        hazard=hazard,
    )


@pytest.fixture
def graph():
    """A 20 m corridor with one hazard (a cone) beside it at x = 10."""
    cone = HazardParams(clearance_radius=1.5, risk_weight=0.8,
                        hazard_class="exclusion_marker")
    attrs = {
        "floor_0": _attrs([10, 0, 1.25], (10, 5, 1.25), "floor"),
        "area_0_0": _attrs([0, 0, 1.0], (2, 2, 1.0), "area"),
        "area_0_1": _attrs([20, 0, 1.0], (2, 2, 1.0), "area"),
        "spot_0_0_0": _attrs([10, 1.0, 0.3], (0.2, 0.2, 0.3), "cone", cone),
    }
    return BodyNavSceneGraph.build(LabeledPoints.from_xyz(np.zeros((1, 3))), attrs)


def test_corridor_runs_between_the_anchors(graph):
    c = build_corridor(graph, "area_0_0", "area_0_1", [], base_width=2.0)
    assert np.allclose(c.centerline[0], graph.attributes("area_0_0").centroid)
    assert np.allclose(c.centerline[-1], graph.attributes("area_0_1").centroid)
    assert len(c.centerline) == len(c.width)
    # samples are dense enough for the width to vary along the route
    assert len(c.centerline) >= 20


def test_corridor_widens_near_hazards_and_not_elsewhere(graph):
    c = build_corridor(graph, "area_0_0", "area_0_1", [], base_width=2.0,
                       hazard_expand=1.5)
    x = c.centerline[:, 0]
    near = np.abs(x - 10.0) < 0.5      # beside the cone
    far = np.abs(x - 10.0) > 6.0       # both ends
    assert c.width[near].max() > 2.0
    assert np.allclose(c.width[far], 2.0)
    assert c.width.max() <= 2.0 + 1.5 + 1e-9


def test_corridor_without_hazards_keeps_base_width():
    attrs = {
        "floor_0": _attrs([10, 0, 1.25], (10, 5, 1.25), "floor"),
        "area_0_0": _attrs([0, 0, 1.0], (2, 2, 1.0), "area"),
        "area_0_1": _attrs([20, 0, 1.0], (2, 2, 1.0), "area"),
    }
    g = BodyNavSceneGraph.build(LabeledPoints.from_xyz(np.zeros((1, 3))), attrs)
    c = build_corridor(g, "area_0_0", "area_0_1", [], base_width=2.0)
    assert np.allclose(c.width, 2.0)


def test_task_mask_covers_corridor_and_hazard_volume(graph):
    c = build_corridor(graph, "area_0_0", "area_0_1", [], base_width=2.0,
                       hazard_expand=0.0)
    pts = np.array([
        [10.0, 0.0, 0.5],    # on the centreline -> inside
        [10.0, 0.9, 0.5],    # within half-width -> inside
        [10.0, 3.0, 0.5],    # 3 m off the line, but inside the cone's 1.5 m
                             # clearance? no -- 2 m from the cone -> outside
        [10.0, 1.4, 0.3],    # inside the cone's clearance radius -> inside
        [-8.0, 0.0, 0.5],    # behind the start -> outside
    ])
    points = LabeledPoints.from_xyz(pts)
    m = task_mask(points, c, graph.find(hazardous=True), graph)
    assert m.tolist() == [True, True, False, True, False]


def test_task_mask_ignores_nodes_without_hazard_params(graph):
    c = build_corridor(graph, "area_0_0", "area_0_1", [], base_width=0.2,
                       hazard_expand=0.0)
    points = LabeledPoints.from_xyz(np.array([[10.0, 4.0, 0.5]]))
    # a non-hazard node id must not contribute a volume
    assert not task_mask(points, c, ["spot_0_0_0", "area_0_0"], graph)[0]


def test_priority_merge_keeps_the_mask_and_thins_the_background():
    rng = np.random.default_rng(0)
    inside = rng.uniform(0, 1, (5000, 3))
    outside = rng.uniform(5, 10, (20000, 3))
    xyz = np.vstack([inside, outside])
    points = LabeledPoints.from_xyz(xyz, semantic=np.array(
        ["cone"] * len(inside) + ["floor"] * len(outside), dtype=object))
    mask = np.zeros(len(xyz), dtype=bool)
    mask[:len(inside)] = True

    out = priority_merge(points, mask, critical_spacing=0.02,
                         background_spacing=0.05)

    kept_in = int(out.extras["v_t"].sum())
    # inside: thinned only to the critical spacing, so most points survive
    assert kept_in > 0.5 * len(inside)
    # outside: one representative per background cell, and no more. This is
    # the property the obstacle model depends on -- a surface thinned to a
    # grid still has a measurable height and slope, where a surface thinned
    # to a random fraction has holes the gate reads as terrain.
    kept_bg = out.xyz[~out.extras["v_t"]]
    cells = np.floor((kept_bg - kept_bg.min(axis=0)) / 0.05)
    assert len(np.unique(cells, axis=0)) == len(kept_bg)
    # and it still covers the background's extent, rather than a sample of it
    assert np.allclose(kept_bg.min(axis=0), outside.min(axis=0), atol=0.05)
    assert np.allclose(kept_bg.max(axis=0), outside.max(axis=0), atol=0.05)
    # labels ride along
    assert set(out.semantic[out.extras["v_t"]]) == {"cone"}


def test_priority_merge_is_deterministic_given_a_seed():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 10, (4000, 3))
    points = LabeledPoints.from_xyz(xyz)
    mask = xyz[:, 0] < 2.0
    a = priority_merge(points, mask, seed=7)
    b = priority_merge(points, mask, seed=7)
    assert np.array_equal(a.xyz, b.xyz)


def test_priority_merge_handles_an_empty_mask():
    xyz = np.random.default_rng(2).uniform(0, 1, (1000, 3))
    points = LabeledPoints.from_xyz(xyz)
    out = priority_merge(points, np.zeros(len(xyz), dtype=bool),
                         background_spacing=0.1)
    # every survivor sits in its own 0.1 m cell, and every occupied cell
    # keeps exactly one survivor
    cells = np.floor((xyz - xyz.min(axis=0)) / 0.1)
    assert len(out.xyz) == len(np.unique(cells, axis=0))
    assert not out.extras["v_t"].any()
