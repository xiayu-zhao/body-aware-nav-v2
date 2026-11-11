"""Stage 1 on as-built scans: hierarchy, geometric segmentation, graph.

The fixtures build a synthetic single-storey interior whose ground truth
is known by construction -- a slab, a ceiling, two walls, a column, a
traffic cone, a desk, and an overhead duct that drops into the band where
a humanoid is blocked and a wheeled platform is not. Every assertion below
is about a property the manuscript claims, not about a tuned constant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bodynav.robots import load_profile
from bodynav.scene_graph.attributes import Affordances, NodeAttributes
from bodynav.scene_graph.geometric import (
    VOCABULARY,
    body_occupancy,
    clearance_map,
    cluster_features,
    cluster_objects,
    label_cluster,
    low_clearance_regions,
    split_structure,
)
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints, detect_floors, segment_areas

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "robots"

FLOOR_Z, CEIL_Z = 0.0, 2.6
DUCT_Z = 1.6  # blocks a 2.0 m humanoid, clears a 1.25 m wheeled base


@pytest.fixture
def room():
    """12 x 8 m room: slab, ceiling, walls, column, cone, desk, duct."""
    rng = np.random.default_rng(0)

    def slab(n, z, xr=(0, 12), yr=(0, 8), jitter=0.005):
        return np.column_stack([
            rng.uniform(*xr, n), rng.uniform(*yr, n),
            np.full(n, z) + rng.normal(0, jitter, n),
        ])

    floor = slab(40000, FLOOR_Z)
    ceiling = slab(20000, CEIL_Z)
    wall_x = np.column_stack([
        np.full(20000, 0.0) + rng.normal(0, 0.01, 20000),
        rng.uniform(0, 8, 20000), rng.uniform(0, CEIL_Z, 20000)])
    wall_y = np.column_stack([
        rng.uniform(0, 12, 20000), np.full(20000, 0.0) + rng.normal(0, 0.01, 20000),
        rng.uniform(0, CEIL_Z, 20000)])
    column = np.column_stack([
        rng.uniform(5.8, 6.2, 6000), rng.uniform(3.8, 4.2, 6000),
        rng.uniform(0.05, CEIL_Z - 0.05, 6000)])
    cone = np.column_stack([
        rng.uniform(2.0, 2.4, 800), rng.uniform(6.0, 6.4, 800),
        rng.uniform(0.02, 0.5, 800)])
    desk = np.column_stack([
        rng.uniform(9.0, 10.4, 3000), rng.uniform(1.0, 2.0, 3000),
        rng.uniform(0.05, 0.75, 3000)])
    duct = np.column_stack([
        rng.uniform(3.0, 9.0, 6000), rng.uniform(4.9, 5.2, 6000),
        rng.uniform(DUCT_Z, DUCT_Z + 0.25, 6000)])

    xyz = np.vstack([floor, ceiling, wall_x, wall_y, column, cone, desk, duct])
    colors = np.full((len(xyz), 3), 0.5)
    n_before_cone = len(floor) + len(ceiling) + len(wall_x) + len(wall_y) + len(column)
    colors[n_before_cone:n_before_cone + len(cone)] = [0.95, 0.4, 0.1]  # orange
    return xyz, colors


# -- floors -------------------------------------------------------------
def test_detect_floors_finds_the_storey(room):
    xyz, _ = room
    pairs = detect_floors(xyz)
    assert len(pairs) == 1
    floor_z, ceiling_z = pairs[0]
    assert floor_z == pytest.approx(FLOOR_Z, abs=0.05)
    assert ceiling_z == pytest.approx(CEIL_Z, abs=0.05)


def test_detect_floors_ignores_a_desk_height_peak(room):
    """A dense horizontal surface 0.75 m up is furniture, not a storey."""
    xyz, _ = room
    rng = np.random.default_rng(1)
    tabletop = np.column_stack([
        rng.uniform(9.0, 10.4, 30000), rng.uniform(1.0, 2.0, 30000),
        np.full(30000, 0.75) + rng.normal(0, 0.003, 30000)])
    pairs = detect_floors(np.vstack([xyz, tabletop]))
    assert all(ceiling - floor >= 1.5 for floor, ceiling in pairs)
    assert any(f == pytest.approx(FLOOR_Z, abs=0.05) for f, _ in pairs)


def test_detect_floors_empty_cloud():
    assert detect_floors(np.empty((0, 3))) == []


# -- structure ----------------------------------------------------------
def test_split_structure_separates_slabs_walls_and_objects(room):
    xyz, _ = room
    split = split_structure(xyz, (FLOOR_Z, CEIL_Z))
    z = xyz[:, 2]

    assert np.all(np.abs(z[split.floor] - FLOOR_Z) <= 0.09)
    assert np.all(np.abs(z[split.ceiling] - CEIL_Z) <= 0.11)
    # every point lands in exactly one class
    stacked = np.stack([split.floor, split.ceiling, split.wall,
                        split.above_ceiling, split.objects])
    assert np.all(stacked.sum(axis=0) <= 1)

    # walls span the storey, so they are found; the desk is not a wall
    assert split.wall.sum() > 20000
    labels = split.labels()
    assert set(labels[split.floor]) == {"floor"}
    assert set(labels[split.ceiling]) == {"ceiling"}


def test_split_structure_labels_are_in_the_vocabulary(room):
    xyz, _ = room
    labels = split_structure(xyz, (FLOOR_Z, CEIL_Z)).labels()
    assert set(np.unique(labels)) <= set(VOCABULARY) | {"unknown"}


# -- instances and labels ----------------------------------------------
def test_cluster_objects_finds_separate_instances(room):
    xyz, _ = room
    split = split_structure(xyz, (FLOOR_Z, CEIL_Z))
    ids = cluster_objects(xyz[split.objects | split.above_ceiling])
    assert ids.max() + 1 >= 3  # column, cone, desk, duct (some may merge)
    assert (ids >= 0).mean() > 0.8


def test_label_cluster_recognizes_each_category():
    storey = 2.6
    pipe = dict(base_h=2.0, top_h=2.4, height=0.2, width=0.2, depth=0.15,
                elongation=6.0, verticality=1.0, volume=0.2, n=500)
    assert label_cluster(pipe, storey=storey) == "pipe"

    duct = dict(base_h=2.1, top_h=2.5, height=0.4, width=1.0, depth=0.8,
                elongation=1.2, verticality=0.4, volume=0.3, n=900)
    assert label_cluster(duct, storey=storey) == "duct"

    column = dict(base_h=0.05, top_h=2.5, height=2.4, width=0.5, depth=0.45,
                  elongation=1.1, verticality=4.8, volume=0.5, n=6000)
    assert label_cluster(column, storey=storey) == "column"

    cone = dict(base_h=0.02, top_h=0.5, height=0.48, width=0.4, depth=0.4,
                elongation=1.0, verticality=1.2, volume=0.08, n=700,
                orangeness=0.3)
    assert label_cluster(cone, storey=storey) == "cone"

    desk = dict(base_h=0.05, top_h=0.75, height=0.7, width=1.4, depth=0.9,
                elongation=1.5, verticality=0.5, volume=0.9, n=3000,
                orangeness=0.0)
    assert label_cluster(desk, storey=storey) == "furniture"


def test_label_cluster_only_emits_vocabulary_terms():
    rng = np.random.default_rng(5)
    for _ in range(200):
        f = dict(
            base_h=rng.uniform(0, 2.5), top_h=rng.uniform(0, 2.6),
            height=rng.uniform(0.05, 2.5), width=rng.uniform(0.05, 3.0),
            depth=rng.uniform(0.05, 3.0), elongation=rng.uniform(1, 8),
            verticality=rng.uniform(0.1, 6), volume=rng.uniform(0.001, 5),
            n=int(rng.integers(40, 5000)), orangeness=rng.uniform(-0.3, 0.4),
        )
        assert label_cluster(f, storey=2.6) in VOCABULARY


def test_cluster_features_uses_colour_for_orangeness(room):
    xyz, colors = room
    orange = colors[:, 0] > 0.9
    f = cluster_features(xyz[orange], colors[orange], FLOOR_Z)
    assert f["orangeness"] > 0.1
    assert f["base_h"] < 0.2


# -- clearance: the multi-embodiment contrast --------------------------
def test_clearance_map_reports_the_duct_and_open_ceiling(room):
    xyz, _ = room
    clear, meta = clearance_map(xyz, (FLOOR_Z, CEIL_Z), cell=0.25, ignore_below=1.25)
    origin = np.asarray(meta["origin"])

    # most of the room is open to the ceiling
    assert np.median(clear) == pytest.approx(CEIL_Z - FLOOR_Z, abs=0.1)

    def cell_at(x, y):
        i, j = np.floor((np.array([x, y]) - origin) / meta["cell"]).astype(int)
        return clear[i, j]

    # under the duct, clearance is the duct's underside ...
    assert cell_at(6.0, 5.05) == pytest.approx(DUCT_Z, abs=0.1)
    # ... and a few metres away the ceiling is what is overhead
    assert cell_at(6.0, 2.0) == pytest.approx(CEIL_Z - FLOOR_Z, abs=0.1)

    # The global minimum is a *wall*, not the duct: a wall spans the whole
    # storey, so it has returns just above the ignore threshold. That is
    # why a low-clearance region is only meaningful once free_below has
    # removed the cells a robot cannot stand in at all.
    assert clear.min() == pytest.approx(1.25, abs=0.05)


def test_low_clearance_regions_separate_the_embodiments(room):
    xyz, _ = room
    h1 = load_profile("unitree_h1", config_dir=CONFIG_DIR)
    husky = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    clear, meta = clearance_map(xyz, (FLOOR_Z, CEIL_Z), cell=0.25,
                                ignore_below=husky.required_vertical_clearance)
    occ = body_occupancy(xyz, (FLOOR_Z, CEIL_Z), meta,
                         band=(0.15, husky.required_vertical_clearance))
    regions = low_clearance_regions(
        clear, meta,
        tall_needs=h1.required_vertical_clearance,
        short_needs=husky.required_vertical_clearance,
        free_below=occ < 3,
    )
    assert regions, "the duct must produce a contrast region"
    biggest = regions[0]
    assert biggest["min_clearance_m"] == pytest.approx(DUCT_Z, abs=0.15)
    # the duct runs along y ~ 5 m, between x = 3 and 9
    cx, cy = biggest["centroid_xy"]
    assert 3.0 <= cx <= 9.0 and 4.5 <= cy <= 5.5
    # and it is genuinely a contrast: passable for the wheeled platform
    assert biggest["min_clearance_m"] > husky.required_vertical_clearance


def test_no_contrast_when_both_robots_fit(room):
    """With a ceiling at 2.6 m and no duct, nothing separates them."""
    xyz, _ = room
    keep = ~((xyz[:, 2] > DUCT_Z - 0.05) & (xyz[:, 2] < DUCT_Z + 0.3)
             & (xyz[:, 1] > 4.8) & (xyz[:, 1] < 5.3))
    clear, meta = clearance_map(xyz[keep], (FLOOR_Z, CEIL_Z), cell=0.25, ignore_below=1.25)
    occ = body_occupancy(xyz[keep], (FLOOR_Z, CEIL_Z), meta, band=(0.15, 1.25))
    regions = low_clearance_regions(clear, meta, tall_needs=2.0, short_needs=1.25,
                                    free_below=occ < 3, min_area_m2=0.5)
    assert regions == []


# -- areas --------------------------------------------------------------
def test_segment_areas_covers_the_free_space(room):
    xyz, _ = room
    area_id, meta = segment_areas(xyz, (FLOOR_Z, CEIL_Z), cell=0.25)
    assert meta["n_areas"] >= 1
    assert meta["free_area_m2"] > 20.0
    assert len(area_id) == len(xyz)
    assert area_id.max() == meta["n_areas"] - 1


# -- graph --------------------------------------------------------------
def _attrs(lo, hi, semantic="clutter"):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return NodeAttributes(centroid=(lo + hi) / 2, aabb_min=lo, aabb_max=hi,
                          semantic=semantic, affordances=Affordances())


def test_graph_hierarchy_and_adjacency():
    pts = LabeledPoints.from_xyz(np.zeros((3, 3)))
    attrs = {
        "floor_0": _attrs([0, 0, 0], [10, 10, 3], "floor"),
        "area_0_0": _attrs([0, 0, 0], [5, 10, 3], "area"),
        "area_0_1": _attrs([5, 0, 0], [10, 10, 3], "area"),
        "spot_0_0_0": _attrs([1, 1, 0], [2, 2, 1]),
        "spot_0_0_1": _attrs([2.5, 1, 0], [3.5, 2, 1]),   # 0.5 m from spot 0
        "spot_0_0_2": _attrs([1, 8, 0], [2, 9, 1]),       # far from both
        "spot_0_1_0": _attrs([6, 1, 0], [7, 2, 1]),       # different area
    }
    g = BodyNavSceneGraph.build(pts, attrs, r_adj=1.0)

    assert g.parent("spot_0_0_0") == "area_0_0"
    assert g.parent("area_0_1") == "floor_0"
    assert g.parent("floor_0") is None
    assert sorted(g.children("area_0_0")) == ["spot_0_0_0", "spot_0_0_1", "spot_0_0_2"]
    assert sorted(g.children("floor_0")) == ["area_0_0", "area_0_1"]

    # adjacency respects both the radius and the parent scope
    assert g.adjacent("spot_0_0_0") == ["spot_0_0_1"]
    assert "spot_0_1_0" not in g.adjacent("spot_0_0_0")
    assert g.adjacent("spot_0_0_2") == []


def test_graph_find_filters_by_level_semantic_and_floor():
    pts = LabeledPoints.from_xyz(np.zeros((3, 3)))
    cone = _attrs([1, 1, 0], [1.4, 1.4, 0.5], "cone")
    cone.affordances.hazardous = True
    attrs = {
        "floor_0": _attrs([0, 0, 0], [10, 10, 3], "floor"),
        "area_0_0": _attrs([0, 0, 0], [5, 10, 3], "area"),
        "spot_0_0_0": cone,
        "spot_0_0_1": _attrs([3, 3, 0], [4, 4, 1], "furniture"),
    }
    g = BodyNavSceneGraph.build(pts, attrs)
    assert g.find(semantic="cone") == ["spot_0_0_0"]
    assert g.find(hazardous=True) == ["spot_0_0_0"]
    assert g.find(level="area") == ["area_0_0"]
    assert g.find(level="spot", floor="floor_0") == ["spot_0_0_0", "spot_0_0_1"]


def test_graph_roundtrips_through_json(tmp_path):
    from bodynav.scene_graph.attributes import HazardParams

    pts = LabeledPoints.from_xyz(np.zeros((3, 3)))
    cone = _attrs([1, 1, 0], [1.4, 1.4, 0.5], "cone")
    cone.affordances.hazardous = True
    cone.hazard = HazardParams(clearance_radius=0.78, risk_weight=0.8,
                               hazard_class="exclusion_marker")
    attrs = {
        "floor_0": _attrs([0, 0, 0], [10, 10, 3], "floor"),
        "area_0_0": _attrs([0, 0, 0], [5, 10, 3], "area"),
        "spot_0_0_0": cone,
    }
    g = BodyNavSceneGraph.build(pts, attrs)
    g.save(str(tmp_path / "g.json"))
    back = BodyNavSceneGraph.load(str(tmp_path / "g.json"))

    assert set(back.g.nodes) == set(g.g.nodes)
    assert back.g.number_of_edges() == g.g.number_of_edges()
    assert back.parent("spot_0_0_0") == "area_0_0"
    haz = back.attributes("spot_0_0_0").hazard
    assert haz.clearance_radius == pytest.approx(0.78)
    assert haz.hazard_class == "exclusion_marker"


def test_graph_rejects_unknown_level_prefix():
    pts = LabeledPoints.from_xyz(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        BodyNavSceneGraph.build(pts, {"widget_0": _attrs([0, 0, 0], [1, 1, 1])})
