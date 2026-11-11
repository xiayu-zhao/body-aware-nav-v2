#!/usr/bin/env python3
"""Build the hierarchical scene graph for an as-built scan (stage 1).

    PYTHONPATH=. .venv/bin/python scripts/build_scene_graph.py \
        --cloud data/labfloor/labfloor.ply \
        --graph results/labfloor/scene_graph.json \
        --json  results/labfloor/scene_graph_stats.json

Regime [A] -- a registered LiDAR cloud with colour and no posed frames --
so segmentation is geometric (:mod:`bodynav.scene_graph.geometric`) rather
than the multi-view SAM prompting of Sec. III-A.1, which the data cannot
support. The hierarchy, the graph and every downstream stage are shared.

Also emits the embodiment clearance analysis: the regions a 1.25 m wheeled
platform can pass and a 2.0 m humanoid cannot. That contrast is the
multi-embodiment claim, measured on one scan.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bodynav.io import load_cloud
from bodynav.robots import load_profile
from bodynav.scene_graph.attributes import Affordances, HazardParams, NodeAttributes
from bodynav.scene_graph.geometric import (
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

#: Categories that carry safety semantics on a construction/lab floor.
#: Cones and stored material mark exclusion; overhead services are the
#: hazard that separates the embodiments.
HAZARD_CLASSES: dict[str, tuple[str, float, float]] = {
    # label: (hazard_class, risk_weight, extra clearance margin in m)
    "cone": ("exclusion_marker", 0.8, 0.5),
    "stored_material": ("stored_material", 0.5, 0.3),
    "pipe": ("overhead_service", 0.4, 0.2),
    "duct": ("overhead_service", 0.4, 0.2),
}

NAVIGABLE_LABELS = {"floor"}


def aabb_attrs(pts: np.ndarray, semantic: str) -> NodeAttributes:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    hazard = None
    haz = HAZARD_CLASSES.get(semantic)
    if haz is not None:
        cls, weight, margin = haz
        radius = float(np.linalg.norm((hi - lo)[:2]) / 2 + margin)
        hazard = HazardParams(clearance_radius=radius, risk_weight=weight, hazard_class=cls)
    return NodeAttributes(
        centroid=pts.mean(axis=0),
        aabb_min=lo,
        aabb_max=hi,
        semantic=semantic,
        affordances=Affordances(
            navigable=semantic in NAVIGABLE_LABELS,
            loadable=semantic == "floor",
            hazardous=hazard is not None,
        ),
        hazard=hazard,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True, type=Path)
    ap.add_argument("--graph", type=Path, default=None, help="scene_graph.json")
    ap.add_argument("--json", type=Path, default=None, help="stats for the paper")
    ap.add_argument("--clearance", type=Path, default=None, help="low_clearance.json")
    ap.add_argument("--labels", type=Path, default=None,
                    help="per-point labels (.npz) consumed by stage 2")
    ap.add_argument("--free-space", type=Path, default=None,
                    help="traversable free-space map (.npz) consumed by stage 4")
    ap.add_argument("--config-dir", type=Path, default=Path("configs/robots"))
    ap.add_argument("--tall", default="unitree_h1")
    ap.add_argument("--short", default="husky_ur5")
    ap.add_argument("--area-cell", type=float, default=0.25)
    ap.add_argument("--seed-spacing", type=float, default=3.5,
                    help="watershed seed spacing (m); ~one corridor width")
    ap.add_argument("--r-adj", type=float, default=1.0)
    ap.add_argument("--max-points", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    xyz, colors = load_cloud(args.cloud, max_points=args.max_points)
    print(f"cloud            {len(xyz):,} points  ({time.time() - t0:.1f}s)")

    floors = detect_floors(xyz)
    if not floors:
        raise SystemExit("no floor/ceiling pair found -- check the up-axis")
    floor_z, ceiling_z = floors[0]
    print(f"storey           floor {floor_z:.3f} m, ceiling {ceiling_z:.3f} m "
          f"-> {ceiling_z - floor_z:.3f} m ({len(floors)} storey(s))")

    split = split_structure(xyz, floors[0])
    print("structure        " + "  ".join(
        f"{k} {v * 100:.1f}%" for k, v in split.meta["fractions"].items()))

    area_id, area_meta, free_mask = segment_areas(
        xyz, floors[0], cell=args.area_cell, seed_spacing=args.seed_spacing,
        return_free=True)
    print(f"areas            {area_meta['n_areas']} zones over "
          f"{area_meta['free_area_m2']} m2 of free space")

    # instances: floor-standing objects *and* the services above the ceiling
    inst_mask = split.objects | split.above_ceiling
    inst_local = cluster_objects(xyz[inst_mask])
    instance_id = np.full(len(xyz), -1, dtype=int)
    instance_id[np.flatnonzero(inst_mask)] = inst_local
    n_inst = int(inst_local.max()) + 1 if len(inst_local) else 0
    print(f"instances        {n_inst} clusters from {int(inst_mask.sum()):,} points")

    semantic = split.labels()
    storey = ceiling_z - floor_z
    features: dict[int, dict] = {}
    for cid in range(n_inst):
        sel = np.flatnonzero(instance_id == cid)
        f = cluster_features(xyz[sel], None if colors is None else colors[sel], floor_z)
        label = label_cluster(f, storey=storey)
        f["label"] = label
        features[cid] = f
        semantic[sel] = label

    counts: dict[str, int] = {}
    for f in features.values():
        counts[f["label"]] = counts.get(f["label"], 0) + 1
    print("labels           " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    # -- clearance: the embodiment contrast ----------------------------
    tall = load_profile(args.tall, config_dir=args.config_dir)
    short = load_profile(args.short, config_dir=args.config_dir)
    clear, cmeta = clearance_map(xyz, floors[0], cell=args.area_cell,
                                 ignore_below=short.required_vertical_clearance)
    occ = body_occupancy(xyz, floors[0], cmeta,
                         band=(0.15, short.required_vertical_clearance))
    # "Scanned" means any return in the cell, not a floor return: under a
    # genuine low soffit the floor is frequently occluded, and requiring it
    # discarded 7 of 8 candidate passages. Whether a region is actually
    # traversable is the planner's call in stage 4, against the abstracted
    # grid -- this analysis only nominates candidates.
    scanned = body_occupancy(xyz, floors[0], cmeta, band=(-1e3, 1e3)) > 0
    regions = low_clearance_regions(
        clear, cmeta,
        tall_needs=tall.required_vertical_clearance,
        short_needs=short.required_vertical_clearance,
        free_below=(occ < 3) & scanned,
    )
    contrast_area = round(sum(r["area_m2"] for r in regions), 2)
    print(f"clearance        {len(regions)} regions ({contrast_area} m2) pass for "
          f"{short.name} ({short.required_vertical_clearance} m) and block "
          f"{tall.name} ({tall.required_vertical_clearance} m)")

    # -- graph ----------------------------------------------------------
    points = LabeledPoints(
        xyz=xyz,
        floor_id=np.zeros(len(xyz), dtype=int),
        area_id=area_id,
        instance_id=instance_id,
        semantic=semantic,
        colors=colors,
    )

    attributes: dict[str, NodeAttributes] = {}
    members: dict[str, np.ndarray] = {}

    attributes["floor_0"] = aabb_attrs(xyz, "floor")
    members["floor_0"] = np.arange(len(xyz))

    for a in range(area_meta["n_areas"]):
        sel = np.flatnonzero(area_id == a)
        if not len(sel):
            continue
        node = f"area_0_{a}"
        attributes[node] = aabb_attrs(xyz[sel], "area")
        attributes[node].affordances.navigable = True
        members[node] = sel

    # a spot belongs to the area its footprint sits in (majority vote over
    # its own points' area ids; unassigned points fall back to the nearest)
    for cid in range(n_inst):
        sel = np.flatnonzero(instance_id == cid)
        areas = area_id[sel]
        areas = areas[areas >= 0]
        if len(areas):
            home = int(np.bincount(areas).argmax())
        else:
            centroid = xyz[sel].mean(axis=0)[:2]
            best, home = np.inf, -1
            for a in range(area_meta["n_areas"]):
                node = f"area_0_{a}"
                if node not in attributes:
                    continue
                d = float(np.linalg.norm(attributes[node].centroid[:2] - centroid))
                if d < best:
                    best, home = d, a
            if home < 0:
                continue
        node = f"spot_0_{home}_{cid}"
        attributes[node] = aabb_attrs(xyz[sel], features[cid]["label"])
        members[node] = sel

    area_edges = [(f"area_0_{a}", f"area_0_{b}") for a, b in area_meta["adjacency"]]
    graph = BodyNavSceneGraph.build(points, attributes, r_adj=args.r_adj,
                                 area_edges=area_edges)
    graph._member_index = members
    levels = {lv: len(graph.nodes_by_level(lv)) for lv in ("floor", "area", "spot")}
    kinds: dict[str, int] = {}
    for _, _, d in graph.g.edges(data=True):
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"graph            {levels} | edges {kinds} | {time.time() - t0:.1f}s total")

    if args.free_space:
        # Stage 4 gates on the same free space stage 1 segmented: one
        # occlusion test, one set of opened doors, one answer about where
        # the robot can stand. Deriving traversability a second time from
        # isolated voxel hits produced a corridor broken into 30 m2 pieces.
        args.free_space.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.free_space, free=free_mask,
            origin=np.asarray(area_meta["origin"], dtype=float),
            cell=np.float64(area_meta["cell"]),
            dims=np.asarray(area_meta["dims"], dtype=int),
            floor_z=np.float64(floor_z), ceiling_z=np.float64(ceiling_z),
        )
        print(f"wrote {args.free_space}")

    if args.labels:
        # Stage 2 voxelizes *labelled* points -- the label is what makes a
        # voxel addressable, and what tells the planner which surface is
        # floor to stand on. Codes plus a table, so 4.3M labels cost 4 MB.
        vocab = sorted({str(v) for v in np.unique(semantic)})
        code = {name: i for i, name in enumerate(vocab)}
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.labels,
            semantic_code=np.array([code[str(v)] for v in semantic], dtype=np.int16),
            vocabulary=np.array(vocab, dtype=object),
            area_id=area_id.astype(np.int32),
            instance_id=instance_id.astype(np.int32),
            floor_id=np.zeros(len(xyz), dtype=np.int16),
        )
        print(f"wrote {args.labels}")

    if args.graph:
        graph.save(str(args.graph))
        print(f"wrote {args.graph}")
    if args.clearance:
        args.clearance.parent.mkdir(parents=True, exist_ok=True)
        args.clearance.write_text(json.dumps({
            "tall_robot": tall.name,
            "tall_needs_m": tall.required_vertical_clearance,
            "short_robot": short.name,
            "short_needs_m": short.required_vertical_clearance,
            "n_regions": len(regions),
            "contrast_area_m2": contrast_area,
            "regions": regions,
            "grid": cmeta,
        }, indent=1) + "\n")
        print(f"wrote {args.clearance}")
    if args.json:
        stats = {
            "points": int(len(xyz)),
            "storey_m": round(float(storey), 3),
            "floor_z_m": round(float(floor_z), 3),
            "ceiling_z_m": round(float(ceiling_z), 3),
            "n_storeys": len(floors),
            "structure_fractions": split.meta["fractions"],
            "areas": area_meta["n_areas"],
            "free_area_m2": area_meta["free_area_m2"],
            "instances": n_inst,
            "label_counts": counts,
            "nodes": levels,
            "nodes_str": f"{levels['floor']} / {levels['area']} / {levels['spot']}",
            "edges": int(graph.g.number_of_edges()),
            "edge_kinds": kinds,
            "hazard_nodes": len(graph.find(hazardous=True)),
            "clearance_contrast_regions": len(regions),
            "clearance_contrast_area_m2": contrast_area,
            "runtime_s": round(time.time() - t0, 1),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
