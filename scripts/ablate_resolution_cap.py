#!/usr/bin/env python3
"""Ablate the robot-aware resolution cap on Eq. 4.

    PYTHONPATH=. .venv/bin/python scripts/ablate_resolution_cap.py --site labfloor

Sec. III-B caps r(v) inside the robot's band at a fraction of the lateral
clearance the gate will test. That cap is ours, not the manuscript's, so
it needs a number rather than an assertion. The sweep asks what the cap
buys and what it costs, against the 2.5 cm reference model of the same
scan:

  fidelity  how much of the reference's reachable floor the map recovers,
            and how far its clearance estimate sits from the reference's
  cost      voxels, i.e. memory

An uncapped map is coarse enough in the band that quantisation closes
corridors; too tight a cap spends memory re-quantising points the
priority merge already thinned. The useful cap is the widest one that
has stopped losing floor.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Fractions of `min_lateral_clearance` to try, plus the uncapped case.
FRACTIONS = [None, 0.4, 0.2, 0.1, 0.05]


def main() -> int:
    from scipy import ndimage

    from bodynav.abstraction import AdaptiveVoxelizer, height_slice
    from bodynav.abstraction.robot_filter import workspace_kernel_mask
    from bodynav.abstraction.task_filter import (build_corridor, priority_merge,
                                              task_mask)
    from bodynav.core.types import as_arrays
    from bodynav.eval.task_maps import build_params, load_points
    from bodynav.io import load_grid_arrays
    from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
    from bodynav.planning.obstacles import (build_collision_primitives,
                                         compute_clearance_fields)
    from bodynav.robots import load_profile
    from bodynav.scene_graph.graph import BodyNavSceneGraph
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="labfloor")
    ap.add_argument("--robot", action="append", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    site = args.site
    robots = args.robot or ["husky_ur5", "unitree_h1"]
    out = args.out or REPO / "results" / site / "ablation_resolution_cap.json"

    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    params = build_params(config)
    graph = BodyNavSceneGraph.load(
        str(REPO / "results" / site / "scene_graph_nodes.json"))
    with np.load(REPO / "data" / site / "free_space.npz") as fp:
        prior = {k: fp[k] for k in fp.files}
    floor_z = float(prior["floor_z"])
    points = load_points(REPO / "data" / site / f"{site}.ply",
                         REPO / "data" / site / "point_labels.npz")
    hazards = graph.find(hazardous=True)

    def fields_of(grid, robot):
        return compute_clearance_fields(
            grid, build_collision_primitives(graph, grid, robot), robot,
            pitch=0.1, floor_z=floor_z, free_prior=prior)

    def reach_and_clearance(fields, planner):
        _, passable = planner._ground_layer(fields)
        lab, _ = ndimage.label(passable)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        area = float(sizes.max()) * fields.pitch ** 2 if sizes.any() else 0.0
        k = np.clip(fields.ground_k + 1, 0, fields.occupied.shape[2] - 1)
        ii, jj = np.nonzero(fields.ground_k >= 0)
        lat = fields.lateral_grid[ii, jj, k[ii, jj]]
        return area, passable, (ii, jj), lat

    results: dict = {"site": site, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "robots": {}}
    for name in robots:
        robot = load_profile(name, config_dir=REPO / "configs" / "robots")
        planner = ValueGuidedAStar(PlannerParams(), robot)

        ref = load_grid_arrays(REPO / "data" / site / f"grid_{name}_reference.npz")
        ref_fields = fields_of(ref, robot)
        ref_area, ref_pass, ref_idx, ref_lat = reach_and_clearance(
            ref_fields, planner)
        lookup = {(int(i), int(j)): v for i, j, v in zip(*ref_idx, ref_lat)}

        sliced = height_slice(points, robot)
        start_id, goal_id = _anchors(graph)
        corridor = build_corridor(graph, start_id, goal_id, [])
        kernel = workspace_kernel_mask(sliced.xyz, corridor.centerline[:, :2],
                                       robot)
        v_t = task_mask(sliced, corridor, hazards, graph) & kernel
        merged = priority_merge(sliced, v_t)

        rows = []
        for frac in FRACTIONS:
            t0 = time.time()
            up = dataclasses.replace(params, band_clearance_fraction=frac)
            grid = as_arrays(AdaptiveVoxelizer(up, robot).voxelize(
                merged, merged.extras["v_t"]))
            f = fields_of(grid, robot)
            area, _, idx, lat = reach_and_clearance(f, planner)
            # clearance error against the reference, on columns both admit
            err = [abs(v - lookup[(int(i), int(j))])
                   for i, j, v in zip(*idx, lat) if (int(i), int(j)) in lookup]
            cap_m = (None if frac is None
                     else round(max(frac * robot.min_lateral_clearance,
                                    params.r_min), 4))
            rows.append({
                "fraction": frac,
                "cap_m": cap_m,
                "voxels": int(len(grid.index)),
                "memory_mb": round(len(grid.index) * 96 / 1e6, 1),
                "reachable_m2": round(area, 1),
                "reachable_frac_of_reference": (round(area / ref_area, 3)
                                                if ref_area else None),
                "clearance_mae_m": (round(float(np.mean(err)), 4) if err else None),
                "build_s": round(time.time() - t0, 1),
            })
            print(f"  {name:11s} cap={str(cap_m):>6} "
                  f"voxels {rows[-1]['voxels']:>9,} "
                  f"reach {area:7.1f} m2 ({rows[-1]['reachable_frac_of_reference']}) "
                  f"clearance MAE {rows[-1]['clearance_mae_m']}")
        results["robots"][name] = {
            "reference_reachable_m2": round(ref_area, 1),
            "reference_voxels": int(len(ref.index)),
            "sweep": rows,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def _anchors(graph):
    """The most separated pair of areas, as stage 2 picks them."""
    import numpy as np

    areas = list(graph.nodes_by_level("area"))
    xy = np.array([np.asarray(graph.attributes(a).centroid)[:2] for a in areas])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return areas[i], areas[j]


if __name__ == "__main__":
    raise SystemExit(main())
