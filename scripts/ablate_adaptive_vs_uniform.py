#!/usr/bin/env python3
"""Adaptive against uniform voxels, at matched memory, across budgets.

    PYTHONPATH=. .venv/bin/python scripts/ablate_adaptive_vs_uniform.py

The batch already compares BodyNav against a uniform grid bisected to the
same voxel count, and BodyNav wins on safety. That is one budget, and a
reviewer is entitled to ask whether a uniform grid with more memory
simply closes the gap. This answers it by sweeping the budget and
building *both* maps at each point, so the comparison is a curve rather
than a single pair.

At each budget the two maps are measured the same way, against the
2.5 cm reference model of the same scan:

  reachable   how much of the floor the reference calls reachable that
              this map also admits -- a map that quantises a corridor
              shut loses floor no planner can recover
  clearance   how far the map's own clearance estimate sits from the
              reference's, on the columns both admit

Each is reported twice: over the whole floor, and restricted to the task
corridor. The distinction decides how the result reads. An adaptive map
spends its budget where the instruction points and is deliberately coarse
elsewhere, so scoring it over the whole building asks it for something it
is not trying to do -- and on that metric a uniform grid at the same
budget does as well or better. The corridor is where the claim lives.

The adaptive budgets are produced by sweeping the band resolution cap;
each resulting voxel count is then handed to the uniform builder as its
target, so the two schemes are compared at equal memory by construction
rather than by choosing a resolution that flatters either.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

#: Cap fractions to sweep; each yields one adaptive budget, and each of
#: those budgets is then matched by a uniform grid.
FRACTIONS = [None, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05]


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

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="labfloor")
    ap.add_argument("--robot", action="append", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    site = args.site
    robots = args.robot or ["husky_ur5", "unitree_h1", "crazyflie21"]
    out = args.out or REPO / "results" / site / "ablation_adaptive_vs_uniform.json"

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
        return area, (ii, jj), lat, passable

    def corridor_mask(fields, centreline, radius_m):
        """Lattice columns within `radius_m` of the task corridor."""
        origin, pitch = np.asarray(fields.origin, float), float(fields.pitch)
        dims = fields.occupied.shape[:2]
        seeds = np.ones(dims, dtype=bool)
        ij = np.round((np.asarray(centreline, float)[:, :2] - origin[:2])
                      / pitch).astype(int)
        ok = ((ij[:, 0] >= 0) & (ij[:, 0] < dims[0])
              & (ij[:, 1] >= 0) & (ij[:, 1] < dims[1]))
        if not ok.any():
            return np.zeros(dims, dtype=bool)
        seeds[ij[ok, 0], ij[ok, 1]] = False
        return ndimage.distance_transform_edt(seeds, sampling=pitch) <= radius_m

    results: dict = {"site": site,
                     "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "robots": {}}
    for name in robots:
        robot = load_profile(name, config_dir=REPO / "configs" / "robots")
        planner = ValueGuidedAStar(PlannerParams(), robot)

        ref = load_grid_arrays(REPO / "data" / site / f"grid_{name}_reference.npz")
        ref_fields = fields_of(ref, robot)
        ref_area, ref_idx, ref_lat, ref_pass = reach_and_clearance(
            ref_fields, planner)
        lookup = {(int(i), int(j)): v for i, j, v in zip(*ref_idx, ref_lat)}

        sliced = height_slice(points, robot)
        start_id, goal_id = _anchors(graph)
        corridor = build_corridor(graph, start_id, goal_id, [])
        # The same radius the workspace kernel uses, so "the corridor" here
        # is the region the abstraction was actually told to refine.
        c_radius = float(robot.r_reach + robot.r_safety) if hasattr(
            robot, "r_reach") else 2.0
        # Each map derives its own lattice origin from its own voxel
        # extents, so the lattices differ in shape by a cell or two (the
        # origins agree to ~1.5 cm, well inside one 10 cm cell). The mask
        # is therefore rebuilt on whichever lattice is being measured
        # rather than shared, which is also what makes the shapes line up.
        c_rad = max(c_radius, 2.0)
        ref_corr = corridor_mask(ref_fields, corridor.centerline, c_rad)
        ref_corr_m2 = float((ref_pass & ref_corr).sum()) * ref_fields.pitch ** 2
        kernel = workspace_kernel_mask(sliced.xyz, corridor.centerline[:, :2],
                                       robot)
        v_t = task_mask(sliced, corridor, hazards, graph) & kernel
        merged = priority_merge(sliced, v_t)
        zero = np.zeros(len(sliced.xyz), dtype=bool)

        def measure(grid, tag, budget):
            f = fields_of(grid, robot)
            area, idx, lat, passable = reach_and_clearance(f, planner)
            in_corr = corridor_mask(f, corridor.centerline, c_rad)
            err, err_c = [], []
            for i, j, v in zip(*idx, lat):
                key = (int(i), int(j))
                if key not in lookup:
                    continue
                e = abs(v - lookup[key])
                err.append(e)
                if in_corr[key]:
                    err_c.append(e)
            corr_m2 = float((passable & in_corr).sum()) * f.pitch ** 2
            row = {
                "scheme": tag,
                "voxels": int(len(grid.index)),
                "memory_mb": round(len(grid.index) * 96 / 1e6, 1),
                "reachable_m2": round(area, 1),
                "reachable_frac_of_reference": (round(area / ref_area, 3)
                                                if ref_area else None),
                "clearance_mae_m": (round(float(np.mean(err)), 4)
                                    if err else None),
                "corridor_m2": round(corr_m2, 1),
                "corridor_frac_of_reference": (round(corr_m2 / ref_corr_m2, 3)
                                               if ref_corr_m2 else None),
                "corridor_clearance_mae_m": (round(float(np.mean(err_c)), 4)
                                             if err_c else None),
                "budget_voxels": int(budget) if budget else None,
            }
            print(f"  {name:11s} {tag:8s} {row['memory_mb']:6.1f} MB  "
                  f"floor {row['reachable_frac_of_reference']}  "
                  f"corridor {row['corridor_frac_of_reference']}  "
                  f"MAE {row['clearance_mae_m']} / "
                  f"corridor {row['corridor_clearance_mae_m']}", flush=True)
            return row

        def uniform_at(target: int):
            """Solve for the resolution giving `target` cells, as D2 does."""
            def count_at(r):
                up = dataclasses.replace(params, r0=r, r_min=r,
                                         alpha=0.0, beta=0.0, gamma=0.0)
                return AdaptiveVoxelizer(up, robot).voxelize(sliced, zero)

            lo_r, hi_r = 0.01, 0.60
            grid = count_at(hi_r)
            if len(count_at(lo_r)) <= target:
                return as_arrays(count_at(lo_r))
            for _ in range(12):
                mid = 0.5 * (lo_r + hi_r)
                g = count_at(mid)
                if len(g) > target:
                    lo_r = mid
                else:
                    hi_r = mid
                grid = g
                if abs(len(g) - target) <= 0.02 * target:
                    break
            return as_arrays(grid)

        rows = []
        for frac in FRACTIONS:
            up = dataclasses.replace(params, band_clearance_fraction=frac)
            a_grid = as_arrays(AdaptiveVoxelizer(up, robot).voxelize(
                merged, merged.extras["v_t"]))
            budget = len(a_grid.index)
            rows.append(measure(a_grid, "adaptive", budget))
            rows.append(measure(uniform_at(budget), "uniform", budget))

        results["robots"][name] = {
            "reference_reachable_m2": round(ref_area, 1),
            "reference_corridor_m2": round(ref_corr_m2, 1),
            "corridor_radius_m": round(c_rad, 2),
            "reference_voxels": int(len(ref.index)),
            "sweep": rows,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def _anchors(graph):
    """The most separated pair of areas, as stage 2 picks them."""
    areas = list(graph.nodes_by_level("area"))
    xy = np.array([np.asarray(graph.attributes(a).centroid)[:2] for a in areas])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return areas[i], areas[j]


if __name__ == "__main__":
    raise SystemExit(main())
