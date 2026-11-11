#!/usr/bin/env python3
"""Collision boxes for the simulator, merged from BodyNav's obstacle lattice.

    PYTHONPATH=. .venv/bin/python scripts/build_colliders.py \
        --grid data/labfloor/grid_husky_ur5.npz --robot husky_ur5

Writes the boxes as an .npz for `scripts/usd_write_colliders.py`, which
runs under Isaac's Python. The split exists because the obstacle model
needs scipy and Isaac's interpreter has only numpy -- so the geometry is
computed here and only written there.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bodynav.io import load_grid_arrays
from bodynav.planning.obstacles import build_collision_primitives, compute_clearance_fields
from bodynav.robots import load_profile
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.sim.colliders import merge_boxes

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--graph", type=Path,
                    default=Path("results/labfloor/scene_graph_nodes.json"))
    ap.add_argument("--robot", default="husky_ur5")
    ap.add_argument("--free-space", type=Path,
                    default=Path("data/labfloor/free_space.npz"))
    ap.add_argument("--pitch", type=float, default=0.1)
    ap.add_argument("--floor-thickness", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path, default=REPO / "configs" / "robots")
    args = ap.parse_args()

    out = args.out or Path(f"data/labfloor/colliders_{args.robot}.npz")
    stats_path = args.json or Path(f"results/labfloor/colliders_{args.robot}.json")

    robot = load_profile(args.robot, config_dir=args.config_dir)
    grid = load_grid_arrays(args.grid)
    graph = BodyNavSceneGraph.load(str(args.graph))
    prior = None
    if args.free_space.exists():
        with np.load(args.free_space) as fp:
            prior = {k: fp[k] for k in fp.files}

    t0 = time.time()
    fields = compute_clearance_fields(
        grid, build_collision_primitives(graph, grid, robot), robot,
        pitch=args.pitch, floor_z=float(prior["floor_z"]) if prior else None,
        free_prior=prior)
    print(f"lattice {fields.occupied.shape} at {args.pitch} m | "
          f"{int(fields.occupied.sum()):,} obstacle cells")

    # The colliders must be the obstacle set the *planner gated on*, not
    # every occupied voxel. Otherwise a path that legitimately passes a
    # cell the gate calls free skims 5 cm from a collider the simulator
    # will enforce -- and the executed run then measures the disagreement
    # between two obstacle definitions rather than the plan.
    occupancy = fields.occupied.copy()
    k_lo, k_hi = fields.band_k
    occupancy[:, :, k_lo:k_hi + 1] = fields.obstacle_2d[:, :, None]
    print(f"gate    {int(fields.obstacle_2d.sum()):,} blocked columns in the "
          f"{k_lo}-{k_hi} band; colliders follow the gate inside it and the "
          f"voxels above it")

    boxes = merge_boxes(occupancy, fields.origin, args.pitch)
    print(f"merged  {boxes.meta['cells']:,} cells -> {len(boxes):,} boxes "
          f"({boxes.meta['reduction'] * 100:.1f}% fewer), exact: "
          f"{boxes.meta['complete']} ({time.time() - t0:.1f}s)")

    # BodyNav treats the floor as support rather than obstacle -- a robot
    # standing on it is not in collision with it -- so it is absent from
    # the obstacle lattice, and a robot dropped into that stage falls
    # through the building. Physics needs it back, as its own collider,
    # built from the scanned floor rather than an infinite plane so the
    # robot cannot drive off the edge of what was mapped.
    # Merge the floor's *footprint* in plan, then give the slab its true
    # thickness. Taking the occupied lattice cell instead puts the walking
    # surface up to a whole cell above the real floor -- a probe dropped
    # on it rested at 0.34 m when the floor is at -0.045 m, and every
    # executed run would inherit that offset.
    footprint = (fields.ground_k >= 0)[:, :, None]
    floor_slab = merge_boxes(footprint, fields.origin, args.pitch)
    floor_lower = floor_slab.lower.copy()
    floor_upper = floor_slab.upper.copy()
    floor_z = float(fields.meta["floor_z"])
    floor_lower[:, 2] = floor_z - args.floor_thickness
    floor_upper[:, 2] = floor_z
    print(f"floor   {int(footprint.sum()):,} columns -> {len(floor_slab):,} boxes, "
          f"top at z = {floor_z:.3f} m")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lower=boxes.lower, upper=boxes.upper,
                        floor_lower=floor_lower, floor_upper=floor_upper,
                        pitch=np.float64(args.pitch),
                        floor_z=np.float64(fields.meta["floor_z"]),
                        robot=np.array(robot.name, dtype=object))
    stats = {"robot": robot.name, **boxes.meta,
             "floor_columns": int(footprint.sum()),
             "floor_boxes": len(floor_slab),
             "floor_top_z": round(floor_z, 4),
             "volume_m3": round(boxes.volume, 2),
             "lattice_dims": list(fields.occupied.shape),
             "runtime_s": round(time.time() - t0, 1)}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1) + "\n")
    print(f"wrote {out} and {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
