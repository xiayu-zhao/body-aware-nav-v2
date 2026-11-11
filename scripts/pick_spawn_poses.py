#!/usr/bin/env python3
"""Choose a standing pose per robot on a site, in the stage's world frame.

    PYTHONPATH=. .venv/bin/python scripts/pick_spawn_poses.py --site labfloor

Runs in the BodyNav venv because it needs the planner's own notion of where
a body can stand; `place_robots.py` then authors the result into USD
under Isaac's Python. The split is the same one `kitpy.sh` exists for.

The pose is the passable cell with the *most* lateral clearance in the
platform's largest reachable component -- not a planned start. A planned
start sits wherever a task began, which is often a doorway; for a figure
the robot should stand somewhere it plainly fits, and somewhere a reader
can see it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Height of each robot's root above the floor beneath it. The Husky's
#: origin is 0.145 m above wheel contact; H1's root is its pelvis. An
#: aerial platform does not stand on anything, so it is placed at the
#: middle of its own flight band instead.
SPAWN_HEIGHT_M = {"husky_ur5": 0.145, "unitree_h1": 1.05}


def spawn_height(robot) -> float:
    if getattr(robot, "kind", "") == "aerial":
        lo, hi = robot.height_band
        return 0.5 * (float(lo) + float(hi))
    return SPAWN_HEIGHT_M.get(robot.name, 0.0)


def main() -> int:
    from scipy import ndimage

    from bodynav.io import load_grid_arrays
    from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
    from bodynav.planning.obstacles import (build_collision_primitives,
                                         compute_clearance_fields)
    from bodynav.robots import load_profile
    from bodynav.scene_graph.frame import from_building_frame
    from bodynav.scene_graph.graph import BodyNavSceneGraph

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="labfloor")
    ap.add_argument("--robot", action="append",
                    default=None, help="default: husky_ur5 and unitree_h1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    site = args.site
    robots = args.robot or ["husky_ur5", "unitree_h1"]
    out = args.out or REPO / "results" / site / "spawn_poses.json"

    info = json.loads((REPO / "results" / site / "cloud_info.json").read_text())
    angle = float(info.get("frame_rotation_deg", 0.0))
    graph = BodyNavSceneGraph.load(
        str(REPO / "results" / site / "scene_graph_nodes.json"))
    with np.load(REPO / "data" / site / "free_space.npz") as fp:
        prior = {k: fp[k] for k in fp.files}
    floor_z = float(prior["floor_z"])

    poses = {"site": site, "frame_rotation_deg": angle, "robots": {}}
    taken: list[np.ndarray] = []  # building-frame cells already spoken for
    for name in robots:
        robot = load_profile(name, config_dir=REPO / "configs" / "robots")
        grid_path = REPO / "data" / site / f"grid_{name}.npz"
        if not grid_path.exists():
            print(f"  skip {name}: no grid at {grid_path}")
            continue
        grid = load_grid_arrays(grid_path)
        fields = compute_clearance_fields(
            grid, build_collision_primitives(graph, grid, robot), robot,
            pitch=0.1, floor_z=floor_z, free_prior=prior)
        planner = ValueGuidedAStar(PlannerParams(), robot)
        layer_k, passable = planner.occupiable_layer(fields)
        if not passable.any():
            print(f"  skip {name}: nowhere to stand")
            continue
        labels, _ = ndimage.label(passable)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        component = labels == int(np.argmax(sizes))

        lat = np.where(component,
                       fields.lateral_grid[np.arange(component.shape[0])[:, None],
                                           np.arange(component.shape[1])[None, :],
                                           layer_k],
                       -np.inf)
        # keep robots apart: two bodies maximising the same clearance
        # field land on the same cell and occupy each other in the render
        for prev in taken:
            ii, jj = np.nonzero(np.isfinite(lat))
            world_ij = fields.world(np.column_stack(
                [ii, jj, np.zeros_like(ii)]))[:, :2]
            far = np.linalg.norm(world_ij - prev[:2], axis=1) >= 3.0
            blocked = (ii[~far], jj[~far])
            lat[blocked] = -np.inf
        if not np.isfinite(lat).any():
            print(f"  skip {name}: no room clear of the robots already placed")
            continue
        i, j = np.unravel_index(int(np.argmax(lat)), lat.shape)
        centre = fields.world(np.array([[i, j, int(layer_k[i, j])]]))[0]
        taken.append(centre)

        # the planner works in the building frame; the stage does not
        world = from_building_frame(centre.reshape(1, 3), angle)[0]
        # an aircraft is placed at the height its own layer put it at;
        # a ground body at its root offset above the floor
        world[2] = (float(centre[2]) if robot.kind == "aerial"
                    else floor_z + spawn_height(robot))
        poses["robots"][name] = {
            "position": [round(float(v), 3) for v in world],
            "yaw_deg": round(-angle, 1),
            "clearance_m": round(float(lat[i, j]), 3),
            "component_m2": round(float(component.sum()) * fields.pitch ** 2, 1),
        }
        print(f"  {name:11s} at {poses['robots'][name]['position']} "
              f"({poses['robots'][name]['clearance_m']} m clearance in a "
              f"{poses['robots'][name]['component_m2']} m2 region)")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(poses, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
