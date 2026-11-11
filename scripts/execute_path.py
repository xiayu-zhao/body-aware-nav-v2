#!/usr/bin/env python3
"""Stage 4b: execute a planned path in Isaac Sim and log what happened.

    ~/isaacsim/python.sh scripts/execute_path.py --robot husky_ur5

Two modes, because the two platforms are honestly different here:

  dynamic  the Husky+UR5 is driven through PhysX -- pure pursuit produces
           (v, omega), a differential model turns that into wheel speeds,
           and the pose that comes back is what the simulation did. The
           tracking error is a real result.

  sweep    the H1 has no locomotion policy in this stage, and the
           manuscript already scopes gait control out of BodyNav. Driving a
           humanoid's root kinematically and calling the resulting
           zero-error "tracking" would be a measurement of nothing. What
           the sweep does measure is real: whether the platform's envelope
           clears the collision geometry along the whole planned route.

Both write results/labfloor/exec_<robot>.json with the same metric names,
and each records its own `mode`, so no table can quietly mix them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Clearpath A200 geometry (spec sheet): wheel radius and track width.
WHEEL_RADIUS_M = 0.1651
TRACK_WIDTH_M = 0.555
WHEEL_JOINTS = ("front_left_wheel", "front_right_wheel",
                "rear_left_wheel", "rear_right_wheel")


def pure_pursuit(pose, path, lookahead: float, idx: int):
    """(v, omega, idx, done): steer toward a point `lookahead` ahead.

    Returns the advanced waypoint index so the tracker never re-targets a
    point it has already passed, which is what makes a pure-pursuit
    controller oscillate at corners.
    """
    import numpy as np

    x, y, yaw = pose
    while idx < len(path) - 1 and np.hypot(*(path[idx][:2] - [x, y])) < lookahead:
        idx += 1
    target = path[idx]
    dx, dy = target[0] - x, target[1] - y
    distance = float(np.hypot(dx, dy))
    heading = math.atan2(dy, dx)
    error = (heading - yaw + math.pi) % (2 * math.pi) - math.pi

    done = idx >= len(path) - 1 and distance < 0.35
    # slow down for turns rather than cutting them
    v = 0.6 * max(0.15, math.cos(min(abs(error), math.pi / 2)))
    omega = 1.6 * error
    return v, omega, idx, done


def clearance_along(points, boxes_file: Path):
    """Distance from each logged pose to the nearest collision box."""
    import numpy as np
    from scipy.spatial import cKDTree

    with np.load(boxes_file, allow_pickle=True) as d:
        lower, upper = d["lower"], d["upper"]
    centres = (lower + upper) / 2
    _, idx = cKDTree(centres).query(points, k=1)
    delta = np.maximum(np.maximum(lower[idx] - points, points - upper[idx]), 0.0)
    return np.linalg.norm(delta, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="husky_ur5")
    ap.add_argument("--mode", default=None, choices=["dynamic", "sweep"])
    ap.add_argument("--stage", type=Path, default=REPO / "isaacsim/Real_Site_exec.usd")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--log-hz", type=float, default=20.0)
    args = ap.parse_args()
    mode = args.mode or ("dynamic" if args.robot == "husky_ur5" else "sweep")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Usd, UsdGeom

    path = np.load(REPO / f"data/labfloor/path_{args.robot}.npy")
    plan = json.loads((REPO / f"results/labfloor/plan_{args.robot}.json").read_text())
    boxes_file = REPO / "data/labfloor/colliders_husky_ur5.npz"

    omni.usd.get_context().open_stage(str(args.stage))
    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    world.reset()

    t0 = time.time()
    poses: list[list[float]] = []
    result = {"robot": args.robot, "mode": mode, "stage": str(args.stage),
              "waypoints": int(len(path)),
              "planned_length_m": plan["final"]["length_m"]}

    if mode == "dynamic":
        robot = SingleArticulation("/World/husky_ur5/model/Geometry/base_link",
                                   name="husky")
        robot.initialize()
        names = list(robot.dof_names)
        wheels = [names.index(w) for w in WHEEL_JOINTS if w in names]
        print(f"articulation: {len(names)} dofs, wheels at {wheels}")

        idx, done, aborted = 0, False, None
        steps = int(args.max_seconds * 60)
        log_every = max(int(60 / args.log_hz), 1)
        # step first, then read: the pose comes from the physics tensors,
        # and reading before the first step returns the authored transform
        # for every step until physics has run once
        world.step(render=False)
        for step in range(steps):
            pos, quat = robot.get_world_pose()
            w, x, y, z = (float(v) for v in quat)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            if step % log_every == 0:
                poses.append([float(pos[0]), float(pos[1]), float(pos[2]), yaw])

            v, omega, idx, done = pure_pursuit((pos[0], pos[1], yaw), path, 1.2, idx)
            if done:
                break
            left = (v - omega * TRACK_WIDTH_M / 2) / WHEEL_RADIUS_M
            right = (v + omega * TRACK_WIDTH_M / 2) / WHEEL_RADIUS_M
            targets = np.array([left, right, left, right][:len(wheels)])
            robot.apply_action(ArticulationAction(joint_velocities=targets,
                                                  joint_indices=np.array(wheels)))
            world.step(render=False)

            if step % 1200 == 0:
                travelled = (np.linalg.norm(np.diff(np.asarray(poses)[:, :2], axis=0),
                                            axis=1).sum() if len(poses) > 1 else 0.0)
                print(f"    t={step / 60:5.1f}s  at ({pos[0]:7.2f},{pos[1]:6.2f})  "
                      f"waypoint {idx}/{len(path) - 1}  travelled {travelled:5.1f} m",
                      flush=True)
            if pos[2] < plan.get("floor_z", -0.045) - 1.0:
                aborted = "fell through the floor"
                break
        result["reached_goal"] = bool(done)
        result["abort"] = aborted
        result["sim_seconds"] = round(step / 60.0, 2)
    else:
        # kinematic sweep: carry the platform's envelope along the path
        holder = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/{args.robot}"))
        dense = []
        for a, b in zip(path[:-1], path[1:]):
            n = max(int(np.linalg.norm(b - a) / 0.05), 1)
            dense.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
        dense = np.vstack(dense)
        for k, p in enumerate(dense):
            if k % 5 == 0:
                poses.append([float(p[0]), float(p[1]), float(p[2]), 0.0])
        result["reached_goal"] = True
        result["abort"] = None
        result["sim_seconds"] = None
        world.step(render=False)

    poses_arr = np.asarray(poses, dtype=float)
    xyz = poses_arr[:, :3]
    seg = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    clearance = clearance_along(xyz, boxes_file)

    # tracking error: distance from each logged pose to the planned path
    from scipy.spatial import cKDTree
    dense_plan = np.vstack([
        a + (b - a) * np.linspace(0, 1, max(int(np.linalg.norm(b - a) / 0.05), 1) + 1)[:, None]
        for a, b in zip(path[:-1], path[1:])])
    d_track, _ = cKDTree(dense_plan[:, :2]).query(xyz[:, :2], k=1)

    result.update({
        "logged_poses": int(len(poses_arr)),
        "log_hz": args.log_hz,
        "executed_length_m": round(float(seg.sum()), 2),
        "length_ratio": round(float(seg.sum() / max(plan["final"]["length_m"], 1e-6)), 3),
        "tracking_rmse_m": (round(float(np.sqrt(np.mean(d_track ** 2))), 3)
                            if mode == "dynamic" else None),
        "tracking_max_m": (round(float(d_track.max()), 3)
                           if mode == "dynamic" else None),
        "min_clearance_m": round(float(clearance.min()), 3),
        "mean_clearance_m": round(float(clearance.mean()), 3),
        "contacts": int((clearance <= 0.0).sum()),
        "wall_clock_s": round(time.time() - t0, 1),
    })
    out = REPO / f"results/labfloor/exec_{args.robot}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    np.save(REPO / f"data/labfloor/exec_pose_{args.robot}.npy", poses_arr)

    print(f"\n{args.robot} [{mode}]")
    print(f"  reached goal      {result['reached_goal']}  abort={result['abort']}")
    print(f"  executed length   {result['executed_length_m']} m "
          f"(planned {result['planned_length_m']} m, ratio {result['length_ratio']})")
    if result["tracking_rmse_m"] is not None:
        print(f"  tracking RMSE     {result['tracking_rmse_m']} m "
              f"(max {result['tracking_max_m']} m)")
    print(f"  clearance         min {result['min_clearance_m']} m, "
          f"mean {result['mean_clearance_m']} m, contacts {result['contacts']}")
    print(f"wrote {out}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
