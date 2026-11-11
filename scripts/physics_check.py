#!/usr/bin/env python3
"""Drop-and-settle test: does the stage actually behave under physics?

    ~/isaacsim/python.sh scripts/physics_check.py

Runs headless. Two questions, which the structural checks cannot answer:
does a robot placed on the collider floor *stay* there rather than falling
through, and does a body pushed into a wall stop? A stage can be perfectly
composed and still be empty to PhysX -- collision APIs on the wrong prim,
a mesh approximation that hollows the geometry -- and the first sign would
otherwise be a robot driving through a building during an experiment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "isaacsim" / "Real_Site_exec.usd"


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        import omni.usd
        from omni.isaac.core import World
    except ImportError:  # Isaac 5+/6 moved the core package
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        import omni.usd
        from isaacsim.core.api import World

    result = {"stage": str(STAGE)}
    omni.usd.get_context().open_stage(str(STAGE))
    stage = omni.usd.get_context().get_stage()

    summary = json.loads((REPO / "results/labfloor/exec_stage.json").read_text())
    with np.load(REPO / "data/labfloor/colliders_husky_ur5.npz", allow_pickle=True) as d:
        floor_z = float(d["floor_z"])

    # Drop over open floor, not over the robot's own start pose -- the
    # first version landed on the Husky's top plate and reported a rest
    # height of 0.40 m, which looks like a floor and is a vehicle.
    waypoints = np.load(REPO / "data/labfloor/path_husky_ur5.npy")
    pos = np.asarray(waypoints[len(waypoints) // 2], dtype=float)
    from pxr import UsdGeom as G

    cube = G.Cube.Define(stage, "/World/probe_drop")
    cube.CreateSizeAttr(0.2)
    G.Xformable(cube).AddTranslateOp().Set(
        Gf.Vec3d(float(pos[0]), float(pos[1]), floor_z + 1.5))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())

    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(240):            # 4 s at 60 Hz
        world.step(render=False)

    xf = G.Xformable(stage.GetPrimAtPath("/World/probe_drop"))
    final = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    rest_z = float(final[2])
    result["drop"] = {
        "start_z": round(floor_z + 1.5, 3), "rest_z": round(rest_z, 3),
        "floor_z": round(floor_z, 3),
        "landed": bool(rest_z > floor_z - 0.3),
    }
    print(f"drop probe: {floor_z + 1.5:.2f} m -> {rest_z:.2f} m "
          f"(floor {floor_z:.2f}) landed={result['drop']['landed']}")

    # Second half of the gate: are the walls solid to physics? A scene
    # query answers it directly, without waiting for a robot to drive into
    # one during an experiment.
    try:
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
        with np.load(REPO / "data/labfloor/colliders_husky_ur5.npz",
                     allow_pickle=True) as d:
            lower, upper = d["lower"], d["upper"]
        tall = (upper[:, 2] - lower[:, 2]) > 0.8
        centre = ((lower + upper) / 2)[tall]
        # aim at the tallest wall box nearest the middle of the building
        target = centre[np.argmin(np.linalg.norm(
            centre[:, :2] - centre[:, :2].mean(axis=0), axis=1))]
        origin = target + np.array([3.0, 0.0, 0.0])
        hit = query.raycast_closest(
            [float(v) for v in origin], [-1.0, 0.0, 0.0], 6.0)
        result["raycast"] = {
            "from": [round(float(v), 2) for v in origin],
            "hit": bool(hit.get("hit", False)),
            "distance_m": round(float(hit.get("distance", -1.0)), 3),
            "collision": str(hit.get("collision", "")),
        }
        print(f"raycast into a wall: hit={result['raycast']['hit']} "
              f"at {result['raycast']['distance_m']} m "
              f"({result['raycast']['collision']})")
    except Exception as exc:
        result["raycast"] = {"error": str(exc)[:200]}
        print(f"raycast check unavailable: {exc}")

    (REPO / "results/labfloor/physics_check.json").write_text(
        json.dumps(result, indent=1) + "\n")
    app.close()
    ok = result["drop"]["landed"] and result.get("raycast", {}).get("hit", False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
