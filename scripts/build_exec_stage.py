#!/usr/bin/env python3
"""Assemble the Isaac Sim stage the robots execute in.

    scripts/kitpy.sh scripts/build_exec_stage.py \
        --out isaacsim/Real_Site_exec.usd

Three layers, each doing one job:

  visual     the as-built point cloud, so what a viewer (and the video)
             sees is the building as scanned;
  physics    BodyNav's own voxels as exact collision meshes, so a contact in
             simulation is a planner failure and not a meshing artefact;
  robots     Husky+UR5 and H1, placed on the floor at their start poses.

Joint drives are authored for the Husky's wheels and arm. The imported
URDF carries no `<dynamics>`, so without gains the arm flops and the
wheels will not hold velocity -- the stage looks right and behaves
uselessly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = Path(__file__).resolve().parents[1]


#: Height of each robot's root above the floor it stands on. The Husky's
#: origin is 0.145 m above the wheel contact; H1's root is its pelvis.
SPAWN_HEIGHT_M = {"husky_ur5": 0.145, "unitree_h1": 1.05}


def relative_to_stage(asset: Path, stage_path: Path) -> str:
    """Asset path relative to the stage layer, so the stage is portable.

    USD resolves a relative reference against the layer that authored it,
    not against the working directory -- a path that happens to work from
    the repo root breaks the moment the stage is opened from anywhere else.
    """
    import os

    return os.path.relpath(asset.resolve(), stage_path.resolve().parent)


def place(stage, path: str, asset: Path, ref: str, position, *,
          orient_deg: float = 0.0):
    """Reference an asset under an Xform that carries the placement.

    The transform goes on a *parent*, not on the referencing prim: a
    referenced robot brings its own ``xformOpOrder``, and adding ops to
    the same prim fights it.
    """
    holder = UsdGeom.Xform.Define(stage, path)
    holder.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in position)))
    if orient_deg:
        holder.AddRotateZOp().Set(float(orient_deg))
    inner = UsdGeom.Xform.Define(stage, f"{path}/model")
    inner.GetPrim().GetReferences().AddReference(ref)
    return holder


#: Link masses in kg for the Husky links the URDF import left without
#: any. The Clearpath A200 is 50 kg all-up; its four wheels already carry
#: 2.637 kg each from the import, so the chassis takes the rest. Without
#: these PhysX reports "negative mass" on the *root* of a driven
#: articulation and approximates the inertia -- the vehicle then lurches
#: instead of driving, which is what it did.
HUSKY_LINK_MASS_KG = {
    "base_link": 33.0,
    "top_chassis_link": 2.0,
    "front_bumper_link": 0.5,
    "rear_bumper_link": 0.5,
    "user_rail_link": 1.0,
    "top_plate_link": 1.5,
}


def author_masses(stage, root: str, masses: dict) -> dict:
    """Give every massless rigid body under ``root`` a plausible mass."""
    fixed = {}
    prim = stage.GetPrimAtPath(root)
    if not prim or not prim.IsValid():
        return fixed
    for body in Usd.PrimRange(prim):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = body.GetName()
        api = UsdPhysics.MassAPI(body) if body.HasAPI(UsdPhysics.MassAPI) \
            else UsdPhysics.MassAPI.Apply(body)
        current = api.GetMassAttr().Get() if api.GetMassAttr() else None
        if current is not None and current > 0:
            continue
        mass = masses.get(name)
        if mass is None:
            continue
        api.CreateMassAttr().Set(float(mass))
        fixed[name] = float(mass)
    return fixed


def drive_joints(stage, root: str, *, wheel_stiffness: float, wheel_damping: float,
                 arm_stiffness: float, arm_damping: float) -> dict:
    """Author drive gains on every revolute joint under ``root``."""
    counts = {"wheel": 0, "arm": 0, "other": 0}
    prim = stage.GetPrimAtPath(root)
    if not prim or not prim.IsValid():
        return counts
    for joint in Usd.PrimRange(prim):
        if not joint.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = joint.GetName().lower()
        if "wheel" in name:
            k, d, kind = wheel_stiffness, wheel_damping, "wheel"
        elif any(t in name for t in ("shoulder", "elbow", "wrist", "arm", "pan", "lift")):
            k, d, kind = arm_stiffness, arm_damping, "arm"
        else:
            k, d, kind = arm_stiffness, arm_damping, "other"
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(k))
        drive.CreateDampingAttr().Set(float(d))
        drive.CreateMaxForceAttr().Set(1000.0)
        counts[kind] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("isaacsim/Real_Site_exec.usd"))
    ap.add_argument("--cloud", type=Path,
                    default=Path("isaacsim/labfloor_loop_light.usd"))
    ap.add_argument("--colliders", type=Path,
                    default=Path("isaacsim/colliders_husky_ur5.usd"))
    # the URDF importer writes a *directory* named husky_ur5.usd; the
    # layer to reference is the .usda inside it
    ap.add_argument("--husky", type=Path, default=Path(
        "isaacsim/husky_ur5/husky_ur5.usd/husky_ur5/husky_ur5.usda"))
    ap.add_argument("--h1", type=Path, default=Path(
        "isaacsim/assets/Assets/Isaac/5.1/Isaac/Robots/Unitree/H1/h1.usd"))
    ap.add_argument("--plans", type=Path, default=Path("results/labfloor"))
    ap.add_argument("--frame-rotation-deg", type=float, default=None,
                    help="building-frame angle; defaults to cloud_info.json")
    args = ap.parse_args()

    info_path = REPO / "results" / "labfloor" / "cloud_info.json"
    angle = args.frame_rotation_deg
    if angle is None and info_path.exists():
        angle = float(json.loads(info_path.read_text()).get("frame_rotation_deg", 0.0))
    angle = angle or 0.0

    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Everything downstream of S1 works in the building frame; the cloud on
    # disk is in the world frame. One rotation on the visual layer puts
    # them in the same place rather than re-exporting the geometry.
    visual = UsdGeom.Xform.Define(stage, "/World/visual")
    visual.AddRotateZOp().Set(float(-angle))
    UsdGeom.Xform.Define(stage, "/World/visual/cloud").GetPrim() \
        .GetReferences().AddReference(relative_to_stage(args.cloud, args.out))

    UsdGeom.Xform.Define(stage, "/World/physics").GetPrim() \
        .GetReferences().AddReference(relative_to_stage(args.colliders, args.out))

    UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr(9.81)

    placed = {}
    for robot, asset, prim_path in (
        ("husky_ur5", args.husky, "/World/husky_ur5"),
        ("unitree_h1", args.h1, "/World/unitree_h1"),
    ):
        plan_file = args.plans / f"plan_{robot}.json"
        if not plan_file.exists() or not asset.exists():
            print(f"  skip {robot}: "
                  f"{'no plan' if not plan_file.exists() else 'asset missing'}")
            continue
        path_file = REPO / "data" / "labfloor" / f"path_{robot}.npy"
        if not path_file.exists():
            print(f"  skip {robot}: no waypoints")
            continue
        waypoints = np.load(path_file)
        start = waypoints[0].copy()
        heading = waypoints[min(3, len(waypoints) - 1)] - start
        yaw = float(np.degrees(np.arctan2(heading[1], heading[0])))
        # stand the robot on the floor, not at the path's z, which is a
        # lattice cell centre a fifth of a metre above it
        collider_file = REPO / "data" / "labfloor" / f"colliders_{robot}.npz"
        floor_z = start[2] - 0.2
        if collider_file.exists():
            with np.load(collider_file, allow_pickle=True) as cd:
                floor_z = float(cd["floor_z"])
        start[2] = floor_z + SPAWN_HEIGHT_M.get(robot, 0.0)
        place(stage, prim_path, asset, relative_to_stage(asset, args.out),
              start, orient_deg=yaw)
        placed[robot] = {"position": [round(float(v), 3) for v in start],
                         "yaw_deg": round(yaw, 1),
                         "waypoints": int(len(waypoints))}
        print(f"  placed {robot} at {placed[robot]['position']} yaw {yaw:.0f} deg")

    masses = author_masses(stage, "/World/husky_ur5", HUSKY_LINK_MASS_KG)
    if masses:
        print(f"  masses  authored {len(masses)} links "
              f"({sum(masses.values()):.1f} kg): {sorted(masses)}")

    gains = drive_joints(stage, "/World/husky_ur5", wheel_stiffness=0.0,
                         wheel_damping=1.0e4, arm_stiffness=1.0e5, arm_damping=1.0e4)
    print(f"  drives  {gains} (wheels velocity-controlled: stiffness 0, damping 1e4)")

    stage.GetRootLayer().Save()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    summary = {"stage": str(args.out), "frame_rotation_deg": angle,
               "robots": placed, "joint_drives": gains,
               "authored_masses_kg": masses}
    (REPO / "results" / "labfloor" / "exec_stage.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
