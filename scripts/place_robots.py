#!/usr/bin/env python3
"""Author robot Xforms into an existing site stage, in place.

    scripts/kitpy.sh scripts/place_robots.py \
        --stage isaacsim/Real_Site_0903.usd \
        --poses results/labfloor/spawn_poses.json

Runs under Isaac's Python for the USD bindings; the poses come from
`pick_spawn_poses.py` in the BodyNav venv.

The stage is edited in place, after a timestamped backup beside it. That
is safe for the data pipeline for one specific reason:
`usd_export_points.py` selects `UsdGeomPoints` prims only, and a robot is
meshes, so re-exporting a stage that now carries robots yields the same
cloud it did before. Re-running this script replaces any robot it placed
previously rather than stacking duplicates.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

REPO = Path(__file__).resolve().parents[1]

#: The Husky's ".usd" is a *directory* -- the URDF importer writes a
#: package layout -- so the reference has to name the layer inside it.
#: Referencing the directory resolves to nothing, silently, and the robot
#: appears in the outliner with an empty bound.
ASSETS = {
    "husky_ur5": (REPO / "isaacsim" / "husky_ur5" / "husky_ur5.usd"
                  / "husky_ur5" / "husky_ur5.usda"),
    "unitree_h1": (REPO / "isaacsim" / "assets" / "Assets" / "Isaac" / "5.1"
                   / "Isaac" / "Robots" / "Unitree" / "H1" / "h1.usd"),
    "crazyflie21": (REPO / "isaacsim" / "assets" / "Assets" / "Isaac" / "4.5"
                    / "Isaac" / "Robots" / "Crazyflie" / "cf2x.usd"),
}


def relative_to_stage(asset: Path, stage_path: Path) -> str:
    """USD resolves a relative reference against the authoring layer, not
    the working directory, so the stage stays portable."""
    import os

    return os.path.relpath(asset.resolve(), stage_path.resolve().parent)


def place(stage, path: str, ref: str, position, yaw_deg: float):
    """Reference the asset under an Xform that carries the placement.

    The transform goes on a *parent*: a referenced robot brings its own
    ``xformOpOrder``, and adding ops to the referencing prim fights it.
    """
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    holder = UsdGeom.Xform.Define(stage, path)
    holder.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in position)))
    if yaw_deg:
        holder.AddRotateZOp().Set(float(yaw_deg))
    inner = UsdGeom.Xform.Define(stage, f"{path}/model")
    inner.GetPrim().GetReferences().AddReference(ref)
    return holder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--poses", required=True, type=Path)
    ap.add_argument("--root", default="/World",
                    help="parent path for the robot Xforms")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    stage_path = (args.stage if args.stage.is_absolute()
                  else REPO / args.stage)
    poses_path = (args.poses if args.poses.is_absolute()
                  else REPO / args.poses)
    if not stage_path.exists():
        raise SystemExit(f"no such stage: {stage_path}")
    poses = json.loads(poses_path.read_text())

    if not args.no_backup:
        backup = stage_path.with_name(
            f"{stage_path.stem}_{time.strftime('%Y%m%d_%H%M')}"
            f"{stage_path.suffix}")
        shutil.copy2(stage_path, backup)
        print(f"backup  {backup.name}")

    stage = Usd.Stage.Open(str(stage_path))
    if not stage.GetPrimAtPath(args.root):
        UsdGeom.Xform.Define(stage, args.root)

    placed = 0
    for name, spec in (poses.get("robots") or {}).items():
        asset = ASSETS.get(name)
        if asset is None or not asset.exists():
            print(f"  skip {name}: asset missing ({asset})")
            continue
        path = f"{args.root.rstrip('/')}/{name}"
        place(stage, path, relative_to_stage(asset, stage_path),
              spec["position"], float(spec.get("yaw_deg", 0.0)))
        print(f"  placed {name:11s} at {spec['position']} "
              f"yaw {spec.get('yaw_deg', 0.0)} deg -> {path}")
        placed += 1

    if not placed:
        print("nothing placed; stage untouched")
        return 1
    stage.GetRootLayer().Save()
    print(f"saved   {stage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
