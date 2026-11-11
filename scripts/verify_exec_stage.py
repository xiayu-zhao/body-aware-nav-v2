#!/usr/bin/env python3
"""Structural checks on the execution stage, before physics is involved.

    scripts/kitpy.sh scripts/verify_exec_stage.py

Physics failures are expensive to debug and most of them are not physics:
an unresolved reference, a robot spawned inside a wall, a floor that does
not extend under the start pose. These are answerable from the composed
stage in a second, so they are answered first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

REPO = Path(__file__).resolve().parents[1]


def boxes_from(path: Path):
    with np.load(path, allow_pickle=True) as d:
        return (d["lower"], d["upper"],
                d.get("floor_lower"), d.get("floor_upper"), float(d["floor_z"]))


def inside(boxes_lower, boxes_upper, point, pad=0.0) -> bool:
    return bool(np.any(np.all((point >= boxes_lower - pad) &
                              (point <= boxes_upper + pad), axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Path, default=Path("isaacsim/Real_Site_exec.usd"))
    ap.add_argument("--colliders", type=Path,
                    default=Path("data/labfloor/colliders_husky_ur5.npz"))
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []
    stage = Usd.Stage.Open(str(args.stage))
    prims = list(stage.Traverse())
    counts = {
        "prims": len(prims),
        "meshes": sum(1 for p in prims if p.IsA(UsdGeom.Mesh)),
        "points": sum(1 for p in prims if p.IsA(UsdGeom.Points)),
        "revolute_joints": sum(1 for p in prims if p.IsA(UsdPhysics.RevoluteJoint)),
        "colliders": sum(1 for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)),
        "drives": sum(1 for p in prims if p.HasAPI(UsdPhysics.DriveAPI)),
    }
    print(json.dumps(counts, indent=1))

    # 1. every reference resolved: an unresolved robot is an empty Xform
    for path, want in (("/World/visual/cloud", "point cloud"),
                       ("/World/physics", "colliders"),
                       ("/World/husky_ur5/model", "husky"),
                       ("/World/unitree_h1/model", "h1")):
        prim = stage.GetPrimAtPath(path)
        ok = prim.IsValid() and len(prim.GetChildren()) > 0
        checks.append((f"{want} reference resolves", ok,
                       f"{len(prim.GetChildren()) if prim.IsValid() else 0} children"))

    checks.append(("physics scene present",
                   stage.GetPrimAtPath("/World/physicsScene").IsValid(), ""))
    checks.append(("colliders carry CollisionAPI", counts["colliders"] >= 2,
                   f"{counts['colliders']} prims"))
    # Count drives *under the Husky*: H1 ships pre-rigged with its own,
    # so a stage-wide count says nothing about the joints we authored.
    husky_drives = sum(1 for p in prims
                       if p.HasAPI(UsdPhysics.DriveAPI)
                       and str(p.GetPath()).startswith("/World/husky_ur5"))
    checks.append(("all 10 husky joints driven", husky_drives == 10,
                   f"{husky_drives} under /World/husky_ur5, "
                   f"{counts['drives']} stage-wide"))

    # 2. each robot stands on floor, not inside an obstacle
    lower, upper, floor_lower, floor_upper, floor_z = boxes_from(args.colliders)
    summary = json.loads((REPO / "results/labfloor/exec_stage.json").read_text())
    for robot, info in summary["robots"].items():
        pos = np.asarray(info["position"], dtype=float)
        column = np.array([pos[0], pos[1], floor_z + 0.05])
        on_floor = inside(floor_lower, floor_upper, column, pad=0.15)
        checks.append((f"{robot}: floor beneath the start pose", on_floor,
                       f"at {pos[:2].round(2).tolist()}"))
        body = np.array([pos[0], pos[1], floor_z + 0.5])
        checks.append((f"{robot}: start pose not inside an obstacle",
                       not inside(lower, upper, body), ""))

    # 3. the colliders actually block: a wall cell must be solid
    centre = (lower + upper) / 2
    tall = centre[(upper[:, 2] - lower[:, 2]) > 0.5]
    checks.append(("obstacle boxes include wall-height geometry", len(tall) > 100,
                   f"{len(tall)} boxes taller than 0.5 m"))

    print()
    ok_all = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        ok_all &= ok
    print("\n" + ("all structural checks pass" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
