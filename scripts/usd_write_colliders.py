#!/usr/bin/env python3
"""Write BodyNav's collision boxes into a USD layer PhysX can collide with.

    scripts/kitpy.sh scripts/usd_write_colliders.py \
        --boxes data/labfloor/colliders_husky_ur5.npz \
        --out isaacsim/colliders_husky_ur5.usd

One static mesh carrying every box, with UsdPhysics collision applied and
the approximation set to `none` -- the exact triangles, not a convex hull.
A convex decomposition of a building's obstacle set would close the
corridors it is supposed to leave open.

The mesh is invisible by default: the point cloud is what a viewer should
see, and the colliders are what physics should feel. Pass --visible to
render them for a figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prim", default="/World/beam_colliders")
    ap.add_argument("--visible", action="store_true")
    args = ap.parse_args()

    data = np.load(args.boxes, allow_pickle=True)
    robot = str(data["robot"])

    def box_mesh(lower, upper):
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        size = upper - lower
        pts = (lower[:, None, :] + corners[None, :, :] * size[:, None, :]).reshape(-1, 3)
        quads = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                          [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)
        f = (quads[None, :, :] + (np.arange(len(lower)) * 8)[:, None, None]).reshape(-1, 4)
        return pts, f

    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, args.prim)
    stage.SetDefaultPrim(world.GetPrim())

    layers = [("obstacles", data["lower"], data["upper"], (0.85, 0.35, 0.2))]
    if "floor_lower" in data.files:
        layers.append(("floor", data["floor_lower"], data["floor_upper"],
                       (0.6, 0.6, 0.62)))

    total_faces = 0
    for name, lower, upper, colour in layers:
        points, faces = box_mesh(lower, upper)
        mesh = UsdGeom.Mesh.Define(stage, f"{args.prim}/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)),
                               Gf.Vec3f(*points.max(axis=0))])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        if not args.visible:
            UsdGeom.Imageable(mesh).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        # exact triangles: a convex hull of a building closes its corridors
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.none)
        prim.CreateAttribute("bodynav:boxes", Sdf.ValueTypeNames.Int).Set(int(len(lower)))
        total_faces += len(faces)
        print(f"  {name:10s} {len(lower):>7,} boxes  {len(faces):>8,} faces")

    root = stage.GetPrimAtPath(args.prim)
    if root:
        root.CreateAttribute("bodynav:source", Sdf.ValueTypeNames.String).Set(str(args.boxes))
        root.CreateAttribute("bodynav:robot", Sdf.ValueTypeNames.String).Set(robot)

    stage.GetRootLayer().Save()
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB): {total_faces:,} faces total, "
          f"collision=none (exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
