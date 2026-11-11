#!/usr/bin/env python3
"""Export point-cloud prims from a USD stage into a PLY the pipeline reads.

Run through ``scripts/kitpy.sh`` -- USD bindings ship only with Isaac's
Python, while the rest of BodyNav runs in ``BodyNav/.venv``:

    scripts/kitpy.sh scripts/usd_export_points.py \
        --stage isaacsim/Real_Site_0903.usd \
        --out data/labfloor/labfloor.ply \
        --json results/labfloor/cloud_info.json

**Why this script exists rather than reading the USD arrays directly.**
The lab-floor scan is authored **Y-up in its own local frame** and carried
to the stage's Z-up world by the prim's transform. Reading
``GetPointsAttr()`` and handing the array to BodyNav would slice the building
sideways in Eq. 2 -- silently, with no error, and every downstream number
wrong. So the world transform is always applied, and the result is checked
against what a building floor must look like.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

REPO = Path(__file__).resolve().parents[1]


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_write_ply():
    """Import bodynav.io.ply by path -- the package __init__ pulls in deps
    that Isaac's interpreter does not have."""
    spec = importlib.util.spec_from_file_location(
        "bodynav_io_ply", REPO / "bodynav" / "io" / "ply.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module out of sys.modules, so a
    # module loaded by path must be registered before it is executed.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.write_ply


def points_prims(stage: Usd.Stage, prim_path: str | None):
    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise SystemExit(f"no prim at {prim_path}")
        return [prim]
    found = [p for p in stage.Traverse() if p.IsA(UsdGeom.Points)]
    if not found:
        raise SystemExit("stage contains no UsdGeomPoints prims")
    return found


def read_prim(prim) -> tuple[np.ndarray, np.ndarray | None]:
    """Points in **world** space, with per-point colour when authored."""
    pts = UsdGeom.Points(prim).GetPointsAttr().Get()
    if pts is None or len(pts) == 0:
        return np.empty((0, 3)), None
    xyz = np.asarray(pts, dtype=np.float64)

    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    m = np.asarray(xf, dtype=np.float64)  # USD is row-vector: v' = v @ M
    xyz = xyz @ m[:3, :3] + m[3, :3]

    colors = None
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    if primvar and primvar.HasAuthoredValue():
        raw = primvar.Get()
        if raw is not None and len(raw):
            rgb = np.asarray(raw, dtype=np.float64)
            if len(rgb) == 1:
                rgb = np.repeat(rgb, len(xyz), axis=0)
            if len(rgb) == len(xyz):
                colors = rgb
    return xyz, colors


def storey(z: np.ndarray, *, bin_m: float = 0.05, min_gap_m: float = 1.5) -> dict:
    """Floor and ceiling heights, taken as histogram modes.

    Percentiles do not work here. This scan carries returns that pass
    *through* the suspended ceiling into the services above it -- a long
    sparse tail reaching ~4.2 m in a building whose storey is 2.57 m --
    so a 99.5th percentile reports a storey nearly twice the real one.
    The floor and the ceiling are both dense planes, so they are modes,
    and the tail above the ceiling is reported separately: it is exactly
    the region where ceiling services intrude on H1's 2.0 m envelope.
    """
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    hist, _ = np.histogram(z, bins=edges)
    centers = edges[:-1] + bin_m / 2

    floor_i = int(np.argmax(hist))
    floor_z = float(centers[floor_i])
    above = centers > floor_z + min_gap_m
    ceiling_z = float(centers[above][np.argmax(hist[above])]) if above.any() else float(z.max())

    return {
        "floor_z_m": round(floor_z, 3),
        "ceiling_z_m": round(ceiling_z, 3),
        "storey_m": round(ceiling_z - floor_z, 3),
        "above_ceiling_frac": round(float(np.mean(z > ceiling_z + bin_m)), 4),
    }


def summarize(xyz: np.ndarray) -> dict:
    """Bounds plus the sanity checks that catch an up-axis mistake."""
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = hi - lo
    vertical_is_shortest = int(np.argmin(extent)) == 2
    return {
        "points": int(len(xyz)),
        "points_raw_m": round(len(xyz) / 1e6, 6),
        "bbox_min": [round(float(v), 3) for v in lo],
        "bbox_max": [round(float(v), 3) for v in hi],
        "extent_m": [round(float(v), 3) for v in extent],
        "footprint": f"{extent[0]:.1f} $\\times$ {extent[1]:.1f}",
        "footprint_m": [round(float(extent[0]), 2), round(float(extent[1]), 2)],
        "z_range_m": [round(float(lo[2]), 3), round(float(hi[2]), 3)],
        "vertical_axis_is_shortest": vertical_is_shortest,
        **storey(xyz[:, 2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, type=Path)
    ap.add_argument("--prim", default=None,
                    help="single Points prim; default is every one on the stage")
    ap.add_argument("--out", type=Path, default=None, help="output .ply")
    ap.add_argument("--json", type=Path, default=None, help="cloud_info.json")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--align", dest="align", action="store_true", default=True,
                    help="rotate into the building frame (default)")
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="keep world axes; grids then rasterize walls as staircases")
    args = ap.parse_args()

    stage = Usd.Stage.Open(str(args.stage))
    if stage is None:
        raise SystemExit(f"could not open {args.stage}")

    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"stage up-axis: {up}   metersPerUnit: {mpu}")
    if up != UsdGeom.Tokens.z:
        print(f"  note: stage is {up}-up; world points are rotated to Z-up on export")
    if abs(mpu - 1.0) > 1e-9:
        print(f"  note: scaling by metersPerUnit = {mpu}")

    chunks, colour_chunks, sources = [], [], []
    for prim in points_prims(stage, args.prim):
        xyz, colors = read_prim(prim)
        if not len(xyz):
            continue
        if up == UsdGeom.Tokens.y:  # world Y-up -> Z-up for Eq. 2
            xyz = xyz[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
        if abs(mpu - 1.0) > 1e-9:
            xyz = xyz * mpu
        chunks.append(xyz)
        colour_chunks.append(colors)
        sources.append(str(prim.GetPath()))
        print(f"  {prim.GetPath()}: {len(xyz):,} points")

    if not chunks:
        raise SystemExit("no points found")

    xyz = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    colors = None
    if all(c is not None for c in colour_chunks):
        colors = (np.concatenate(colour_chunks) if len(colour_chunks) > 1
                  else colour_chunks[0])

    step = args.stride
    if args.max_points and len(xyz) > args.max_points:
        step = max(step, -(-len(xyz) // args.max_points))
    if step > 1:
        xyz = xyz[::step]
        colors = colors[::step] if colors is not None else None
        print(f"  subsampled by {step} -> {len(xyz):,} points")

    frame_angle = 0.0
    if args.align:
        frame = _load_module("bodynav/scene_graph/frame.py", "bodynav_frame")
        # the floor is a histogram mode, not a percentile -- the same
        # reasoning as `storey()`, and a 1.6 deg error in the axis comes
        # from getting it wrong
        floor_z = storey(xyz[:, 2])["floor_z_m"]
        floor_pts = xyz[np.abs(xyz[:, 2] - floor_z) < 0.15][:, :2]
        frame_angle = frame.principal_axis(floor_pts if len(floor_pts) > 1000 else xyz[:, :2])
        before = xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max() - xyz[:, 1].min()
        xyz = frame.to_building_frame(xyz, frame_angle)
        after = xyz[:, 0].max() - xyz[:, 0].min(), xyz[:, 1].max() - xyz[:, 1].min()
        print(f"\nbuilding frame      rotated {frame_angle:+.2f} deg about Z")
        print(f"  footprint         {before[0]:.1f} x {before[1]:.1f} m -> "
              f"{after[0]:.1f} x {after[1]:.1f} m")

    info = summarize(xyz)
    info.update(
        frame="building" if args.align else "world",
        frame_rotation_deg=round(float(frame_angle), 4),
        sessions=len(sources),
        source_prims=sources,
        source_stage=str(args.stage),
        stage_up_axis=str(up),
        stride=step,
        has_color=colors is not None,
    )

    print(f"\nworld-space bounds  min {info['bbox_min']}  max {info['bbox_max']}")
    print(f"extent (m)          {info['extent_m']}")
    print(f"footprint (m)       {info['footprint_m']}")
    print(f"floor / ceiling     {info['floor_z_m']} m / {info['ceiling_z_m']} m"
          f"   -> storey {info['storey_m']} m")
    print(f"above the ceiling   {info['above_ceiling_frac'] * 100:.1f}% of returns"
          " (services, and where H1's 2.0 m envelope binds)")
    if not info["vertical_axis_is_shortest"]:
        print("\n  WARNING: Z is not the shortest extent. For a building floor that "
              "\n  usually means the up-axis is wrong -- check before trusting Eq. 2.",
              file=sys.stderr)

    if args.out:
        write_ply = _load_write_ply()
        write_ply(
            args.out, xyz, colors,
            comments=[f"source_stage={args.stage}",
                      f"source_prims={';'.join(sources)}",
                      "frame=world Z-up (local-to-world transform applied)"],
        )
        print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(info, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
