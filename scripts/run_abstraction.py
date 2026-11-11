"""Run stage 2 (robot- and task-aware abstraction) on a saved point cloud.

Stage 2 needs no scene graph and no LLM, so this points straight at a saved
scan and reports the reduction numbers of Sec. III-B.1. The full pipeline
uses scripts/run_stage2.py instead, which also writes the per-robot grids.

Usage:
    python scripts/run_abstraction.py data/labfloor/labfloor.ply \\
        --robot husky_ur5 --max-points 20000000 --save-grid outputs/labfloor.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bodynav.abstraction import AdaptiveVoxelizer, VoxelizerParams, height_slice
from bodynav.io import cloud_info, detect_up_axis, save_grid
from bodynav.robots import load_profile
from bodynav.scene_graph.hierarchy import LabeledPoints
from bodynav.sim import OfflineBridge

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def build_params(config: dict) -> VoxelizerParams:
    v = config["abstraction"]["voxel"]
    return VoxelizerParams(
        r0=v["r0_m"],
        r_min=v["r_min_m"],
        alpha=v["alpha"],
        beta=v["beta"],
        gamma=v["gamma"],
        foothold_buffer=v["foothold_buffer_m"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cloud", help="path to a .ply point cloud")
    ap.add_argument("--robot", default="husky_ur5",
                    help="robot profile name: a file in configs/robots/")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=None,
                    help="cap loaded points by raising the stride")
    ap.add_argument("--up-axis", default="z", choices=["x", "y", "z"],
                    help="vertical axis of the source file (the raw lab-floor scan is Y-up)")
    ap.add_argument("--save-grid", default=None, help="write the VoxelGrid (.npz)")
    ap.add_argument("--json", default=None, help="write stats as JSON")
    args = ap.parse_args(argv)

    import yaml

    with open(CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    info = cloud_info(args.cloud)
    print(f"cloud    : {info['path']}")
    print(f"  format : {info['format']}")
    print(f"  points : {info['n_points']:,}")
    print(f"  fields : {', '.join(info['fields'])}")

    robot = load_profile(args.robot)

    t0 = time.perf_counter()
    bridge = OfflineBridge(
        args.cloud, stride=args.stride, max_points=args.max_points,
        up_axis=args.up_axis,
    )
    xyz, _ = bridge.acquire_scan()
    t_load = time.perf_counter() - t0
    print(f"\nloaded   : {len(xyz):,} points in {t_load:.1f}s "
          f"(source up-axis: {args.up_axis})")
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    print(f"  extent : {np.array2string(hi - lo, precision=1)} m")

    # Slicing on the wrong axis silently produces meaningless geometry, so
    # check the loaded (already rotated) cloud really is Z-up.
    detected = detect_up_axis(xyz[:: max(len(xyz) // 2_000_000, 1)])
    if detected != "z":
        print(f"\n  WARNING: after rotation the cloud still looks '{detected}'-up, "
              f"not z-up.\n           Re-run with --up-axis {detected} "
              "if the slice below looks wrong.")

    points = LabeledPoints.from_xyz(xyz)

    t0 = time.perf_counter()
    sliced = height_slice(
        points, robot,
        overhead_keep_ratio=config["abstraction"]["overhead_keep_ratio"],
    )
    t_slice = time.perf_counter() - t0
    kept = len(sliced.xyz)
    print(f"\nEq. 2 slice ({robot.name}, band {robot.height_band} m):")
    print(f"  kept   : {kept:,} / {len(xyz):,} "
          f"({100 * (1 - kept / max(len(xyz), 1)):.1f}% reduction) in {t_slice:.1f}s")

    voxelizer = AdaptiveVoxelizer(build_params(config), robot)
    # No grounded task yet (stage 3 supplies V_t); phi_T is zero, so this
    # measures the robot-cue-only abstraction.
    task_mask = np.zeros(kept, dtype=bool)

    t0 = time.perf_counter()
    grid = voxelizer.voxelize(sliced, task_mask)
    t_vox = time.perf_counter() - t0
    stats = voxelizer.last_stats

    print(f"\nEq. 4 adaptive voxelization: {t_vox:.1f}s")
    print(f"  voxels : {stats.n_voxels:,}")
    print(f"  vs raw : {100 * (1 - stats.n_voxels / max(len(xyz), 1)):.2f}% reduction")
    print(f"  vs slice: {100 * stats.reduction:.2f}% reduction")
    print(f"  mean r : {stats.mean_resolution * 100:.1f} cm")
    print(f"  fine (<=25cm) share: {100 * stats.fine_share:.1f}%")
    for r, n in sorted(stats.voxels_by_resolution.items()):
        print(f"    r={r * 100:5.1f} cm : {n:>12,}")

    payload = {
        "cloud": info["path"],
        "robot": robot.name,
        "raw_points": int(info["n_points"]),
        "loaded_points": int(len(xyz)),
        "sliced_points": int(kept),
        "voxels": int(stats.n_voxels),
        "mean_resolution_m": stats.mean_resolution,
        "fine_share": stats.fine_share,
        "voxels_by_resolution": {str(k): v for k, v in stats.voxels_by_resolution.items()},
        "timings_s": {"load": t_load, "slice": t_slice, "voxelize": t_vox},
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nstats -> {args.json}")
    if args.save_grid:
        save_grid(grid, args.save_grid)
        print(f"grid  -> {args.save_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
