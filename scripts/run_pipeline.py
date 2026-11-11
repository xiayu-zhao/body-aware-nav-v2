"""Run the full BodyNav pipeline on one scan + instruction.

Usage:
    python scripts/run_pipeline.py --cloud scans/site.ply \
        --robot unitree_h1 --instruction "..." [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse

from bodynav import BodyNavPipeline
from bodynav.robots import load_profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True, help="point cloud (.ply)")
    ap.add_argument("--frames", default=None, help="posed RGB-D frame dir")
    ap.add_argument("--robot", default="unitree_h1")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    pipeline = BodyNavPipeline.from_config(args.config, robot=load_profile(args.robot))
    result = pipeline.run(args.cloud, args.instruction, frames=args.frames)
    print(f"success={result.success}  waypoints={len(result.path)}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
