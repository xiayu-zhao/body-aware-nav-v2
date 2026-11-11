#!/usr/bin/env python3
"""Re-run every site's batch after a change to how tasks are sampled.

    PYTHONPATH=. .venv/bin/python scripts/rerun_all.py

Keeps the existing file layout -- ``batch.json`` for the two ground
platforms, ``batch_uav.json`` for the quadrotor, per site -- so every
downstream script (merge_batches, make_tables, the figure generators)
reads what it already expects. Previous files are moved aside with a
suffix rather than overwritten, because a result that cannot be compared
against the one it replaced is hard to check.

Cheap sites first: scene2 is a 68 k cloud, scene1 250 k, the lab floor
4.3 M, and a re-sample builds a fresh task map per task because the
corridors move.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITES = ("scene2", "scene1", "labfloor")
JOBS = (("batch.json", ["husky_ur5", "unitree_h1"]),
        ("batch_uav.json", ["crazyflie21"]))


def run(site: str, name: str, robots: list[str], seeds: int, tasks: int,
        suffix: str, dry: bool) -> bool:
    out = REPO / "results" / site / name
    if out.exists() and not dry:
        keep = out.with_name(f"{out.stem}{suffix}{out.suffix}")
        shutil.move(str(out), str(keep))
        print(f"  kept previous as {keep.name}", flush=True)
    cmd = [sys.executable, "scripts/run_batch.py", "--site", site,
           "--seeds", str(seeds), "--tasks", str(tasks), "--out", str(out)]
    for r in robots:
        cmd += ["--robot", r]
    if dry:
        print("  " + " ".join(cmd), flush=True)
        return True
    t0 = time.time()
    print(f"  {site}/{name}: starting", flush=True)
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  {site}/{name}: FAILED rc={p.returncode}\n"
              f"{p.stdout[-1200:]}\n{p.stderr[-1200:]}", flush=True)
        return False
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-11:]
    print(f"  {site}/{name}: done in {time.time() - t0:.0f}s\n    "
          + "\n    ".join(tail), flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--suffix", default="_precentroidfix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ok = True
    for site in args.sites:
        print(f"== {site} ==", flush=True)
        for name, robots in JOBS:
            ok &= run(site, name, robots, args.seeds, args.tasks,
                      args.suffix, args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
