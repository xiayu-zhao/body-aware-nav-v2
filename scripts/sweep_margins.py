#!/usr/bin/env python3
"""Sweep how the stated margin is chosen, and re-measure the comparison.

    PYTHONPATH=. .venv/bin/python scripts/sweep_margins.py

The margins in the main results are not fixed in advance: ``route_choice``
picks, per task, the smallest rung of a fixed ladder that the *direct*
route violates while a compliant route still exists. That is a defensible
choice -- a rung the shortest path already satisfies asks for nothing,
and one no route can meet measures the building -- but it invites the
obvious question, whether the margins were tuned to favour the method.

Two sweeps answer it from opposite sides:

  offset   take the n-th binding rung instead of the first, i.e. demand
           something n rungs *stricter* than the minimum that bites.
           BodyNav's advantage should survive; the number of tasks that
           survive is itself a measurement, and differs sharply by site.

  fixed    override the ladder with one margin for every task. Tasks are
           kept whether or not that value binds, and each records which,
           so the table can show how much of the benchmark a fixed margin
           throws away rather than merely asserting that it does.

Writes ``results/<site>/sweep_margin/<tag>.json``, one file per
configuration, each carrying its own ``margin_offset`` / ``fixed_margin``
so a file can never be mistaken for a default-ladder run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROBOTS = ("husky_ur5", "unitree_h1", "crazyflie21")
# Cheap sites first: scene2 is a 68 k cloud, the lab floor is 4.3 M, and a
# sweep builds a fresh task map per task because the corridor moves.
SITES = ("scene2", "scene1", "labfloor")
OFFSETS = (0, 1, 2)
FIXED = (1.5, 2.5, 3.5)


def configs() -> list[tuple[str, list[str]]]:
    out = []
    for off in OFFSETS:
        out.append((f"offset{off}", [f"--margin-offset={off}"]))
    for m in FIXED:
        out.append((f"fixed{m:g}".replace(".", "p"), [f"--fixed-margin={m}"]))
    return out


def run(site: str, tag: str, extra: list[str], seeds: int, tasks: int,
        dry: bool) -> bool:
    out = REPO / "results" / site / "sweep_margin" / f"{tag}.json"
    if out.exists():
        print(f"  {site}/{tag}: exists, skipping", flush=True)
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/run_batch.py", "--site", site,
           "--seeds", str(seeds), "--tasks", str(tasks),
           "--out", str(out)] + extra
    for r in ROBOTS:
        cmd += ["--robot", r]
    if dry:
        print("  " + " ".join(cmd), flush=True)
        return True
    t0 = time.time()
    print(f"  {site}/{tag}: starting", flush=True)
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  {site}/{tag}: FAILED rc={p.returncode}\n"
              f"{p.stdout[-1500:]}\n{p.stderr[-1500:]}", flush=True)
        return False
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-10:]
    print(f"  {site}/{tag}: done in {time.time() - t0:.0f}s\n    "
          + "\n    ".join(tail), flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these config tags")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [(t, e) for t, e in configs()
            if args.only is None or t in args.only]
    ok = True
    for site in args.sites:
        print(f"== {site} ==", flush=True)
        for tag, extra in todo:
            ok &= run(site, tag, extra, args.seeds, args.tasks, args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
