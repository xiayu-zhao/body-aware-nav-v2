#!/usr/bin/env python3
"""Pool a site's per-platform batches into one summary.

    PYTHONPATH=. .venv/bin/python scripts/merge_batches.py

The ground platforms and the aerial one were sampled separately, because
each draws its tasks inside its own reachable component -- but they run
the same methods through the same planner and the same gate, so the
comparison across methods is valid pooled and is far more informative
that way: on the ground alone PSR and SCR are saturated and separate
nothing, while pooled they range 58--100% and 42--100%.

Reads only ``batch.json`` and ``batch_uav.json``. It must not read
``batch_cm3.json``: those runs are already inside ``batch.json`` (same
run ids, same summary), and pooling both double-counts every 3D-costmap
run and inflates the site's run total.

Writes ``results/<site>/batch_all.json`` in the same shape the batch
driver writes, so ``make_tables.py`` reads it with the same keys.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from bodynav.eval.results import load_results   # maps pre-rename keys

REPO = Path(__file__).resolve().parents[1]
SOURCES = ("batch.json", "batch_uav.json")


def load_batch_module():
    """Reuse the driver's own summariser rather than restating it here."""
    spec = importlib.util.spec_from_file_location(
        "run_batch", REPO / "scripts" / "run_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_site(site_dir: Path, rb) -> dict | None:
    runs, robots, methods, seeds, tasks = [], [], [], None, {}
    for name in SOURCES:
        path = site_dir / name
        if not path.exists():
            continue
        d = load_results(path)
        runs += d["runs"]
        for r in d.get("robots", []):
            if r not in robots:
                robots.append(r)
        for m in d.get("methods", []):
            if m not in methods:
                methods.append(m)
        seeds = d.get("seeds", seeds)
        for k, v in (d.get("tasks") or {}).items():
            tasks[k] = v
    if not runs:
        return None

    ids = [(r["robot"], r["method"], r["task_id"], r["seed"]) for r in runs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{site_dir.name}: duplicate runs in the pool -- "
                         "a source file overlaps another")

    summary = rb.build_summary(runs, robots, methods, {
        "seeds": seeds,
        "tasks": tasks,
        "sources": list(SOURCES),
        "reachable_area": {r: max(x["reachable_m2"] for x in runs
                                  if x["robot"] == r) for r in robots},
    })
    summary["runs"] = runs
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*",
                    default=["scene1", "scene2", "labfloor"])
    args = ap.parse_args()
    rb = load_batch_module()
    for site in args.sites:
        d = REPO / "results" / site
        merged = merge_site(d, rb)
        if merged is None:
            print(f"{site}: no batches")
            continue
        out = d / "batch_all.json"
        out.write_text(json.dumps(merged, indent=1))
        n_tasks = len(merged.get("tasks") or {})
        print(f"{site}: {len(merged['runs'])} runs, {n_tasks} tasks, "
              f"{len(merged['robots'])} platforms -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
