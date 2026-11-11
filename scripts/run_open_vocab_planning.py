"""Planning runs for the open-vocabulary study, in parallel and resumable.

    PYTHONPATH=. .venv/bin/python scripts/run_open_vocab_planning.py \
        --backend claude-sonnet --grounding llm --seeds 3 --jobs 6

One ``run_batch.py`` per (wording level, site, robot), method ``bodynav``,
over the instruction set's 99 tasks. Each writes
``results/open_vocab/plan/<backend>_<grounding>/<level>_<site>_<robot>.json``;
a job whose file exists is skipped, so an interrupted sweep resumes.

A paid run that reaches the budget cap does not fail -- the agent error is
recorded and planning falls back -- so every finished job is checked for
"budget exhausted"; if found, its file is discarded and the sweep stops.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SET = REPO / "benchmarks" / "open_vocab" / "instruction_set.yaml"
SITES = ("scene1", "scene2", "labfloor")
ROBOTS = ("husky_ur5", "unitree_h1", "crazyflie21")
LEVELS = ("structured", "paraphrase", "free_form", "implicit")


def run_job(args, level, site, robot, out: Path) -> str:
    log = out.with_suffix(".log")
    cmd = [sys.executable, "scripts/run_batch.py", "--site", site, "--robot", robot,
           "--method", "bodynav", "--seeds", str(args.seeds),
           "--instruction-set", str(SET), "--level", level,
           "--grounding", args.grounding, "--backend", args.backend,
           "--out", str(out)]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    with log.open("w") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not out.exists():
        return f"FAILED (exit {rc}; see {log.relative_to(REPO)})"
    runs = json.loads(out.read_text())["runs"]
    if any("budget exhausted" in str(r.get("agent_error", "")) for r in runs):
        out.rename(out.with_suffix(".budget_exhausted.json"))
        return "BUDGET EXHAUSTED"
    ok = sum(1 for r in runs if r.get("agent_ok"))
    return f"{len(runs)} runs, agent ok {ok}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--grounding", choices=("lexical", "llm"), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--sites", nargs="*", default=list(SITES),
                    help="lab-floor batches peak at ~9.5 GB (scenes ~1.4 GB), so the "
                         "lab floor can be run as its own, less parallel stream")
    args = ap.parse_args()

    outdir = REPO / "results" / "open_vocab" / "plan" / f"{args.backend}_{args.grounding}"
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(lv, s, r, outdir / f"{lv}_{s}_{r}.json")
            for lv in args.levels for s in args.sites for r in ROBOTS]
    todo = [t for t in todo if not t[3].exists()]
    print(f"{args.backend}/{args.grounding}: {len(todo)} jobs to run", flush=True)
    stop = False
    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(run_job, args, *t): t for t in todo}
        for fut in as_completed(futs):
            lv, s, r, _ = futs[fut]
            status = fut.result()
            print(f"  {lv:11s} {s:8s} {r:12s} {status}", flush=True)
            if status == "BUDGET EXHAUSTED" and not stop:
                stop = True
                for f in futs:
                    f.cancel()
    if stop:
        print("stopped: the LLM budget cap was reached")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
