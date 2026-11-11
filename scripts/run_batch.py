#!/usr/bin/env python3
"""S9: the batch evaluation -- sampled tasks x platforms x methods x seeds.

    PYTHONPATH=. .venv/bin/python scripts/run_batch.py --seeds 3
    PYTHONPATH=. .venv/bin/python scripts/run_batch.py --offline   # no API

**What the tasks are.** Instructions are *sampled per platform*, inside
that platform's reachable component (`bodynav.eval.sampling`). Scoring a
robot on "go to the north bay" when no route to the north bay exists for
a body that size measures the building, not the planner: it was 42 of the
72 failures in the previous batch. The reachable area is not swept under
the rug -- it is reported per platform, because the difference between
what a Husky and an H1 can reach on the same floor *is* the
multi-embodiment result. A quarter of each task set is drawn deliberately
outside the component: those are the instructions the system should
refuse, and refusing them is scored as correct rejection.

**What the methods are.** All six run through the same `plan_once`, the
same A*, and the same post-processing. They differ in the map they plan
on and in what guides the search -- the variables the paper is about:

  bodynav            per-task adaptive map, agent value maps
  bodynav-no-agent   per-task adaptive map, reference value maps (D4: ablates stage 3)
  bodynav-fixed-grid one map for all tasks, agent value maps  (ablates Eq. 4's phi_T)
  pure-planner    per-task adaptive map, no value guidance (lower bound)
  subgoal-astar   per-task adaptive map, unguided A* between subgoals (D3)
  costmap-2d      planar costmap from a sensor slice       (D1: ablates Eq. 2)
  costmap-3d      embodiment-blind 3D occupancy grid        (D1 for a flying body)
  uniform-voxel   uniform voxels at matched memory         (D2: ablates Eq. 4's r(v))

Eq. 4 makes resolution a function of the grounded instruction, so BodyNav's
map is rebuilt per task -- running every instruction against one map
specialised for another is not the method. `bodynav-fixed-grid` is exactly
that mistake, kept as an ablation, so the value of conditioning the
abstraction on the instruction is a measured number rather than a claim.
D1 and D2 stay task-independent because that is what they are: maps built
without knowing the instruction.

Every method is given the *same* start and goal -- the sampled task's, not
one of its own choosing -- so none of them is solving an easier problem.
BodyNav's grounding is measured separately, as grounding accuracy.

**What judges them.** Not any method's own map. Every trajectory is
scored against a 2.5 cm reference model of the same scan -- about five
times BodyNav's memory, which is why nothing plans on it. Grading BodyNav with
BodyNav's abstraction would let a map that coarsens clutter widen a corridor
and score itself for fitting through: on this floor BodyNav's own map calls
87.5 m2 reachable where the reference calls 75.0 m2, so that is not a
hypothetical.

**What the metrics are.** PSR is a path that exists, over the feasible
tasks. SCR is a path every step of which clears the body, the overhead,
and the hazard margins, on the reference model. Agent success and
grounding accuracy are reported as their own numbers rather than folded
into PSR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from bodynav.eval.results import load_results   # maps pre-rename keys

REPO = Path(__file__).resolve().parents[1]

#: ``map`` selects which obstacle model the method plans on; every method
#: is *scored* on "true" regardless.
METHODS = {
    "bodynav": dict(value_maps="bodynav", use_instruction=True, map="task"),
    "bodynav-no-agent": dict(value_maps="bodynav", use_instruction=False, map="task"),
    "bodynav-fixed-grid": dict(value_maps="bodynav", use_instruction=True, map="fixed"),
    "pure-planner": dict(value_maps="none", use_instruction=False, map="task"),
    "subgoal-astar": dict(value_maps="subgoals", use_instruction=False, map="task"),
    "costmap-2d": dict(value_maps="none", use_instruction=False, map="costmap"),
    "costmap-3d": dict(value_maps="none", use_instruction=False, map="costmap3d"),
    "uniform-voxel": dict(value_maps="none", use_instruction=False, map="uniform"),
    # Revision controls (advisor items 17-18). All reuse BodyNav's parse and
    # grounding; "prog" re-executes the program a stored BodyNav run
    # generated (no LLM call), "det" builds Eqs. (5)-(7) deterministically.
    # Only the named factor changes: field source, margin gate, or map.
    "prog-gate": dict(value_maps="bodynav", use_instruction=True, map="task", replay=True, gate=True),
    "prog-nogate": dict(value_maps="bodynav", use_instruction=True, map="task", replay=True, gate=False),
    "det-gate": dict(value_maps="bodynav", use_instruction=True, map="task", deterministic=True, gate=True),
    "det-nogate": dict(value_maps="bodynav", use_instruction=True, map="task", deterministic=True, gate=False),
    "uniform-prog-gate": dict(value_maps="bodynav", use_instruction=True, map="uniform", replay=True, gate=True),
    # Reviewer item (body-surface margins): the stored program and stored
    # grounding, with every gate margin raised by the robot half-width so the
    # stated distance holds from the body surface, not the path centre. The
    # per-task map is built at the raised margin; the fields are unchanged.
    "prog-gate-body": dict(value_maps="bodynav", use_instruction=True, map="task", replay=True,
                           gate=True, body_margin=True, stored_grounding=True),
    # Grounded-costmap baseline: what a practitioner would build --
    # the same LLM grounding (stored, no model call) on a conventional map,
    # with the grounded hazards inflated by their margins as hard obstacles
    # (the margin gate is exactly that inflation) and no value guidance.
    # 2D sensor-slice costmap for the ground robots; generic 3D grid as the
    # aerial counterpart (a planar map cannot describe flight).
    "costmap-gate": dict(value_maps="none", use_instruction=False, map="costmap", gate=True,
                         stored_grounding=True),
    "costmap3d-gate": dict(value_maps="none", use_instruction=False, map="costmap3d", gate=True,
                           stored_grounding=True),
}


def apply_instruction_set(tasks, path: Path, level: str, *, site: str, robot: str):
    """Swap each task's wording for its ``level`` rewrite; keep its truth.

    A rewrite is checked against the task it replaces -- the template text
    it was written from must match the sampled one -- so a re-sampled task
    set can never silently pair new tasks with old wordings.
    """
    import dataclasses

    entries = {(e["site"], e["robot"], e["task_id"]): e
               for e in yaml.safe_load(path.read_text())["tasks"]}
    out = []
    for t in tasks:
        e = entries.get((site, robot, t.id))
        if e is None:
            raise SystemExit(f"{path}: no entry for {site}/{robot}/{t.id}")
        if e["structured"] != t.text:
            raise SystemExit(f"{path}: {site}/{robot}/{t.id} was written for "
                             f"{e['structured']!r}, but the sampled task reads {t.text!r}")
        text = e["structured"] if level == "structured" else e["levels"][level]
        out.append(dataclasses.replace(t, text=text))
    return out


def load_plan_module():
    spec = importlib.util.spec_from_file_location("bodynav_plan_path",
                                                  REPO / "scripts" / "plan_path.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reach_m2(planner, fields) -> float:
    """Area of the largest region this platform can occupy in this map."""
    from scipy import ndimage

    _, passable = planner._ground_layer(fields)
    lab, _ = ndimage.label(passable)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return float(sizes.max()) * fields.pitch ** 2 if sizes.any() else 0.0


def _stat(values, fn=statistics.mean):
    vals = [v for v in values if v is not None]
    return round(float(fn(vals)), 3) if vals else None


def summarise(runs: list[dict]) -> dict:
    """PSR, SCR, correct rejection and the distributions.

    PSR and SCR are computed over the tasks that *are* feasible for the
    platform; the infeasible ones are scored on the opposite question --
    did the system correctly report that no path exists -- and reported as
    their own rate. Pooling the two would let a method that never finds
    anything score well on half the benchmark.
    """
    if not runs:
        return {}
    feas = [r for r in runs if r.get("feasible_by_construction", True)]
    infeas = [r for r in runs if not r.get("feasible_by_construction", True)]
    ok = [r for r in feas if r.get("success")]
    safe = [r for r in ok if r.get("clearance_ok") and r.get("hazard_ok")
            and r.get("vertical_ok", True)]
    rejected = [r for r in infeas if not r.get("success")]

    lengths = [r["final"]["normalized_length"] for r in ok]
    clear = [r["final"]["min_clearance_m"] for r in ok]
    agent_runs = [r for r in runs if "agent_ok" in r]
    grounded = [r for r in runs if r.get("grounded_goal") is not None
                or r.get("goal_match") is not None]

    out = {
        "runs": len(runs),
        "feasible_tasks": len(feas),
        "infeasible_tasks": len(infeas),
        "psr": round(100.0 * len(ok) / len(feas), 1) if feas else None,
        "scr": round(100.0 * len(safe) / len(feas), 1) if feas else None,
        "scr_of_planned": round(100.0 * len(safe) / len(ok), 1) if ok else None,
        "correct_rejection": (round(100.0 * len(rejected) / len(infeas), 1)
                              if infeas else None),
        "normalized_length_mean": _stat(lengths),
        "normalized_length_sd": (round(statistics.stdev(lengths), 3)
                                 if len(lengths) > 1 else 0.0),
        "min_clearance_mean": _stat(clear),
        "min_clearance_worst": round(min(clear), 3) if clear else None,
        "clearance_violations": sum(1 for r in ok if not r.get("clearance_ok")),
        "vertical_violations": sum(1 for r in ok if not r.get("vertical_ok", True)),
        "hazard_violations": sum(1 for r in ok if not r.get("hazard_ok")),
        "worst_vertical_deficit_m": _stat(
            [r.get("vertical_deficit_m") for r in ok], max),
        # Memory is half the claim: a map that wins on PSR while carrying
        # five times the voxels has not won the argument the paper makes.
        "map_voxels_mean": _stat([r.get("map_voxels") for r in runs]),
        "map_mb_mean": (round(_stat([r.get("map_voxels") for r in runs])
                              * 96 / 1e6, 1)
                        if any(r.get("map_voxels") for r in runs) else None),
        "t_plan_mean_s": _stat([r.get("t_plan_s") for r in runs]),
        "expansions_mean": _stat([r.get("expansions") for r in runs]),
        "llm_tokens_mean": _stat([r["agent"]["llm_tokens"] for r in runs
                                  if r.get("agent")]),
        "llm_latency_mean_s": _stat([r["agent"]["llm_latency_s"] for r in runs
                                     if r.get("agent")]),
        "llm_retries_total": sum(r["agent"].get("retries", 0) for r in runs
                                 if r.get("agent")),
        "failures": sorted({r.get("failure_reason") for r in feas
                            if not r.get("success") and r.get("failure_reason")}),
    }
    # Instruction compliance -- the question PSR and SCR do not ask.
    haz = [r for r in feas if r.get("hazard_task", {}).get("min_distance_m")
           is not None]
    if haz:
        out["hazard_distance_mean_m"] = _stat(
            [r["hazard_task"]["min_distance_m"] for r in haz])
        out["hazard_distance_worst_m"] = round(
            min(r["hazard_task"]["min_distance_m"] for r in haz), 3)
        stated = [r for r in haz if r["hazard_task"].get("compliant") is not None]
        if stated:
            out["hazard_margin_compliance"] = round(
                100.0 * sum(1 for r in stated if r["hazard_task"]["compliant"])
                / len(stated), 1)
    if agent_runs:
        out["agent_success"] = round(
            100.0 * sum(1 for r in agent_runs if r["agent_ok"]) / len(agent_runs), 1)
    if grounded:
        out["grounding_accuracy"] = round(
            100.0 * sum(1 for r in grounded if r.get("goal_match")) / len(grounded), 1)
    return out


def priority_response(runs: list[dict]) -> dict:
    """How much the *instruction* moved the path, per method.

    Each pair is one start and one goal asked for twice: once as "take
    the shortest route", once as "stay at least N metres clear of X".
    Both paths are measured against the same hazard. A method with no
    instruction input returns the identical path for both, so its
    response is exactly zero -- not by construction of the metric, but
    because it genuinely did not read the request.
    """
    pairs: dict[tuple, dict] = {}
    for r in runs:
        if not r.get("success") or not r.get("pair_id"):
            continue
        key = (r["robot"], r["method"], r["seed"], r["pair_id"])
        slot = "safety" if r.get("hazard_margin_m") else "efficiency"
        pairs.setdefault(key, {})[slot] = r

    out: dict[str, dict] = {}
    for (robot, method, _seed, _pid), members in pairs.items():
        if len(members) != 2:
            continue
        eff, saf = members["efficiency"], members["safety"]
        d_eff = (eff.get("hazard_task") or {}).get("min_distance_m")
        d_saf = (saf.get("hazard_task") or {}).get("min_distance_m")
        acc = out.setdefault(method, {"pairs": 0, "d_hazard": [], "d_length": [],
                                      "identical": 0})
        acc["pairs"] += 1
        acc["d_length"].append(saf["final"]["length_m"] - eff["final"]["length_m"])
        if d_eff is not None and d_saf is not None:
            acc["d_hazard"].append(d_saf - d_eff)
        if abs(saf["final"]["length_m"] - eff["final"]["length_m"]) < 1e-6:
            acc["identical"] += 1

    for method, acc in out.items():
        n = max(acc["pairs"], 1)
        acc["hazard_gain_mean_m"] = (round(float(np.mean(acc["d_hazard"])), 3)
                                     if acc["d_hazard"] else None)
        acc["length_cost_mean_m"] = (round(float(np.mean(acc["d_length"])), 3)
                                     if acc["d_length"] else None)
        acc["identical_paths_pct"] = round(100.0 * acc["identical"] / n, 1)
        del acc["d_hazard"], acc["d_length"], acc["identical"]
    return out


def by_task(runs: list[dict]) -> dict:
    """Per-task breakdown, so a reviewer can see which tasks are hard
    rather than only that some are."""
    out: dict = {}
    for r in runs:
        key = f"{r.get('robot')}/{r.get('task_id')}"
        e = out.setdefault(key, {
            "instruction": r.get("instruction_text"),
            "feasible": r.get("feasible_by_construction", True),
            "separation_m": r.get("separation_m"), "methods": {}})
        e["methods"].setdefault(r["method"], []).append(bool(r.get("success")))
    for e in out.values():
        e["methods"] = {m: round(100.0 * sum(v) / len(v), 1)
                        for m, v in e["methods"].items()}
    return out


def build_summary(runs, robots, methods, extra: dict) -> dict:
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robots": robots, "methods": methods,
        "overall": summarise(runs),
        "by_method": {m: summarise([r for r in runs if r.get("method") == m])
                      for m in methods},
        "by_robot": {rb: summarise([r for r in runs if r.get("robot") == rb])
                     for rb in robots},
        "by_robot_method": {
            f"{rb}/{m}": summarise([r for r in runs if r.get("robot") == rb
                                    and r.get("method") == m])
            for rb in robots for m in methods},
        "by_priority": {
            p: summarise([r for r in runs if r.get("priority") == p])
            for p in sorted({r.get("priority") for r in runs if r.get("priority")})},
        "by_task": by_task(runs),
        "priority_response": priority_response(runs),
        **extra,
        "runs": runs,
    }


def report(summary: dict, methods: list[str]) -> None:
    print(f"\n{'method':15s} {'PSR':>6} {'SCR':>6} {'rej':>6} {'norm len':>9} "
          f"{'clear':>7} {'haz d':>7} {'comply':>7} {'resp m':>7} {'same':>6}")
    for m in methods:
        s = summary["by_method"].get(m) or {}
        if not s:
            continue
        pr = (summary.get("priority_response") or {}).get(m, {})
        print(f"{m:15s} {_fmt(s.get('psr'))} {_fmt(s.get('scr'))} "
              f"{_fmt(s.get('correct_rejection'))} "
              f"{str(s.get('normalized_length_mean')):>9} "
              f"{str(s.get('min_clearance_mean')):>7} "
              f"{str(s.get('hazard_distance_mean_m')):>7} "
              f"{_fmt(s.get('hazard_margin_compliance'))} "
              f"{str(pr.get('hazard_gain_mean_m')):>7} "
              f"{str(pr.get('identical_paths_pct')):>6}")


def _kv(v) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) else "-"


def _fmt(v) -> str:
    return f"{v:>5.1f}%" if isinstance(v, (int, float)) else f"{'-':>6}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="append", default=None)
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=12,
                    help="tasks sampled per platform (the full set)")
    ap.add_argument("--task-seed", type=int, default=0,
                    help="seed of the task sampler; fixes the benchmark")
    ap.add_argument("--instructions", type=int, default=None,
                    help="how many of the sampled tasks to run, for chunking")
    ap.add_argument("--skip", type=int, default=0,
                    help="offset into the sampled task list, for chunking")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instruction-set", type=Path, default=None,
                    help="YAML of rewritten instructions (benchmarks/open_vocab/); "
                         "each sampled task's text is replaced by its --level variant, "
                         "its ground truth is kept")
    ap.add_argument("--level", default="structured",
                    help="which wording in --instruction-set to run (structured = the "
                         "sampled template text, unchanged)")
    ap.add_argument("--only-runs", default=None,
                    help="comma-separated task_id:seed pairs; plan only these (targeted "
                         "reruns, e.g. scripts/rerun_failed_agent.py)")
    ap.add_argument("--grounding", choices=("lexical", "llm"), default="lexical",
                    help="lexical: synonym-table grounding; llm: the model maps "
                         "each phrase to the scene's categories (open vocabulary)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pitch", type=float, default=0.1)
    ap.add_argument("--site", default="labfloor",
                    help="which site to evaluate: the tag used under "
                         "data/<site>/ and results/<site>/. The pipeline is "
                         "site-agnostic; only the paths differ.")
    ap.add_argument("--margin-offset", type=int, default=0,
                    help="take the n-th binding rung of the margin ladder "
                         "instead of the first, i.e. a demand n rungs "
                         "stricter than the minimum that bites")
    ap.add_argument("--fixed-margin", type=float, default=None,
                    help="override the ladder with one margin for every "
                         "task; tasks are kept whether or not it binds, and "
                         "each run records which")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--config-dir", type=Path, default=REPO / "configs" / "robots")
    ap.add_argument("--replay-from", action="append", type=Path, default=None,
                    help="stored batch JSON(s) whose BodyNav ('bodynav') runs supply the "
                         "generated programs for the prog-* controls")
    ap.add_argument("--merge", nargs="*", type=Path, default=None,
                    help="combine chunk files into one summary and exit")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(f"results/{args.site}/batch.json")

    if args.merge is not None:
        # Chunked runs exist because a single 400-run process gets killed
        # long before it finishes; the chunks are independent, so merging
        # is concatenating their run lists and re-summarising.
        runs: list[dict] = []
        tasks_meta: dict = {}
        reach_report: dict = {}
        for chunk in args.merge:
            blob = load_results(Path(chunk))
            runs.extend(blob["runs"])
            tasks_meta.update(blob.get("tasks", {}))
            reach_report.update(blob.get("reachable_area", {}))
        robots = sorted({r["robot"] for r in runs})
        methods = sorted({r["method"] for r in runs})
        summary = build_summary(runs, robots, methods, {
            "chunks": [str(c) for c in args.merge],
            "tasks": tasks_meta, "reachable_area": reach_report,
            "seeds": len({r["seed"] for r in runs})})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"merged {len(args.merge)} chunks, {len(runs)} runs -> {args.out}")
        report(summary, methods)
        return 0

    robots = args.robot or ["husky_ur5", "unitree_h1"]
    methods = args.method or list(METHODS)

    from bodynav.eval.baselines import costmap_2d_fields, costmap_3d_fields
    from bodynav.eval.sampling import APPROACH_RADIUS_M, sample_tasks
    from bodynav.io import load_grid_arrays
    from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
    from bodynav.planning.obstacles import build_collision_primitives, compute_clearance_fields
    from bodynav.robots import load_profile

    plan_mod = load_plan_module()
    config = yaml.safe_load(args.config.read_text())
    from bodynav.scene_graph.graph import BodyNavSceneGraph
    site = args.site
    graph = BodyNavSceneGraph.load(
        str(REPO / f"results/{site}/scene_graph_nodes.json"))
    with np.load(REPO / f"data/{site}/free_space.npz") as fp:
        prior = {k: fp[k] for k in fp.files}

    from bodynav.eval.task_maps import TaskMapBuilder, build_params, load_points

    builder = None
    if any(METHODS[m]["map"] == "task" for m in methods):
        t_load = time.time()
        pts = load_points(REPO / f"data/{site}/{site}.ply",
                          REPO / f"data/{site}/point_labels.npz")
        builder = TaskMapBuilder(pts, graph, build_params(config),
                                 cache_dir=REPO / f"data/{site}/task_grids")
        print(f"  cloud {len(pts.xyz):,} points loaded in "
              f"{time.time() - t_load:.1f}s for per-task abstraction")

    runs: list[dict] = []
    tasks_meta: dict = {}
    reach_report: dict = {}
    t0 = time.time()

    for robot_name in robots:
        robot = load_profile(robot_name, config_dir=args.config_dir)
        grid = load_grid_arrays(REPO / f"data/{site}/grid_{robot_name}.npz")
        t_fields = time.time()
        fields = compute_clearance_fields(
            grid, build_collision_primitives(graph, grid, robot), robot,
            pitch=args.pitch, floor_z=float(prior["floor_z"]), free_prior=prior)

        ref_path = REPO / f"data/{site}/grid_{robot_name}_reference.npz"
        if not ref_path.exists():
            raise SystemExit(
                f"{ref_path} missing -- build it with:\n"
                f"  scripts/run_stage2.py ... --robot {robot_name} --reference")
        ref_grid = load_grid_arrays(ref_path)
        ref_fields = compute_clearance_fields(
            ref_grid, build_collision_primitives(graph, ref_grid, robot), robot,
            pitch=args.pitch, floor_z=float(prior["floor_z"]), free_prior=prior)

        maps_for = {"fixed": (grid, fields)}
        if any(METHODS[m]["map"] == "costmap" for m in methods):
            maps_for["costmap"] = (grid, costmap_2d_fields(fields))
        if any(METHODS[m]["map"] == "costmap3d" for m in methods):
            maps_for["costmap3d"] = (grid, costmap_3d_fields(fields))
        if any(METHODS[m]["map"] == "uniform" for m in methods):
            u_path = REPO / f"data/{site}/grid_{robot_name}_uniform.npz"
            if not u_path.exists():
                raise SystemExit(
                    f"{u_path} missing -- build it with:\n"
                    f"  scripts/run_stage2.py ... --robot {robot_name} --uniform")
            u_grid = load_grid_arrays(u_path)
            maps_for["uniform"] = (u_grid, compute_clearance_fields(
                u_grid, build_collision_primitives(graph, u_grid, robot), robot,
                pitch=args.pitch, floor_z=float(prior["floor_z"]), free_prior=prior))
        print(f"  {robot_name}: {len(maps_for)} obstacle model(s) + reference "
              f"built in {time.time() - t_fields:.1f}s, reused for every run")

        planner = ValueGuidedAStar(
            PlannerParams(eta=float(config["planning"]["eta"]),
                          goal_snap_m=APPROACH_RADIUS_M), robot)
        # Tasks are drawn on the *reference* component, so "feasible" means
        # a route exists in the building rather than in some method's model
        # of it. A task no method could do is not a fair test; a task only
        # one method's map believes in is not one either.
        all_tasks = sample_tasks(graph, ref_fields, planner, robot,
                                 n=args.tasks, seed=args.task_seed,
                                 margin_offset=args.margin_offset,
                                 fixed_margin=args.fixed_margin)
        if not all_tasks:
            print(f"  {robot_name}: no sampleable tasks -- skipping")
            continue
        n_take = args.instructions if args.instructions is not None else len(all_tasks)
        tasks = all_tasks[args.skip:args.skip + n_take]
        if args.instruction_set is not None:
            tasks = apply_instruction_set(tasks, args.instruction_set, args.level,
                                          site=site, robot=robot_name)
        for t in all_tasks:
            tasks_meta[f"{robot_name}/{t.id}"] = t.as_dict()
        reach_report[robot_name] = {
            "bodynav_map_m2": round(_reach_m2(planner, fields), 1),
            "reference_m2": round(_reach_m2(planner, ref_fields), 1),
            "costmap_m2": (round(_reach_m2(planner, maps_for["costmap"][1]), 1)
                           if "costmap" in maps_for else None),
            "uniform_m2": (round(_reach_m2(planner, maps_for["uniform"][1]), 1)
                           if "uniform" in maps_for else None),
            "reference_voxels": int(len(ref_grid.index)),
            "bodynav_voxels": int(len(grid.index)),
        }
        print(f"  {robot_name}: {len(all_tasks)} sampled tasks "
              f"({sum(t.feasible_by_construction for t in all_tasks)} feasible) "
              f"in {all_tasks[0].meta['reachable_area_m2']} m2 reachable; "
              f"running {len(tasks)}")

        # Eq. 4's map, rebuilt for each instruction and memoised. Several
        # methods share a task's map -- pure-planner and subgoal-astar
        # differ from BodyNav in what guides the search, not in what they can
        # see, and giving them a different map would confound the two.
        task_fields: dict[str, tuple] = {}

        def task_map(robot, task, margin=0.0, _cache=task_fields):
            key = (task.id, round(float(margin), 2))
            if key in _cache:
                return _cache[key]
            g = builder.grid_for(robot, task.start_id, task.goal_id,
                                 verbose=True, hazard_margin_m=margin)
            f = compute_clearance_fields(
                g, build_collision_primitives(graph, g, robot), robot,
                pitch=args.pitch, floor_z=float(prior["floor_z"]),
                free_prior=prior)
            _cache[key] = (g, f)
            return _cache[key]

        def pre_ground(text, _memo={}):
            """Parse and ground once per instruction, before any map."""
            if text in _memo:
                return _memo[text]
            try:
                from bodynav.llm.agent import AgentConfig, ConstrainedAgent
                from bodynav.llm.backends import load_backend

                client = load_backend(args.backend, offline=args.offline)
                agent = ConstrainedAgent(
                    AgentConfig(model=args.backend or "llm-large",
                                grounding=args.grounding), client)
                _memo[text] = agent.pre_ground(text, graph)
            except Exception:
                _memo[text] = None
            return _memo[text]

        stored = {}
        for path in args.replay_from or []:
            for sr in load_results(Path(path))["runs"]:
                if sr.get("method") == "bodynav":
                    stored[(sr["robot"], sr["task_id"], sr["seed"])] = sr
        if any(METHODS[m].get("replay") for m in methods) and not stored:
            raise SystemExit("prog-* controls need --replay-from <stored batch.json>")

        cache: dict = {}
        total = len(tasks) * len(methods) * args.seeds
        done = 0
        only = ({tuple(x.split(":")) for x in args.only_runs.split(",")}
                if args.only_runs else None)
        for task in tasks:
            for seed in range(args.seeds):
                if only is not None and (task.id, str(seed)) not in only:
                    continue
                for method in methods:
                    spec = METHODS[method]
                    done += 1
                    # A method with no LLM in it is deterministic, so
                    # re-running it per seed measures nothing and costs
                    # the same as running it. Seed 0's result is carried
                    # forward, flagged.
                    # (stored-grounding methods read each seed's own grounding)
                    if ((not spec["use_instruction"] or spec.get("deterministic"))
                            and not spec.get("stored_grounding") and seed > 0):
                        prev = next((r for r in reversed(runs)
                                     if r["method"] == method
                                     and r["task_id"] == task.id
                                     and r["robot"] == robot_name), None)
                        if prev is not None:
                            clone = dict(prev)
                            clone.update(seed=seed, repeat_of_seed_0=True)
                            runs.append(clone)
                            continue
                    # Close the bidirectional loop: ground the instruction
                    # *before* the map is built, so the margins it states
                    # decide where the map has to be accurate. Baselines
                    # get margin 0 and therefore the same map they had.
                    pre = None
                    margin = 0.0
                    offset = (robot.min_lateral_clearance / 2.0) if spec.get("body_margin") else 0.0
                    if spec.get("stored_grounding"):
                        # no model call: rebuild the resolved task from the stored run
                        from bodynav.core.types import Priority, TaskSpec
                        from bodynav.llm.grounding import ResolvedTask
                        srun = stored.get((robot_name, task.id, seed)) or {}
                        sg = (srun.get("agent") or {}).get("ground") or {}
                        sp = (srun.get("agent") or {}).get("parse") or {}
                        pre = ResolvedTask(
                            task=TaskSpec(start=None, goal=sp.get("goal") or task.goal_category,
                                          hazards=list(sp.get("hazards") or []),
                                          priority=Priority(task.priority), raw_instruction=task.text),
                            start_id=task.start_id, goal_ids=list(sg.get("goal_ids") or []),
                            hazard_ids=list(sg.get("hazard_ids") or []),
                            hazard_margins={k: float(v) for k, v in (sg.get("hazard_margins") or {}).items()}
                        ) if sg else None
                        if spec["map"] == "task" and pre is not None and pre.hazard_margins:
                            margin = max(pre.hazard_margins.values()) + offset
                    elif spec["use_instruction"]:
                        pre = pre_ground(task.text)
                        if spec["map"] == "task" and pre is not None and pre.hazard_margins:
                            margin = max(pre.hazard_margins.values())
                    replay_code, src = None, None
                    if spec.get("replay"):
                        src = stored.get((robot_name, task.id, seed))
                        # "" reproduces a stored agent failure: the sandbox
                        # rejects it and the run falls back as the original did.
                        replay_code = ((src or {}).get("agent") or {}).get("code") or ""
                    if spec["map"] == "task":
                        plan_grid, plan_fields = task_map(robot, task, margin)
                    else:
                        plan_grid, plan_fields = maps_for[spec["map"]]
                    try:
                        r = plan_mod.plan_once(
                            robot, plan_grid, graph, prior, config,
                            instruction=task.text if spec["use_instruction"] else None,
                            backend=args.backend, offline=args.offline,
                            grounding=args.grounding,
                            pitch=args.pitch, value_maps=spec["value_maps"],
                            priority=task.priority, seed=seed, verbose=False,
                            fields=plan_fields, true_fields=ref_fields, cache=cache,
                            start_id=task.start_id, goal_id=task.goal_id,
                            start_xyz=np.array(task.start_xyz),
                            goal_xyz=np.array(task.goal_xyz),
                            hazard_category=task.hazard_category,
                            hazard_margin_m=task.hazard_margin_m,
                            hazard_node_ids=task.hazard_node_ids,
                            agent_pre=pre, replay_code=replay_code,
                            deterministic_fields=bool(spec.get("deterministic")),
                            hard_gate=bool(spec.get("gate", True)),
                            gate_margin_offset=offset)
                    except Exception as exc:
                        r = {"robot": robot_name, "success": False,
                             "failure_reason": f"{type(exc).__name__}: {exc}"[:200]}
                    r.update({
                        "method": method, "map_model": spec["map"],
                        "map_voxels": int(len(plan_grid.index)),
                        "task_id": task.id, "instruction_text": task.text,
                        "instruction_level": args.level, "grounding": args.grounding,
                        "goal_category": task.goal_category,
                        "priority": task.priority,
                        "separation_m": round(task.separation_m, 2),
                        "hazard_category": task.hazard_category,
                        "hazard_margin_m": task.hazard_margin_m,
                        "pair_id": task.pair_id,
                        "feasible_by_construction": task.feasible_by_construction,
                        "seed": seed})
                    if spec.get("replay") or spec.get("deterministic"):
                        ref = src or stored.get((robot_name, task.id, seed)) or {}
                        sg = (ref.get("agent") or {}).get("ground") or {}
                        r["control_source_found"] = bool(ref)
                        r["grounding_matches_stored"] = (
                            pre is not None and bool(sg)
                            and list(pre.goal_ids) == list(sg.get("goal_ids") or [])
                            and {k: round(float(v), 3) for k, v in pre.hazard_margins.items()}
                            == {k: round(float(v), 3) for k, v in (sg.get("hazard_margins") or {}).items()})
                        r["instruction_matches_stored"] = ref.get("instruction_text") == task.text
                    # The path stays. Re-scoring a metric used to mean
                    # re-planning every run, agent calls included; with the
                    # trajectory on disk it is a local computation.
                    r.pop("agent_trace", None)
                    runs.append(r)
                    if task.feasible_by_construction:
                        flag = "ok  " if r.get("success") else "FAIL"
                    else:
                        flag = "rej " if not r.get("success") else "WRONG"
                    print(f"  [{done:3d}/{total}] {flag} {robot_name:11s} "
                          f"{method:14s} {task.id} seed{seed} "
                          f"{r.get('final', {}).get('length_m', '-')}", flush=True)

    summary = build_summary(runs, robots, methods, {
        "margin_offset": args.margin_offset,
        "fixed_margin": args.fixed_margin,
        "tasks": tasks_meta, "reachable_area": reach_report, "seeds": args.seeds,
        "task_seed": args.task_seed,
        "wall_clock_s": round(time.time() - t0, 1)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    report(summary, methods)
    print(f"\nwrote {args.out} ({summary['wall_clock_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
