#!/usr/bin/env python3
"""Does scan completion make routes safer? A simulated-occlusion test.

    PYTHONPATH=. .venv/bin/python scripts/ablate_completion.py --site scene1 [--dry]

The simulated stages have complete geometry, so occlusion can be imposed
and its cost measured against the truth. For every object Sec. III-C would
complete (a category prior, navigation-critical, reaching into a ground
body's band), a contiguous 40% end of its surface is withheld -- what
occlusion removes, as in ``holdout_deviation``. Three worlds are then
planned on:

  full       the complete cloud (the evaluated pipeline)
  occluded   withheld points gone; spot boxes and the stage-1 free-space
             map rebuilt from what is left
  completed  occluded cloud after ``audit_and_complete``; spot boxes from
             observed + synthetic points

Scene-graph ids, areas and edges are held fixed, so every stored BodyNav
program and its grounding replay unchanged: no model call is made (parse
and grounding come from the stored run). Every route is judged on the
*full* 2.5 cm reference and against the *full* hazard boxes, so a route
that clips geometry the occluded map lost is scored as unsafe.

Writes results/<site>/ablation_completion.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from bodynav.eval.results import load_results   # maps pre-rename keys

REPO = Path(__file__).resolve().parents[1]
FRACTION = 0.4
GROUND_TOP = 1.85          # highest ground band (H1); in-band test


def withheld_mask(pts: np.ndarray, fraction: float = FRACTION) -> np.ndarray:
    """True for the contiguous end of an object along its principal axis."""
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    s = c @ vt[0]
    return s >= np.quantile(s, 1.0 - fraction)


def rebox(graph, node_ids_by_instance: dict, xyz, instance, ids) -> None:
    """Spot AABB and centroid from the points each instance now has."""
    for cid in ids:
        node = node_ids_by_instance.get(int(cid))
        if node is None:
            continue
        sel = instance == cid
        if not sel.any():
            continue
        a = graph.attributes(node)
        p = xyz[sel]
        a.aabb_min = p.min(axis=0)
        a.aabb_max = p.max(axis=0)
        a.centroid = p.mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="scene1")
    ap.add_argument("--robot", action="append", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--dry", action="store_true",
                    help="occlusion and completion statistics only; no planning")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    args = ap.parse_args()
    site = args.site
    robots = args.robot or ["husky_ur5", "unitree_h1"]

    from bodynav.abstraction.completion import (CATEGORY_PRIORS, NAVIGATION_CRITICAL,
                                             CompletionParams, audit_and_complete)
    from bodynav.core.types import Priority, TaskSpec
    from bodynav.eval.baselines import hazard_compliance
    from bodynav.eval.sampling import APPROACH_RADIUS_M
    from bodynav.eval.task_maps import TaskMapBuilder, build_params, load_points
    from bodynav.io import load_grid_arrays
    from bodynav.llm.grounding import ResolvedTask
    from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
    from bodynav.planning.obstacles import build_collision_primitives, compute_clearance_fields
    from bodynav.robots import load_profile
    from bodynav.scene_graph.graph import BodyNavSceneGraph
    from bodynav.scene_graph.hierarchy import detect_floors, segment_areas

    import importlib.util
    spec = importlib.util.spec_from_file_location("plan_path", REPO / "scripts" / "plan_path.py")
    plan_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_mod)

    t0 = time.time()
    config = yaml.safe_load(args.config.read_text())
    comp = config["completion"]
    cparams = CompletionParams(tau_c=comp["tau_c"], emphasis_strength=comp["emphasis_strength"],
                               max_synthetic_ratio=comp["max_synthetic_ratio"])
    graph_path = REPO / f"results/{site}/scene_graph_nodes.json"
    true_graph = BodyNavSceneGraph.load(str(graph_path))
    with np.load(REPO / f"data/{site}/free_space.npz") as fp:
        true_prior = {k: fp[k] for k in fp.files}
    pts = load_points(REPO / f"data/{site}/{site}.ply", REPO / f"data/{site}/point_labels.npz")
    xyz = np.asarray(pts.xyz, float)
    instance = np.asarray(pts.instance_id, int)
    semantic = np.asarray(pts.semantic, dtype=object)
    floor_z = float(true_prior["floor_z"])

    by_instance = {}
    for node in true_graph.nodes_by_level("spot"):
        by_instance[int(node.split("_")[-1])] = node

    # -- occlusion ---------------------------------------------------------
    keep = np.ones(len(xyz), bool)
    targets = []
    for cid in np.unique(instance):
        if cid < 0:
            continue
        sel = np.flatnonzero(instance == cid)
        cat = str(semantic[sel[0]])
        if cat not in CATEGORY_PRIORS or cat not in NAVIGATION_CRITICAL:
            continue
        if (xyz[sel, 2] - floor_z).min() > GROUND_TOP or len(sel) < 10:
            continue
        cut = withheld_mask(xyz[sel])
        keep[sel[cut]] = False
        targets.append(dict(instance=int(cid), node=by_instance.get(int(cid)), category=cat,
                            points=int(len(sel)), withheld=int(cut.sum())))
    occluded = pts.select(keep) if hasattr(pts, "select") else None
    graph_o = BodyNavSceneGraph.load(str(graph_path))
    rebox(graph_o, by_instance, occluded.xyz, np.asarray(occluded.instance_id, int),
          [t["instance"] for t in targets])

    completed, _, report = audit_and_complete(graph_o, occluded, cparams)
    synth = (np.asarray(completed.extras.get("synthetic"), bool)
             if "synthetic" in completed.extras else np.zeros(len(completed.xyz), bool))
    target_ids = {t["instance"] for t in targets}
    graph_c = BodyNavSceneGraph.load(str(graph_path))
    rebox(graph_c, by_instance, completed.xyz, np.asarray(completed.instance_id, int),
          [t["instance"] for t in targets])
    comp_inst = np.asarray(completed.instance_id, int)
    for t in targets:
        t["synthetic_added"] = int((synth & (comp_inst == t["instance"])).sum())
    print(f"{site}: {len(targets)} in-band completable objects, "
          f"{sum(t['withheld'] for t in targets):,} points withheld; completion added "
          f"{int(synth.sum()):,} synthetic points to "
          f"{sum(1 for t in targets if t['synthetic_added'])} of them "
          f"(all completed spots: {sum(report.completed_by_category.values())})", flush=True)

    # -- stage-1 free space, changed only where the geometry changed --------
    floor = detect_floors(xyz)[0]

    lo_xy, hi_xy = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)

    def free_of(x):
        # Synthetic points can extend past the scan footprint (an extruded
        # duct); stage 1's grid is defined by that footprint, so they are
        # clipped to it for the free-space test only.
        inside = np.all((x[:, :2] >= lo_xy) & (x[:, :2] <= hi_xy), axis=1)
        _, meta, free = segment_areas(x[inside], floor, cell=float(true_prior["cell"]),
                                      return_free=True)
        return free, meta

    f_full, m_full = free_of(xyz)
    priors = {"full": true_prior}
    for name, cloud in (("occluded", occluded), ("completed", completed)):
        f_var, m_var = free_of(np.asarray(cloud.xyz, float))
        if f_var.shape != true_prior["free"].shape or f_full.shape != f_var.shape:
            raise SystemExit(f"free-space grid changed shape for {name}")
        changed = f_full != f_var
        p = dict(true_prior)
        p["free"] = np.where(changed, f_var, true_prior["free"])
        priors[name] = p
        print(f"  free space {name}: {int(changed.sum())} cells differ from full "
              f"({changed.sum() * float(true_prior['cell']) ** 2:.1f} m2)", flush=True)

    out = {"site": site, "fraction_withheld": FRACTION, "targets": targets,
           "synthetic_points": int(synth.sum()),
           "free_cells_changed": {k: int((priors[k]["free"] != true_prior["free"]).sum())
                                  for k in ("occluded", "completed")},
           "runs": []}
    # Figure marks (item 9): the withheld and synthetic points of this test.
    wx = xyz[~keep]
    np.savez_compressed(REPO / f"results/{site}/ablation_completion_marks.npz",
                        withheld=wx.astype(np.float32),
                        synthetic=np.asarray(completed.xyz)[synth].astype(np.float32),
                        synthetic_instance=comp_inst[synth].astype(np.int32),
                        target_instances=np.array([t["instance"] for t in targets], np.int32))
    if args.dry:
        print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
        return 0

    worlds = {"full": (pts, true_graph), "occluded": (occluded, graph_o),
              "completed": (completed, graph_c)}
    stored = [r for f in ("batch.json",) for r in
              json.loads((REPO / f"results/{site}/{f}").read_text())["runs"]
              if r["method"] == "bodynav" and r["seed"] < args.seeds]
    tasks = load_results(REPO / f"results/{site}/batch.json")["tasks"]
    vparams = build_params(config)

    for robot_name in robots:
        robot = load_profile(robot_name, config_dir=REPO / "configs" / "robots")
        ref_grid = load_grid_arrays(REPO / f"data/{site}/grid_{robot_name}_reference.npz")
        ref_fields = compute_clearance_fields(
            ref_grid, build_collision_primitives(true_graph, ref_grid, robot), robot,
            pitch=0.1, floor_z=floor_z, free_prior=true_prior)
        planner = ValueGuidedAStar(PlannerParams(eta=float(config["planning"]["eta"]),
                                                 goal_snap_m=APPROACH_RADIUS_M), robot)
        builders = {
            name: TaskMapBuilder(cloud, g, vparams,
                                 cache_dir=REPO / (f"data/{site}/task_grids"
                                                   + ("" if name == "full" else f"_{name}")))
            for name, (cloud, g) in worlds.items()}
        runs_here = [r for r in stored if r["robot"] == robot_name]
        for r in runs_here:
            task = SimpleNamespace(**tasks[f"{robot_name}/{r['task_id']}"])
            ag = r.get("agent") or {}
            gr, ps = ag.get("ground") or {}, ag.get("parse") or {}
            margin = max((gr.get("hazard_margins") or {}).values(), default=0.0)
            for name, (cloud, g) in worlds.items():
                t1 = time.time()
                resolved = ResolvedTask(
                    task=TaskSpec(start=None, goal=ps.get("goal") or task.goal_category,
                                  hazards=list(ps.get("hazards") or []),
                                  priority=Priority(r["priority"]),
                                  raw_instruction=task.text),
                    start_id=task.start_id, goal_ids=list(gr.get("goal_ids") or []),
                    hazard_ids=list(gr.get("hazard_ids") or []),
                    hazard_margins={k: float(v) for k, v in (gr.get("hazard_margins") or {}).items()})
                grid = builders[name].grid_for(robot, task.start_id, task.goal_id,
                                               hazard_margin_m=margin)
                fields = compute_clearance_fields(
                    grid, build_collision_primitives(g, grid, robot), robot, pitch=0.1,
                    floor_z=floor_z, free_prior=priors[name])
                try:
                    res = plan_mod.plan_once(
                        robot, grid, g, priors[name], config, instruction=task.text,
                        pitch=0.1, value_maps="bodynav", priority=task.priority, seed=r["seed"],
                        verbose=False, fields=fields, true_fields=ref_fields, cache={},
                        start_id=task.start_id, goal_id=task.goal_id,
                        start_xyz=np.array(task.start_xyz), goal_xyz=np.array(task.goal_xyz),
                        hazard_category=task.hazard_category,
                        hazard_margin_m=task.hazard_margin_m,
                        hazard_node_ids=task.hazard_node_ids,
                        agent_pre=resolved, replay_code=ag.get("code") or "",
                        hard_gate=True)
                except Exception as exc:  # recorded, not raised
                    res = {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}"[:200]}
                row = {"world": name, "robot": robot_name, "task_id": r["task_id"],
                       "seed": r["seed"], "pair_id": task.pair_id,
                       "hazard_margin_m": task.hazard_margin_m,
                       "feasible_by_construction": task.feasible_by_construction,
                       "success": bool(res.get("success")),
                       "failure_reason": res.get("failure_reason"),
                       "margin_fallback": res.get("margin_fallback")}
                if res.get("success"):
                    path = np.asarray(res["path"], float)
                    # judged on the full world: reference fields and true boxes
                    ht = hazard_compliance(path, true_graph, robot,
                                           category=task.hazard_category,
                                           margin_m=task.hazard_margin_m,
                                           node_ids=task.hazard_node_ids, floor_z=floor_z)
                    near = min((float(np.min(np.linalg.norm(
                        path[:, :2] - np.asarray(true_graph.attributes(t["node"]).centroid)[:2],
                        axis=1))) for t in targets if t["node"]), default=None)
                    row.update(clearance_ok=res.get("clearance_ok"),
                               vertical_ok=res.get("vertical_ok"),
                               hazard_ok=res.get("hazard_ok"),
                               worst_lateral_m=res.get("worst_lateral_m"),
                               min_hazard_distance_m=ht.get("min_distance_m"),
                               margin_compliant=ht.get("compliant"),
                               length_m=res["final"]["length_m"],
                               nearest_target_m=near, path=res["path"])
                out["runs"].append(row)
                print(f"  {robot_name:10s} {r['task_id']} s{r['seed']} {name:9s} "
                      f"{'ok ' if row['success'] else 'FAIL'} "
                      f"safe={row.get('clearance_ok')} margin={row.get('margin_compliant')} "
                      f"({time.time() - t1:.1f}s)", flush=True)

    out["wall_clock_s"] = round(time.time() - t0, 1)
    dest = REPO / f"results/{site}/ablation_completion.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest} ({out['wall_clock_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
