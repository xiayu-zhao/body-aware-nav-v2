#!/usr/bin/env python3
"""Re-plan a chosen task through every method and keep the trajectories.

    PYTHONPATH=. .venv/bin/python scripts/export_figure_paths.py \
        --site labfloor --robot unitree_h1 --pair p02

The batch drops accepted paths -- 882 of them would dwarf the metrics
file -- so the qualitative figures re-plan the handful they draw. Same
code path as the batch, same maps, same seeds, so a route in a figure is
the route that produced the number in the table.

Writes one npz per (robot, task, method) plus a manifest naming the
hazard the instruction referred to and the margin it stated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_plan_module():
    spec = importlib.util.spec_from_file_location(
        "bodynav_plan_path", REPO / "scripts" / "plan_path.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    from bodynav.eval.baselines import costmap_2d_fields, costmap_3d_fields
    from bodynav.eval.sampling import APPROACH_RADIUS_M, sample_tasks
    from bodynav.eval.task_maps import TaskMapBuilder, build_params, load_points
    from bodynav.io import load_grid_arrays
    from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
    from bodynav.planning.obstacles import (build_collision_primitives,
                                         compute_clearance_fields)
    from bodynav.robots import load_profile
    from bodynav.scene_graph.graph import BodyNavSceneGraph

    run_batch = importlib.util.spec_from_file_location(
        "bodynav_run_batch", REPO / "scripts" / "run_batch.py")
    rb = importlib.util.module_from_spec(run_batch)
    sys.modules["bodynav_run_batch"] = rb
    run_batch.loader.exec_module(rb)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="labfloor")
    ap.add_argument("--robot", action="append", default=None)
    ap.add_argument("--pair", default=None, help="pair id, e.g. p02")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    site = args.site
    robots = args.robot or ["husky_ur5", "unitree_h1"]
    out = args.out or REPO / "results" / site / "figure_paths"
    out.mkdir(parents=True, exist_ok=True)

    plan_mod = load_plan_module()
    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    graph = BodyNavSceneGraph.load(
        str(REPO / "results" / site / "scene_graph_nodes.json"))
    with np.load(REPO / "data" / site / "free_space.npz") as fp:
        prior = {k: fp[k] for k in fp.files}
    pts = load_points(REPO / "data" / site / f"{site}.ply",
                      REPO / "data" / site / "point_labels.npz")
    builder = TaskMapBuilder(pts, graph, build_params(config),
                             cache_dir=REPO / "data" / site / "task_grids")

    manifest: dict = {"site": site, "tasks": {}}
    for robot_name in robots:
        robot = load_profile(robot_name, config_dir=REPO / "configs" / "robots")
        grid = load_grid_arrays(REPO / "data" / site / f"grid_{robot_name}.npz")
        fixed = compute_clearance_fields(
            grid, build_collision_primitives(graph, grid, robot), robot,
            pitch=0.1, floor_z=float(prior["floor_z"]), free_prior=prior)
        ref_grid = load_grid_arrays(
            REPO / "data" / site / f"grid_{robot_name}_reference.npz")
        ref_fields = compute_clearance_fields(
            ref_grid, build_collision_primitives(graph, ref_grid, robot), robot,
            pitch=0.1, floor_z=float(prior["floor_z"]), free_prior=prior)
        uni_grid = load_grid_arrays(
            REPO / "data" / site / f"grid_{robot_name}_uniform.npz")
        uni_fields = compute_clearance_fields(
            uni_grid, build_collision_primitives(graph, uni_grid, robot), robot,
            pitch=0.1, floor_z=float(prior["floor_z"]), free_prior=prior)
        maps_for = {"fixed": (grid, fixed),
                    "costmap": (grid, costmap_2d_fields(fixed)),
                    "costmap3d": (grid, costmap_3d_fields(fixed)),
                    "uniform": (uni_grid, uni_fields)}

        planner = ValueGuidedAStar(
            PlannerParams(eta=float(config["planning"]["eta"]),
                          goal_snap_m=APPROACH_RADIUS_M), robot)
        tasks = sample_tasks(graph, ref_fields, planner, robot, n=12, seed=0)
        chosen = [t for t in tasks
                  if t.feasible_by_construction
                  and (args.pair is None or t.pair_id == args.pair)]
        if not chosen:
            print(f"  {robot_name}: no task matching --pair {args.pair}")
            continue

        cache: dict = {}
        for task in chosen:
            pre = None
            margin = 0.0
            for method, spec in rb.METHODS.items():
                if spec["map"] == "task":
                    if spec["use_instruction"] and pre is None:
                        from bodynav.llm.agent import AgentConfig, ConstrainedAgent
                        from bodynav.llm.backends import load_backend
                        client = load_backend(None, offline=args.offline)
                        agent = ConstrainedAgent(AgentConfig(model="llm-large"),
                                                 client)
                        pre = agent.pre_ground(task.text, graph)
                        if pre.hazard_margins:
                            margin = max(pre.hazard_margins.values())
                    g = builder.grid_for(robot, task.start_id, task.goal_id,
                                         hazard_margin_m=margin)
                    f = compute_clearance_fields(
                        g, build_collision_primitives(graph, g, robot), robot,
                        pitch=0.1, floor_z=float(prior["floor_z"]),
                        free_prior=prior)
                    plan_grid, plan_fields = g, f
                else:
                    plan_grid, plan_fields = maps_for[spec["map"]]
                try:
                    r = plan_mod.plan_once(
                        robot, plan_grid, graph, prior, config,
                        instruction=task.text if spec["use_instruction"] else None,
                        offline=args.offline, pitch=0.1,
                        value_maps=spec["value_maps"], priority=task.priority,
                        seed=0, verbose=False, fields=plan_fields,
                        true_fields=ref_fields, cache=cache,
                        start_id=task.start_id, goal_id=task.goal_id,
                        start_xyz=np.array(task.start_xyz),
                        goal_xyz=np.array(task.goal_xyz),
                        hazard_category=task.hazard_category,
                        hazard_margin_m=task.hazard_margin_m,
                        hazard_node_ids=task.hazard_node_ids,
                        agent_pre=pre if spec["use_instruction"] else None)
                except Exception as exc:
                    print(f"    {method}: {type(exc).__name__}: {exc}")
                    continue
                if not r.get("success"):
                    print(f"    {method}: {r.get('failure_reason')}")
                    continue
                np.savez_compressed(
                    out / f"{robot_name}_{task.id}_{method}.npz",
                    path=np.asarray(r["path"], dtype=float))
                print(f"    {robot_name} {task.id} {method:15s} "
                      f"{r['final']['length_m']:6.2f} m  "
                      f"hazard {(r.get('hazard_task') or {}).get('min_distance_m')}")
            manifest["tasks"][f"{robot_name}/{task.id}"] = {
                **task.as_dict(),
                "hazard_node_ids": list(task.hazard_node_ids),
            }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
