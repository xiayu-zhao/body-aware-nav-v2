#!/usr/bin/env python3
"""Stage 2 per embodiment: Eq. 2 slice -> Eq. 3 kernel -> V_t -> Eq. 4 grid.

    PYTHONPATH=. .venv/bin/python scripts/run_stage2.py \
        --cloud data/labfloor/labfloor.ply \
        --graph results/labfloor/scene_graph_nodes.json \
        --robot husky_ur5 --robot unitree_h1

Runs the same scan through the same pipeline once per robot and writes
``results/labfloor/abstraction_<robot>.json``. The point of the exercise
is the *difference* between the outputs: one building, one scan, two maps,
because the bodies differ. That contrast is the paper's multi-embodiment
claim and it needs no robot to move.

The task corridor is grounded in the scene graph rather than in an
instruction, since stage 3 does not exist yet. Once it does, the same
call takes the LLM's grounded entities and nothing else changes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

from bodynav.abstraction import AdaptiveVoxelizer, VoxelizerParams, height_slice
from bodynav.abstraction.robot_filter import workspace_kernel_mask
from bodynav.abstraction.task_filter import build_corridor, priority_merge, task_mask
from bodynav.io import load_cloud, save_grid
from bodynav.robots import load_profile
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints, detect_floors

REPO = Path(__file__).resolve().parents[1]


def build_params(config: dict) -> VoxelizerParams:
    v = config["abstraction"]["voxel"]
    return VoxelizerParams(
        r0=v["r0_m"], r_min=v["r_min_m"], alpha=v["alpha"],
        beta=v["beta"], gamma=v["gamma"],
        foothold_buffer=v["foothold_buffer_m"],
    )


def pick_task_anchors(graph: BodyNavSceneGraph) -> tuple[str, str]:
    """The most distant pair of areas that are actually connected.

    Two constraints, both of which matter. The pair must be far apart, or
    the corridor covers a corner of the floor and the reduction looks
    better than it is. And it must be *reachable*: the scanned free space
    on this floor is not one component -- the loop is broken where a door
    is shut or a corridor is stacked with material -- so the farthest pair
    overall may have no route between them, and a corridor drawn straight
    between them would cut through the building.
    """
    import networkx as nx

    areas = graph.nodes_by_level("area")
    if len(areas) < 2:
        raise SystemExit("need at least two areas to define a task corridor")

    sub = graph.g.subgraph(areas).copy()
    sub.remove_edges_from([
        (u, v) for u, v, d in sub.edges(data=True) if d.get("kind") != "adjacency"
    ])
    component = max(nx.connected_components(sub), key=len)
    reachable = [a for a in areas if a in component]
    if len(reachable) < 2:
        raise SystemExit("no two areas are connected -- check the free-space map")

    xy = np.array([graph.attributes(a).centroid[:2] for a in reachable])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    print(f"task anchors    {reachable[i]} -> {reachable[j]} "
          f"({d[i, j]:.1f} m apart, largest connected component: "
          f"{len(reachable)}/{len(areas)} areas)")
    return reachable[i], reachable[j]


def run_robot(name: str, points: LabeledPoints, graph: BodyNavSceneGraph,
              params: VoxelizerParams, floor: tuple[float, float],
              config_dir: Path, args) -> tuple[dict, object]:
    robot = load_profile(name, config_dir=config_dir)
    t0 = time.time()
    n_raw = len(points.xyz)

    # -- Eq. 2: operational height band ------------------------------
    sliced = height_slice(points, robot)
    n_slice = len(sliced.xyz)
    slice_red = 1.0 - n_slice / n_raw

    # -- Eq. 3: workspace kernel along the task corridor --------------
    start_id, goal_id = pick_task_anchors(graph)
    corridor = build_corridor(graph, start_id, goal_id, [])
    kernel = workspace_kernel_mask(sliced.xyz, corridor.centerline[:, :2], robot)

    # -- V_t: corridor u hazard volumes, intersected with the kernel --
    hazards = graph.find(hazardous=True)
    v_t = task_mask(sliced, corridor, hazards, graph)
    v_t_kernel = v_t & kernel
    merged = priority_merge(sliced, v_t_kernel,
                            critical_spacing=args.critical_spacing,
                            background_spacing=args.background_density)
    n_task = len(merged.xyz)

    # -- Eq. 4: adaptive voxelization ---------------------------------
    vox = AdaptiveVoxelizer(params, robot)
    grid = vox.voxelize(merged, merged.extras["v_t"])
    stats = vox.last_stats
    elapsed = time.time() - t0

    by_res = {f"{r:.3f}": int(n) for r, n in sorted(stats.voxels_by_resolution.items())}
    mem_mb = len(grid) * 96 / 1e6  # ~96 B per sparse voxel record

    route_pts = corridor.centerline
    route_len = float(np.linalg.norm(np.diff(route_pts, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(route_pts[-1] - route_pts[0]))

    out = {
        "robot": robot.name,
        "band": f"{robot.height_band[0]:.2f}--{robot.height_band[1]:.2f}",
        "band_m": list(robot.height_band),
        "band_ratio": round(
            (robot.height_band[1] - robot.height_band[0]) / (floor[1] - floor[0]), 4),
        "kernel_r": round(robot.r_reach + robot.r_safety, 3),
        "required_vertical_clearance": robot.required_vertical_clearance,
        "points_raw": n_raw,
        "points_after_slice": n_slice,
        "slice_reduction": round(slice_red, 4),
        "slice": f"{n_slice / 1e6:.2f} ({slice_red * 100:.0f}\\%)",
        "kernel_keeps": int(kernel.sum()),
        "v_t_points": int(v_t_kernel.sum()),
        "points_after_taskfilter": n_task,
        "taskfilter_reduction": round(1.0 - n_task / n_raw, 4),
        "taskfilter": f"{n_task / 1e6:.2f} ({(1 - n_task / n_raw) * 100:.0f}\\%)",
        "n_voxels": len(grid),
        # against the RAW cloud: `stats.reduction` is relative to the
        # task-filtered input, which understates the pipeline's effect and
        # is not the number Table R2 asks for.
        "voxels": f"{len(grid) / 1e3:.1f} ({(1 - len(grid) / n_raw) * 100:.1f}\\%)",
        "voxel_reduction_vs_input": round(stats.reduction, 4),
        "total_reduction_vs_raw": round(1.0 - len(grid) / n_raw, 5),
        "mean_resolution_m": round(stats.mean_resolution, 4),
        "fine_share": round(stats.fine_share * 100, 2),
        "voxels_by_resolution": by_res,
        "time_s": round(elapsed, 2),
        "memory_mb": round(mem_mb, 1),
        "time_mem": f"{elapsed:.1f} / {mem_mb:.1f}",
        "task": {"start": start_id, "goal": goal_id, "hazard_nodes": len(hazards),
                 "corridor_len_m": round(route_len, 1),
                 "straight_line_m": round(straight, 1),
                 "detour_ratio": round(route_len / straight, 2) if straight else None},
    }
    return out, grid


def run_robot_reference(name: str, points: LabeledPoints, params: VoxelizerParams,
                        config_dir: Path, resolution_m: float = 0.025):
    """The yardstick: the finest faithful map of the same slice.

    Not a method -- nothing plans on this. It exists because grading BodyNav
    with BodyNav's own abstraction is grading the work with the answer key it
    wrote. A map that coarsens clutter can quietly widen a corridor, and
    the only way to know whether it has is to check the accepted path
    against a map that did not coarsen anything.

    2.5 cm is the pipeline's own r_min, so this is the finest map the
    abstraction stage could ever produce -- roughly five times BodyNav's
    memory, which is exactly why it is a reference and not a baseline.
    The evidence threshold is left at the pipeline's value: at 2.5 cm two
    voxels is 5 cm of material, the same physical amount two voxels means
    at BodyNav's 2.5 cm modal resolution, so the two maps are asking the same
    question of the same cloud.
    """
    robot = load_profile(name, config_dir=config_dir)
    t0 = time.time()
    sliced = height_slice(points, robot)
    up = dataclasses.replace(params, r0=resolution_m, r_min=resolution_m,
                             alpha=0.0, beta=0.0, gamma=0.0)
    grid = AdaptiveVoxelizer(up, robot).voxelize(
        sliced, np.zeros(len(sliced.xyz), dtype=bool))
    stats = {
        "robot": robot.name, "map_model": "reference",
        "resolution_m": resolution_m,
        "points_after_slice": len(sliced.xyz),
        "n_voxels": len(grid),
        "memory_mb": round(len(grid) * 96 / 1e6, 1),
        "time_s": round(time.time() - t0, 2),
    }
    return stats, grid


def run_robot_uniform(name: str, points: LabeledPoints, graph: BodyNavSceneGraph,
                      params: VoxelizerParams, floor: tuple[float, float],
                      config_dir: Path, args, target_voxels: int):
    """D2 -- uniform-resolution voxels at matched memory.

    The ablation of Eq. 4 alone. Eq. 2's height band is kept, because
    removing it as well would confound this baseline with D1 and neither
    result would say which term did the work. What changes is only *how
    the budget is spent*: BodyNav puts 2.5 cm cells where the instruction
    points and 40 cm cells everywhere else; this baseline spends the same
    number of cells at one resolution everywhere.

    The resolution is not chosen by hand -- it is solved for, by bisection
    on the voxel count, so the comparison is at equal memory rather than
    at a resolution flattering to either side.
    """
    robot = load_profile(name, config_dir=config_dir)
    t0 = time.time()
    n_raw = len(points.xyz)
    sliced = height_slice(points, robot)
    zero_mask = np.zeros(len(sliced.xyz), dtype=bool)

    def count_at(r: float) -> tuple[int, object]:
        up = dataclasses.replace(params, r0=r, r_min=r,
                                 alpha=0.0, beta=0.0, gamma=0.0)
        grid = AdaptiveVoxelizer(up, robot).voxelize(sliced, zero_mask)
        return len(grid), grid

    lo_r, hi_r, best = 0.01, 0.60, None
    n_lo, _ = count_at(lo_r)
    n_hi, grid = count_at(hi_r)
    if n_lo <= target_voxels:
        best, r_star = grid, lo_r
        n_star = n_lo
    else:
        for _ in range(12):
            mid = 0.5 * (lo_r + hi_r)
            n_mid, grid_mid = count_at(mid)
            if n_mid > target_voxels:
                lo_r = mid
            else:
                hi_r = mid
            best, r_star, n_star = grid_mid, mid, n_mid
            if abs(n_mid - target_voxels) <= 0.02 * target_voxels:
                break
    elapsed = time.time() - t0
    stats = {
        "robot": robot.name, "map_model": "uniform",
        "band": f"{robot.height_band[0]:.2f}--{robot.height_band[1]:.2f}",
        "points_raw": n_raw, "points_after_slice": len(sliced.xyz),
        "target_voxels": int(target_voxels),
        "n_voxels": int(n_star),
        "uniform_resolution_m": round(float(r_star), 4),
        "memory_ratio_vs_adaptive": round(n_star / max(target_voxels, 1), 3),
        "time_s": round(elapsed, 2),
        "memory_mb": round(n_star * 96 / 1e6, 1),
    }
    return stats, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True, type=Path)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None,
                    help="per-point labels from stage 1 (.npz)")
    ap.add_argument("--robot", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/labfloor"))
    ap.add_argument("--grid-dir", type=Path, default=Path("data/labfloor"))
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--config-dir", type=Path, default=Path("configs/robots"))
    ap.add_argument("--critical-spacing", type=float, default=0.02)
    ap.add_argument("--background-density", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=None)
    ap.add_argument("--reference", action="store_true",
                    help="also build the fine reference map used to judge "
                         "every method, as grid_<robot>_reference.npz")
    ap.add_argument("--uniform-target", type=int, default=None,
                    help="voxel count the uniform (D2) map must match; "
                         "defaults to the adaptive run's own count. Set it to "
                         "BodyNav's mean per-task map size when the evaluation "
                         "rebuilds maps per instruction, or 'equal memory' "
                         "compares against a map BodyNav no longer uses.")
    ap.add_argument("--uniform", action="store_true",
                    help="also build the D2 uniform-resolution grid at the "
                         "same voxel count, as grid_<robot>_uniform.npz")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    params = build_params(config)

    xyz, colors = load_cloud(args.cloud, max_points=args.max_points)
    floor = detect_floors(xyz)[0]
    graph = BodyNavSceneGraph.load(str(args.graph))
    if args.labels and args.labels.exists():
        lab = np.load(args.labels, allow_pickle=True)
        vocab = lab["vocabulary"]
        semantic = np.asarray(vocab[lab["semantic_code"]], dtype=object)
        points = LabeledPoints(
            xyz=xyz, floor_id=lab["floor_id"].astype(int),
            area_id=lab["area_id"].astype(int),
            instance_id=lab["instance_id"].astype(int),
            semantic=semantic, colors=colors,
        )
        counts = {str(k): int(v) for k, v in
                  zip(*np.unique(semantic, return_counts=True))}
        print(f"labels {counts}")
    else:
        print("labels none -- voxels will carry 'unknown', and stage 4 will "
              "have no floor to stand on")
        points = LabeledPoints.from_xyz(xyz, colors=colors)
    print(f"cloud {len(xyz):,} points | storey {floor[1] - floor[0]:.3f} m | "
          f"graph {graph.g.number_of_nodes()} nodes")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.grid_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.robot:
        stats, grid = run_robot(name, points, graph, params, floor,
                                args.config_dir, args)
        summary[name] = stats
        (args.out_dir / f"abstraction_{name}.json").write_text(
            json.dumps(stats, indent=1) + "\n")
        save_grid(grid, args.grid_dir / f"grid_{name}.npz")
        if args.reference:
            r_stats, r_grid = run_robot_reference(name, points, params,
                                                  args.config_dir)
            (args.out_dir / f"abstraction_{name}_reference.json").write_text(
                json.dumps(r_stats, indent=1) + "\n")
            save_grid(r_grid, args.grid_dir / f"grid_{name}_reference.npz")
            print(f"  reference       {r_stats['n_voxels']:>10,} voxels at "
                  f"{r_stats['resolution_m'] * 100:.1f} cm "
                  f"({r_stats['n_voxels'] / max(stats['n_voxels'], 1):.1f}x "
                  f"adaptive memory)")
        if args.uniform:
            u_stats, u_grid = run_robot_uniform(
                name, points, graph, params, floor, args.config_dir, args,
                target_voxels=args.uniform_target or stats["n_voxels"])
            (args.out_dir / f"abstraction_{name}_uniform.json").write_text(
                json.dumps(u_stats, indent=1) + "\n")
            save_grid(u_grid, args.grid_dir / f"grid_{name}_uniform.npz")
            print(f"  D2 uniform      {u_stats['n_voxels']:>10,} voxels at "
                  f"r = {u_stats['uniform_resolution_m'] * 100:.1f} cm "
                  f"({u_stats['memory_ratio_vs_adaptive']:.2f}x adaptive memory)")
        print(f"\n{name}")
        print(f"  band {stats['band']} m ({stats['band_ratio'] * 100:.0f}% of storey)"
              f"   kernel r = {stats['kernel_r']} m")
        print(f"  Eq. 2 slice      {stats['points_after_slice']:>10,}  "
              f"({stats['slice_reduction'] * 100:.1f}% reduction)")
        print(f"  V_t + merge      {stats['points_after_taskfilter']:>10,}  "
              f"({stats['taskfilter_reduction'] * 100:.1f}% reduction)")
        print(f"  Eq. 4 voxels     {stats['n_voxels']:>10,}  "
              f"({stats['total_reduction_vs_raw'] * 100:.2f}% vs raw)   "
              f"mean r = {stats['mean_resolution_m'] * 100:.1f} cm")
        print(f"  resolutions      {stats['voxels_by_resolution']}")
        print(f"  {stats['time_s']} s, {stats['memory_mb']} MB")

    if len(summary) > 1:
        names = list(summary)
        a, b = summary[names[0]], summary[names[1]]
        print(f"\ncontrast  {names[0]} vs {names[1]}: "
              f"slice {a['slice_reduction'] * 100:.1f}% vs {b['slice_reduction'] * 100:.1f}%, "
              f"voxels {a['n_voxels']:,} vs {b['n_voxels']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
