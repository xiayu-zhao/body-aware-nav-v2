#!/usr/bin/env python3
"""Stage 2b: semantics-guided completion and emphasis (Sec. III-C).

    PYTHONPATH=. .venv/bin/python scripts/run_completion.py \
        --cloud data/labfloor/labfloor.ply \
        --labels data/labfloor/point_labels.npz --robot husky_ur5

Audits every spot against its category prior, completes those below
tau_c, emits the emphasis cue phi_E, and re-voxelizes with it so the
effect on the map is measured rather than assumed.

The accuracy of the synthesized geometry is checked by **holdout**: a
contiguous slice of each object's observed surface is withheld, the rest
is completed, and the synthetic points are compared against what was
withheld. Sec. III-C is unfalsifiable without that number -- it reports
geometry that by definition nobody observed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from bodynav.abstraction import AdaptiveVoxelizer, VoxelizerParams, height_slice
from bodynav.abstraction.completion import (
    CATEGORY_PRIORS,
    CompletionParams,
    audit_and_complete,
    completeness_score,
    holdout_deviation,
)
from bodynav.abstraction.robot_filter import workspace_kernel_mask
from bodynav.abstraction.task_filter import build_corridor, priority_merge, task_mask
from bodynav.io import load_cloud
from bodynav.robots import load_profile
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints, detect_floors

REPO = Path(__file__).resolve().parents[1]


def load_labelled(cloud: Path, labels: Path):
    xyz, colors = load_cloud(cloud)
    lab = np.load(labels, allow_pickle=True)
    vocab = lab["vocabulary"]
    semantic = np.asarray(vocab[lab["semantic_code"]], dtype=object)
    return LabeledPoints(
        xyz=xyz, floor_id=lab["floor_id"].astype(int),
        area_id=lab["area_id"].astype(int),
        instance_id=lab["instance_id"].astype(int),
        semantic=semantic, colors=colors,
    ), colors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--graph", type=Path,
                    default=Path("results/labfloor/scene_graph_nodes.json"))
    ap.add_argument("--robot", default="husky_ur5")
    ap.add_argument("--out", type=Path,
                    default=Path("results/labfloor/completion.json"))
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--config-dir", type=Path, default=REPO / "configs" / "robots")
    ap.add_argument("--holdout-samples", type=int, default=40)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    comp_cfg = config["completion"]
    params = CompletionParams(
        tau_c=comp_cfg["tau_c"],
        emphasis_strength=comp_cfg["emphasis_strength"],
        max_synthetic_ratio=comp_cfg["max_synthetic_ratio"],
    )
    robot = load_profile(args.robot, config_dir=args.config_dir)
    points, _ = load_labelled(args.cloud, args.labels)
    graph = BodyNavSceneGraph.load(str(args.graph))
    floor = detect_floors(points.xyz)[0]
    print(f"cloud {len(points.xyz):,} points | tau_c {params.tau_c} | robot {robot.name}")

    t0 = time.time()
    augmented, emphasis, report = audit_and_complete(graph, points, params)
    n_synth = len(augmented.xyz) - len(points.xyz)
    print(f"audit    {report.audited} spots with a category prior, "
          f"{report.incomplete} below tau_c")
    print(f"complete {n_synth:,} synthetic points across "
          f"{sum(report.completed_by_category.values())} spots "
          f"({time.time() - t0:.1f}s)")
    top3 = sorted(report.completed_by_category.items(), key=lambda kv: -kv[1])[:3]
    print(f"  by category: {top3}")
    print(f"  mean completeness {report.mean_completeness_before:.3f} -> "
          f"{report.mean_completeness_after:.3f}")

    # -- holdout accuracy of the synthesized surfaces -------------------
    rng = np.random.default_rng(0)
    instance = np.asarray(points.instance_id)
    semantic = np.asarray(points.semantic, dtype=object)
    candidates = [i for i in np.unique(instance) if i >= 0
                  and str(semantic[np.flatnonzero(instance == i)[0]]) in CATEGORY_PRIORS]
    rng.shuffle(candidates)
    deviations, by_category = [], {}
    for spot_id in candidates:
        if len(deviations) >= args.holdout_samples:
            break
        sel = np.flatnonzero(instance == spot_id)
        category = str(semantic[sel[0]])
        dev = holdout_deviation(points.xyz[sel], category, params)
        if dev is not None:
            deviations.append(dev)
            by_category.setdefault(category, []).append(dev)
    dev_cm = float(np.mean(deviations)) * 100 if deviations else None
    per_category = {k: round(float(np.mean(v)) * 100, 1)
                    for k, v in sorted(by_category.items())}
    if dev_cm is not None:
        print(f"holdout  {len(deviations)} objects, mean deviation "
              f"{dev_cm:.1f} cm (median "
              f"{float(np.median(deviations)) * 100:.1f} cm)")
        print(f"  by category (cm): {per_category}")

    # -- effect on the map ----------------------------------------------
    v = config["abstraction"]["voxel"]
    vparams = VoxelizerParams(r0=v["r0_m"], r_min=v["r_min_m"], alpha=v["alpha"],
                              beta=v["beta"], gamma=v["gamma"],
                              foothold_buffer=v["foothold_buffer_m"])

    areas = graph.nodes_by_level("area")
    corridor = build_corridor(graph, areas[0], areas[-1], [])
    hazard_ids = graph.find(hazardous=True)

    def voxelize(pts: LabeledPoints, phi_e=None):
        """Run the abstraction stages with phi_E riding along.

        The cue is carried in ``extras``, which ``LabeledPoints.select``
        subsets with every other per-point array -- so it survives the
        Eq. 2 slice and the priority merge without any stage having to
        know about it.
        """
        if phi_e is not None:
            pts = LabeledPoints(
                xyz=pts.xyz, floor_id=pts.floor_id, area_id=pts.area_id,
                instance_id=pts.instance_id, semantic=pts.semantic,
                colors=pts.colors, extras={**pts.extras, "phi_e": phi_e},
            )
        sliced = height_slice(pts, robot)
        kernel = workspace_kernel_mask(sliced.xyz, corridor.centerline[:, :2], robot)
        v_t = task_mask(sliced, corridor, hazard_ids, graph) & kernel
        merged = priority_merge(sliced, v_t)
        vox = AdaptiveVoxelizer(vparams, robot)
        grid = vox.voxelize(merged, merged.extras["v_t"],
                            emphasis=merged.extras.get("phi_e"))
        return grid, vox.last_stats

    grid_before, stats_before = voxelize(points)
    grid_after, stats_after = voxelize(augmented, emphasis)
    growth = 100.0 * (len(grid_after) / max(len(grid_before), 1) - 1.0)
    print(f"voxels   {len(grid_before):,} -> {len(grid_after):,} "
          f"({growth:+.1f}%), fine share overall "
          f"{stats_before.fine_share * 100:.1f}% -> {stats_after.fine_share * 100:.1f}%")

    # The global fine share barely moves because the task corridor is
    # already at r_min: phi_E has headroom only where phi_T is zero. So
    # the measurement that means anything is the resolution *at the
    # emphasised spots themselves*.
    # emphasis is indexed on the *augmented* cloud, which is longer
    emphasised = {int(i) for i in
                  np.unique(np.asarray(augmented.instance_id)[emphasis > 0]) if i >= 0}

    def spot_resolution(grid, spots):
        from bodynav.core.types import as_arrays

        arrays = as_arrays(grid)
        ids = np.array([-1 if s is None else int(str(s).split("_")[-1])
                        for s in arrays.spot_id])
        sel = np.isin(ids, list(spots)) if spots else np.zeros(len(ids), dtype=bool)
        if not sel.any():
            return None, 0
        res = arrays.resolution[sel]
        return float(np.mean(res)), int(sel.sum())

    res_before, n_before = spot_resolution(grid_before, emphasised)
    res_after, n_after = spot_resolution(grid_after, emphasised)
    if res_before and res_after:
        print(f"emphasis {len(emphasised)} spots carry phi_E > 0: "
              f"mean voxel {res_before * 100:.1f} cm ({n_before:,} voxels) -> "
              f"{res_after * 100:.1f} cm ({n_after:,} voxels)")

    out = {
        "robot": robot.name,
        "tau_c": params.tau_c,
        "spots_audited": report.audited,
        "spots_incomplete": report.incomplete,
        "spots_incomplete_pct": round(100.0 * report.incomplete / max(report.audited, 1), 1),
        "spots_incomplete_str": f"{report.audited} / {report.incomplete}",
        "completed_by_category": report.completed_by_category,
        "top3": ", ".join(f"{k} ({n})" for k, n in top3) or "none",
        "synthetic_points": int(n_synth),
        "synthetic_share_pct": round(100.0 * n_synth / max(len(points.xyz), 1), 2),
        "c_before": round(report.mean_completeness_before, 3),
        "c_after": round(report.mean_completeness_after, 3),
        "c_before_after": f"{report.mean_completeness_before:.2f} $\\rightarrow$ "
                          f"{report.mean_completeness_after:.2f}",
        "holdout_objects": len(deviations),
        "synth_dev": round(dev_cm, 1) if dev_cm is not None else None,
        "synth_dev_median_cm": (round(float(np.median(deviations)) * 100, 1)
                                if deviations else None),
        "synth_dev_by_category_cm": per_category,
        "voxels_before": len(grid_before),
        "voxels_after": len(grid_after),
        "voxel_growth": round(growth, 1),
        "fine_share_before": round(stats_before.fine_share * 100, 2),
        "fine_share_after": round(stats_after.fine_share * 100, 2),
        "emph_share": f"{stats_before.fine_share * 100:.1f} $\\rightarrow$ "
                      f"{stats_after.fine_share * 100:.1f}",
        # The measurement that carries Sec. III-C's claim: resolution *at
        # the emphasised spots*. It was printed and never recorded, which
        # left the paper quoting a number no JSON could back.
        "emph_spots": len(emphasised),
        "emph_voxel_cm_before": (round(res_before * 100, 1) if res_before else None),
        "emph_voxel_cm_after": (round(res_after * 100, 1) if res_after else None),
        "emph_voxel_cm": ((f"{res_before * 100:.1f} $\\rightarrow$ "
                           f"{res_after * 100:.1f}")
                          if res_before and res_after else None),
        "synth_dev_mean_median": (
            f"{dev_cm:.1f} / {float(np.median(deviations)) * 100:.1f}"
            if dev_cm is not None and deviations else None),
        "runtime_s": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
