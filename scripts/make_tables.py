#!/usr/bin/env python3
"""Emit LaTeX table bodies from the JSON that the pipeline stages write.

Every number in the paper comes from a file under ``results/labfloor/``
and reaches ``root.tex`` through this script -- no number is typed into the
manuscript by hand.  A value that has not been produced yet renders as

    \\textit{[MISSING: plan.psr]}

which is visible in the PDF and greppable in the source, so an unfilled
result can never be mistaken for a real one.

Usage
-----
    python scripts/make_tables.py                 # write tables/*.tex
    python scripts/make_tables.py --check         # exit 1 if anything missing
    python scripts/make_tables.py --list-missing  # print the unfilled keys
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from bodynav.eval.results import load_results as load_result_json   # maps pre-rename keys

PKG = Path(__file__).resolve().parents[1]           # this repository
RESULTS = PKG / "results" / "labfloor"
# Inside the paper tree the tables live beside the package; standalone they do not.
TABLES = PKG.parent / "tables" if (PKG.parent / "tables").is_dir() else PKG / "tables"

MISSING = object()


def load_results(results_dir: Path) -> dict:
    """Merge every ``*.json`` in the directory under its file stem."""
    out: dict = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            out[path.stem] = load_result_json(path)
        except json.JSONDecodeError as exc:
            print(f"warning: {path.name} is not valid JSON ({exc})", file=sys.stderr)
    # Sibling site directories, namespaced by site (`scene1.batch...`), so a
    # cross-site table can be written without privileging one site's keys.
    for site_dir in sorted(results_dir.parent.iterdir()):
        if not site_dir.is_dir() or site_dir == results_dir:
            continue
        site: dict = {}
        for path in sorted(site_dir.glob("*.json")):
            try:
                site[path.stem] = load_result_json(path)
            except json.JSONDecodeError:
                continue
        if site:
            out[site_dir.name] = site
    return out


def lookup(data: dict, key: str):
    """Resolve a dotted key such as ``plan_nav01.psr`` against the results."""
    node = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return MISSING
    return node


def fmt(value, spec: str | None) -> str:
    if value is MISSING:
        return None
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


class Cell:
    """A table cell: literal text, or a results key with an optional format."""

    def __init__(self, key: str, spec: str | None = None, literal: bool = False,
                 scale: float = 1.0):
        self.key, self.spec, self.literal = key, spec, literal
        self.scale = scale

    def render(self, data: dict, missing: list[str]) -> str:
        if self.literal:
            return self.key
        raw = lookup(data, self.key)
        # Scale rather than a "%" format: LaTeX reads a bare percent sign
        # as the start of a comment and would eat the rest of the row.
        if self.scale != 1.0 and isinstance(raw, (int, float)):
            raw = raw * self.scale
        text = fmt(raw, self.spec)
        if text is None:
            missing.append(self.key)
            return r"\textit{[MISSING: %s]}" % self.key.replace("_", r"\_")
        return text


def lit(text: str) -> Cell:
    return Cell(text, literal=True)


def val(key: str, spec: str | None = None) -> Cell:
    return Cell(key, spec)


#: The comparators that have no instruction channel at all. "Best
#: baseline" has to be the largest of these, computed per site: naming one
#: method and calling it best was wrong wherever another one beat it.
BASELINE_METHODS = ("pure-planner", "subgoal-astar", "costmap-2d",
                    "costmap-3d", "uniform-voxel")


class BestOf(Cell):
    """The maximum of one metric over the baseline methods at one site."""

    def __init__(self, prefix: str, metric: str, spec: str | None = None):
        super().__init__(f"{prefix}.by_method.<best>.{metric}", spec)
        self.prefix, self.metric = prefix, metric

    def render(self, data: dict, missing: list[str]) -> str:
        vals = []
        for m in BASELINE_METHODS:
            raw = lookup(data, f"{self.prefix}.by_method.{m}.{self.metric}")
            if isinstance(raw, (int, float)):
                vals.append(raw)
        if not vals:
            missing.append(self.key)
            return r"\textit{[MISSING: %s]}" % self.key.replace("_", r"\_")
        return fmt(max(vals), self.spec)


def pct(key: str, spec: str = ".1f") -> Cell:
    """A stored fraction rendered as a percentage number (no % sign)."""
    return Cell(key, spec, scale=100.0)


# --- table definitions -------------------------------------------------
# Rows are (label, cell, cell, ...).  Column count is set by the header.
TABLE_SPECS = {
    "labfloor_site": dict(
        caption="Lab-floor site and scene graph.",
        header=["Metric", "Value"],
        rows=[
            ("Points, raw / used ($10^6$)", val("cloud_info.points_raw_m", ".2f"),),
            ("Footprint (m)", val("cloud_info.footprint")),
            ("Storey height (m)", val("scene_graph.storey_m", ".2f")),
            ("Scan sessions", val("cloud_info.sessions")),
            ("Graph nodes (floor / area / spot)", val("scene_graph.nodes_str")),
            ("Graph edges", val("scene_graph.edges")),
            ("Functional areas / free space (m$^2$)", val("scene_graph.areas")),
            ("Hazard-bearing nodes", val("scene_graph.hazard_nodes")),
            ("Regions passable by Husky, blocked for H1",
             val("low_clearance.n_regions")),
        ],
    ),
    "labfloor_abstraction": dict(
        caption="Adaptive abstraction per embodiment on the same as-built scan.",
        header=["Metric", "Husky+UR5", "Unitree H1", "Crazyflie"],
        rows=[
            (label,) + tuple(cell(f"abstraction_{r}.{field}")
                             for r in ("husky_ur5", "unitree_h1", "crazyflie21"))
            for label, field, cell in (
                ("Height band (m)", "band", val),
                ("Band / storey", "band_ratio", lambda k: val(k, ".2f")),
                ("After height slice ($10^6$ pts)", "points_after_slice",
                 lambda k: Cell(k, ".2f", scale=1e-6)),
                ("After $V_t$ ($10^6$ pts)", "points_after_taskfilter",
                 lambda k: Cell(k, ".2f", scale=1e-6)),
                ("Voxels ($10^3$)", "n_voxels",
                 lambda k: Cell(k, ".1f", scale=1e-3)),
                ("Point-to-voxel count reduction (\\%)", "total_reduction_vs_raw",
                 lambda k: pct(k, ".1f")),
                ("Voxels with edge $\\le$0.25\\,m (\\%)", "fine_share",
                 lambda k: val(k, ".1f")),
                ("Kernel radius (m)", "kernel_r", lambda k: val(k, ".2f")),
                ("Map build: slice to voxels (s)", "time_s", lambda k: val(k, ".1f")),
                ("Memory (MB)", "memory_mb", lambda k: val(k, ".1f")),
            )
        ],
    ),
    "labfloor_completion": dict(
        caption="Semantics-guided completion and emphasis on the as-built scan.",
        header=["Metric", "Value"],
        rows=[
            ("Spots audited / below $\\tau_c$", val("completion.spots_incomplete_str")),
            ("Top-3 completed categories", val("completion.top3")),
            ("Synthetic points (\\% of cloud)",
             val("completion.synthetic_share_pct", ".2f")),
            ("Mean completeness, before $\\rightarrow$ after", val("completion.c_before_after")),
            ("Holdout deviation, mean / median (cm)",
             val("completion.synth_dev_mean_median")),
            ("Fine-voxel share, global (\\%)", val("completion.emph_share")),
            ("Mean voxel edge at emphasized spots (cm)",
             val("completion.emph_voxel_cm")),
            ("Total voxel-count growth (\\%)", val("completion.voxel_growth", ".1f")),
        ],
    ),
    "labfloor_planning": dict(
        caption="Value-guided A* on the as-built floor, per embodiment.",
        header=["Metric", "Husky+UR5", "Unitree H1"],
        rows=[
            ("Reachable area (m$^2$)", val("plan_husky_ur5.reachable_m2", ".0f"),
             val("plan_unitree_h1.reachable_m2", ".0f")),
            ("Lateral clearance required (m)", lit("0.85"), lit("0.60")),
            ("Vertical clearance required (m)", lit("1.25"), lit("2.00")),
            ("Hazards in band / above it", val("plan_husky_ur5.n_hazards"),
             val("plan_unitree_h1.n_hazards")),
            ("Path length (m)", val("plan_husky_ur5.final.length_m", ".2f"),
             val("plan_unitree_h1.final.length_m", ".2f")),
            ("Normalized length", val("plan_husky_ur5.final.normalized_length", ".3f"),
             val("plan_unitree_h1.final.normalized_length", ".3f")),
            ("Min / mean clearance (m)", val("plan_husky_ur5.final.min_clearance_m", ".2f"),
             val("plan_unitree_h1.final.min_clearance_m", ".2f")),
            ("Waypoints, raw $\\rightarrow$ final", val("plan_husky_ur5.raw.waypoints"),
             val("plan_unitree_h1.raw.waypoints")),
            ("Expansions", val("plan_husky_ur5.expansions"),
             val("plan_unitree_h1.expansions")),
            ("Planning time (s)", val("plan_husky_ur5.t_plan_s", ".3f"),
             val("plan_unitree_h1.t_plan_s", ".3f")),
        ],
    ),
    "labfloor_agent": dict(
        caption="Stage 3 on one lab-floor instruction, both backends.",
        header=["Metric", "LLM-Large", "LLM-Mini"],
        rows=[
            ("Model", val("plan_husky_ur5_llm.agent.model"),
             val("plan_husky_ur5_mini.agent.model")),
            ("Schema / regeneration retries", val("plan_husky_ur5_llm.agent.retries"),
             val("plan_husky_ur5_mini.agent.retries")),
            ("Tokens per instruction", val("plan_husky_ur5_llm.agent.llm_tokens"),
             val("plan_husky_ur5_mini.agent.llm_tokens")),
            ("LLM latency (s)", val("plan_husky_ur5_llm.agent.llm_latency_s", ".1f"),
             val("plan_husky_ur5_mini.agent.llm_latency_s", ".1f")),
            ("Planning time (s)", val("plan_husky_ur5_llm.t_plan_s", ".3f"),
             val("plan_husky_ur5_mini.t_plan_s", ".3f")),
            ("Path length (m)", val("plan_husky_ur5_llm.final.length_m", ".2f"),
             val("plan_husky_ur5_mini.final.length_m", ".2f")),
            ("Normalized length", val("plan_husky_ur5_llm.final.normalized_length", ".3f"),
             val("plan_husky_ur5_mini.final.normalized_length", ".3f")),
            ("Min clearance (m)", val("plan_husky_ur5_llm.final.min_clearance_m", ".3f"),
             val("plan_husky_ur5_mini.final.min_clearance_m", ".3f")),
        ],
    ),
    "labfloor_stage": dict(
        caption="Simulation stage built from our method's own map.",
        header=["Metric", "Value"],
        rows=[
            ("Obstacle cells (0.1 m lattice)", val("colliders_husky_ur5.cells")),
            ("Merged collision boxes", val("colliders_husky_ur5.boxes")),
            ("Prim reduction", val("colliders_husky_ur5.reduction", ".1%")),
            ("Decomposition exact", val("colliders_husky_ur5.complete")),
            ("Floor slab boxes", val("colliders_husky_ur5.floor_boxes")),
            ("Collider volume (m$^3$)", val("colliders_husky_ur5.volume_m3", ".0f")),
            ("Drop probe rest height (m)", val("physics_check.drop.rest_z", ".3f")),
            ("Raycast into a wall (m)", val("physics_check.raycast.distance_m", ".2f")),
        ],
    ),
    "labfloor_baselines": dict(
        caption="Tasks sampled inside each platform's reachable component, "
                "3 platforms $\\times$ 3 seeds, pooled. Every method runs "
                "through the "
                "same planner and the same feasibility gate, and every "
                "trajectory is scored against a 2.5\\,cm reference model of the "
                "scan rather than against the map its own method planned on. "
                "The last three columns are the instruction: tasks come in "
                "pairs sharing a start and goal, asked once as ``take the "
                "shortest route'' and once as ``stay at least $N$ metres clear "
                "of $X$''. \\emph{Response} is the resulting gain in distance "
                "from the named hazard and \\emph{Same} the share of pairs "
                "answered with equal stored path lengths -- 100\\,\\% for every method "
                "that has no instruction input.",
        header=["Method", "PSR (\\%)", "SCR (\\%)", "Norm. len.",
                "\\shortstack{Hazard\\\\dist. (m)}", "\\shortstack{Margin\\\\met (\\%)}",
                "\\shortstack{Response\\\\(m)}", "\\shortstack{Same\\\\length (\\%)}"],
        rows=[
            (label,
             val(f"batch_all.by_method.{k}.psr", ".0f"),
             val(f"batch_all.by_method.{k}.scr", ".0f"),
             val(f"batch_all.by_method.{k}.normalized_length_mean", ".3f"),
             val(f"batch_all.by_method.{k}.hazard_distance_mean_m", ".2f"),
             val(f"batch_all.by_method.{k}.hazard_margin_compliance", ".0f"),
             val(f"batch_all.priority_response.{k}.hazard_gain_mean_m", ".2f"),
             val(f"batch_all.priority_response.{k}.identical_paths_pct", ".0f"))
            for label, k in (
                ("Ours", "bodynav"),
                ("w/o task map", "bodynav-fixed-grid"),
                ("Instruction-blind", "bodynav-no-agent"),
                ("Pure-Planner", "pure-planner"),
                ("Subgoal A*", "subgoal-astar"),
                ("Sensor-slice costmap", "costmap-2d"),
                ("Generic 3D grid", "costmap-3d"),
                ("Uniform voxels", "uniform-voxel"),
            )
        ],
    ),
    "sites": dict(
        caption="One pipeline, three sites. Scenes~1--2 are the simulated "
                "stages, the laboratory floor the as-built scan. Eq.~2's "
                "reduction tracks how much of the storey a platform's band "
                "occupies, which is why no single figure can be quoted for "
                "it. The last three rows are the instruction: our method meets the "
                "stated hazard margin on every site, and the strongest "
                "baseline on each meets it only where the direct route "
                "happened to satisfy it already.",
        header=["", "Scene 1", "Scene 2", "Lab floor"],
        rows=[
            ("Footprint (m)", val("scene1.cloud_info.footprint"),
             val("scene2.cloud_info.footprint"), val("cloud_info.footprint")),
            ("Storey (m)", val("scene1.scene_graph.storey_m", ".2f"),
             val("scene2.scene_graph.storey_m", ".2f"),
             val("scene_graph.storey_m", ".2f")),
            ("Free floor (m$^2$)",
             val("scene1.scene_graph.free_area_m2", ".0f"),
             val("scene2.scene_graph.free_area_m2", ".0f"),
             val("scene_graph.free_area_m2", ".0f")),
            ("Height-slice reduction, Husky (\\%)",
             pct("scene1.abstraction_husky_ur5.slice_reduction"),
             pct("scene2.abstraction_husky_ur5.slice_reduction"),
             pct("abstraction_husky_ur5.slice_reduction")),
            ("Height-slice reduction, H1 (\\%)",
             pct("scene1.abstraction_unitree_h1.slice_reduction"),
             pct("scene2.abstraction_unitree_h1.slice_reduction"),
             pct("abstraction_unitree_h1.slice_reduction")),
            ("Height-slice reduction, Crazyflie (\\%)",
             pct("scene1.abstraction_crazyflie21.slice_reduction"),
             pct("scene2.abstraction_crazyflie21.slice_reduction"),
             pct("abstraction_crazyflie21.slice_reduction")),
            ("Our safety (\\%)",
             val("scene1.batch.by_method.bodynav.scr", ".1f"),
             val("scene2.batch.by_method.bodynav.scr", ".1f"),
             val("batch.by_method.bodynav.scr", ".1f")),
            ("Our margin met (\\%)",
             val("scene1.batch.by_method.bodynav.hazard_margin_compliance", ".1f"),
             val("scene2.batch.by_method.bodynav.hazard_margin_compliance", ".1f"),
             val("batch.by_method.bodynav.hazard_margin_compliance", ".1f")),
            ("Best external baseline margin met (\\%)",
             BestOf("scene1.batch", "hazard_margin_compliance", ".1f"),
             BestOf("scene2.batch", "hazard_margin_compliance", ".1f"),
             BestOf("batch", "hazard_margin_compliance", ".1f")),
            ("Our instruction response (m)",
             val("scene1.batch.priority_response.bodynav.hazard_gain_mean_m", ".2f"),
             val("scene2.batch.priority_response.bodynav.hazard_gain_mean_m", ".2f"),
             val("batch.priority_response.bodynav.hazard_gain_mean_m", ".2f")),
            ("\\emph{Aerial:} our margin met (\\%)",
             val("scene1.batch_uav.by_method.bodynav.hazard_margin_compliance", ".1f"),
             val("scene2.batch_uav.by_method.bodynav.hazard_margin_compliance", ".1f"),
             val("batch_uav.by_method.bodynav.hazard_margin_compliance", ".1f")),
            ("\\emph{Aerial:} our response (m)",
             val("scene1.batch_uav.priority_response.bodynav.hazard_gain_mean_m", ".2f"),
             val("scene2.batch_uav.priority_response.bodynav.hazard_gain_mean_m", ".2f"),
             val("batch_uav.priority_response.bodynav.hazard_gain_mean_m", ".2f")),
        ],
    ),
    "labfloor_batch": dict(
        caption="Our method per platform over the same instruction set.",
        header=["Metric", "Husky+UR5", "Unitree H1"],
        rows=[
            ("Runs", val("batch.by_robot_method.husky_ur5/bodynav.runs"),
             val("batch.by_robot_method.unitree_h1/bodynav.runs")),
            ("PSR (\\%)", val("batch.by_robot_method.husky_ur5/bodynav.psr", ".1f"),
             val("batch.by_robot_method.unitree_h1/bodynav.psr", ".1f")),
            ("SCR (\\%)", val("batch.by_robot_method.husky_ur5/bodynav.scr", ".1f"),
             val("batch.by_robot_method.unitree_h1/bodynav.scr", ".1f")),
            ("Normalized length, mean $\\pm$ sd",
             val("batch.by_robot_method.husky_ur5/bodynav.normalized_length_mean", ".3f"),
             val("batch.by_robot_method.unitree_h1/bodynav.normalized_length_mean", ".3f")),
            ("Min clearance, mean / worst (m)",
             val("batch.by_robot_method.husky_ur5/bodynav.min_clearance_mean", ".2f"),
             val("batch.by_robot_method.unitree_h1/bodynav.min_clearance_mean", ".2f")),
            ("Planning time (s)",
             val("batch.by_robot_method.husky_ur5/bodynav.t_plan_mean_s", ".3f"),
             val("batch.by_robot_method.unitree_h1/bodynav.t_plan_mean_s", ".3f")),
            ("LLM tokens / instruction",
             val("batch.by_robot_method.husky_ur5/bodynav.llm_tokens_mean", ".0f"),
             val("batch.by_robot_method.unitree_h1/bodynav.llm_tokens_mean", ".0f")),
            ("LLM latency (s)",
             val("batch.by_robot_method.husky_ur5/bodynav.llm_latency_mean_s", ".1f"),
             val("batch.by_robot_method.unitree_h1/bodynav.llm_latency_mean_s", ".1f")),
            ("Schema / regeneration retries",
             val("batch.by_robot_method.husky_ur5/bodynav.llm_retries_total"),
             val("batch.by_robot_method.unitree_h1/bodynav.llm_retries_total")),
            ("Reachable floor, our map (m$^2$)",
             val("batch.reachable_area.husky_ur5.bodynav_map_m2", ".1f"),
             val("batch.reachable_area.unitree_h1.bodynav_map_m2", ".1f")),
            ("Reachable floor, 2.5\\,cm reference (m$^2$)",
             val("batch.reachable_area.husky_ur5.reference_m2", ".1f"),
             val("batch.reachable_area.unitree_h1.reference_m2", ".1f")),
            ("Map size, mean per task (MB)",
             val("batch.by_robot_method.husky_ur5/bodynav.map_mb_mean", ".0f"),
             val("batch.by_robot_method.unitree_h1/bodynav.map_mb_mean", ".0f")),
            ("Hazard margin met (\\%)",
             val("batch.by_robot_method.husky_ur5/bodynav.hazard_margin_compliance", ".0f"),
             val("batch.by_robot_method.unitree_h1/bodynav.hazard_margin_compliance", ".0f")),
            ("Map size, ours / reference ($10^3$ voxels)",
             val("batch.reachable_area.husky_ur5.bodynav_voxels", ".0f"),
             val("batch.reachable_area.unitree_h1.bodynav_voxels", ".0f")),
        ],
    ),
    "labfloor_execution": dict(
        caption="Execution in simulation on as-built geometry. The two "
                "platforms are measured differently and the mode says which.",
        header=["Metric", "Husky+UR5", "Unitree H1"],
        rows=[
            ("Mode", val("exec_husky_ur5.mode"), val("exec_unitree_h1.mode")),
            ("Reached goal", val("exec_husky_ur5.reached_goal"),
             val("exec_unitree_h1.reached_goal")),
            ("Planned length (m)", val("exec_husky_ur5.planned_length_m", ".2f"),
             val("exec_unitree_h1.planned_length_m", ".2f")),
            ("Executed length (m)", val("exec_husky_ur5.executed_length_m", ".2f"),
             val("exec_unitree_h1.executed_length_m", ".2f")),
            ("Length ratio", val("exec_husky_ur5.length_ratio", ".3f"),
             val("exec_unitree_h1.length_ratio", ".3f")),
            ("Tracking RMSE (m)", val("exec_husky_ur5.tracking_rmse_m", ".3f"),
             val("exec_unitree_h1.tracking_rmse_m", ".3f")),
            ("Min / mean clearance (m)", val("exec_husky_ur5.min_clearance_m", ".3f"),
             val("exec_unitree_h1.min_clearance_m", ".3f")),
            ("Contacts with collision geometry", val("exec_husky_ur5.contacts"),
             val("exec_unitree_h1.contacts")),
            ("Logged poses", val("exec_husky_ur5.logged_poses"),
             val("exec_unitree_h1.logged_poses")),
        ],
    ),
}


def render_table(name: str, spec: dict, data: dict) -> tuple[str, list[str]]:
    missing: list[str] = []
    ncol = len(spec["header"])
    lines = [
        f"% generated by scripts/make_tables.py -- do not edit by hand",
        r"\begin{tabular}{l" + "c" * (ncol - 1) + "}",
        r"\toprule",
        " & ".join(spec["header"]) + r" \\",
        r"\midrule",
    ]
    for row in spec["rows"]:
        label, cells = row[0], row[1:]
        rendered = [c.render(data, missing) for c in cells]
        lines.append(f"{label} & " + " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=TABLES)
    ap.add_argument("--check", action="store_true", help="exit 1 if any value is missing")
    ap.add_argument("--list-missing", action="store_true")
    args = ap.parse_args()

    data = load_results(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    all_missing: list[str] = []
    for name, spec in TABLE_SPECS.items():
        body, missing = render_table(name, spec, data)
        (args.out / f"{name}.tex").write_text(body)
        all_missing += missing
        state = "complete" if not missing else f"{len(missing)} missing"
        print(f"tables/{name}.tex  [{state}]")

    if args.list_missing and all_missing:
        print("\nunfilled keys:")
        for key in sorted(set(all_missing)):
            print("  ", key)
    if args.check and all_missing:
        print(f"\n{len(set(all_missing))} unfilled keys", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
