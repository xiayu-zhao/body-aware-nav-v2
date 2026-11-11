"""Summarise the open-vocabulary study: understanding, planning, cost.

    PYTHONPATH=. .venv/bin/python scripts/analyze_open_vocab.py

Reads the planning sweep (``results/open_vocab/plan/<backend>_<grounding>/``)
and the understanding scores (``results/open_vocab/parse_*.jsonl``), and
writes ``results/open_vocab/summary.json`` plus a printed report.

Planning metrics use the definitions of ``run_batch.summarise`` (the template
benchmark's): PSR and SCR over feasible tasks; stated-margin compliance over
runs whose instruction states a margin; agent success; grounding accuracy.
One composite is added for this study:

  IFS  instruction-following success -- planned, safe (SCR), right goal
       category, and the stated margin kept where one is stated; over
       feasible tasks.

95% intervals resample *tasks* (a task's repeats move together), 1000
draws, fixed seed.
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "results" / "open_vocab" / "plan"
OUT = REPO / "results" / "open_vocab" / "summary.json"
LEVELS = ("structured", "paraphrase", "free_form", "implicit")
CONFIGS = [  # (backend, grounding, label, paid)
    ("claude-sonnet", "llm", "Claude Sonnet 5", True),
    ("llm-large", "llm", "GPT-5.2", True),
    ("qwen3.5-9b", "llm", "Qwen3.5-9B (local)", False),
    ("gemma3-12b", "llm", "Gemma-3-12B (local)", False),
    ("llm-large", "lexical", "GPT-5.2, lexical grounding", True),
]
METRICS = ("ifs", "psr", "scr", "margin", "goal", "agent")
#: Understanding-only scores cover every evaluated model. Llama-3.1-8B is
#: in this table but not in planning: its value-map code never validated
#: (0/396 runs), so its planning would duplicate Gemma's fallback case.
UNDERSTANDING = [
    ("claude-sonnet", "llm", "Claude Sonnet 5"), ("llm-large", "llm", "GPT-5.2"),
    ("qwen3.5-9b", "llm", "Qwen3.5-9B (local)"), ("llama3.1-8b", "llm", "Llama-3.1-8B (local)"),
    ("gemma3-12b", "llm", "Gemma-3-12B (local)"),
    ("claude-sonnet", "lexical", "Claude Sonnet 5, lexical"), ("llm-large", "lexical", "GPT-5.2, lexical"),
    ("qwen3.5-9b", "lexical", "Qwen3.5-9B, lexical"), ("llama3.1-8b", "lexical", "Llama-3.1-8B, lexical"),
    ("gemma3-12b", "lexical", "Gemma-3-12B, lexical"),
]
N_BOOT = 1000


def per_run(r: dict) -> dict:
    """The study's binary outcomes for one run (None = not applicable)."""
    feas = r.get("feasible_by_construction", True)
    ok = bool(r.get("success"))
    safe = ok and r.get("clearance_ok") and r.get("hazard_ok") and r.get("vertical_ok", True)
    stated = (r.get("hazard_task") or {}).get("compliant")
    margin_ok = None if stated is None else bool(stated)
    goal = bool(r.get("goal_match"))
    return {
        "feasible": feas,
        "psr": ok if feas else None,
        "scr": bool(safe) if feas else None,
        "margin": margin_ok if feas else None,
        "goal": goal,
        "agent": bool(r.get("agent_ok")) if "agent_ok" in r else None,
        "ifs": (bool(safe) and goal and margin_ok is not False) if feas else None,
        "tokens": (r.get("agent") or {}).get("llm_tokens"),
        "latency": (r.get("agent") or {}).get("llm_latency_s"),
    }


def load(backend: str, grounding: str) -> dict:
    """{level: {task_key: [per_run, ...]}} over all repeats."""
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(PLAN / f"{backend}_{grounding}" / "*.json"))):
        if "budget" in f:
            continue
        level = Path(f).name.split("_", 1)[0] if not Path(f).name.startswith("free_form") else "free_form"
        site = Path(f).stem.replace(f"{level}_", "", 1).split("_", 1)[0]
        for r in json.loads(Path(f).read_text())["runs"]:
            out[level][(site, r["robot"], r["task_id"])].append(per_run(r))
    return out


def rate(tasks: dict, metric: str, keys=None) -> float | None:
    keys = list(tasks) if keys is None else keys
    vals = [v[metric] for k in keys for v in tasks[k] if v[metric] is not None]
    return 100.0 * sum(vals) / len(vals) if vals else None


def boot_ci(tasks: dict, metric: str, rng: random.Random):
    keys = [k for k in tasks if any(v[metric] is not None for v in tasks[k])]
    if not keys:
        return None, None
    draws = []
    for _ in range(N_BOOT):
        sample = [rng.choice(keys) for _ in keys]
        x = rate(tasks, metric, sample)
        if x is not None:
            draws.append(x)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def understanding(backend: str, grounding: str) -> dict:
    """Mean 'all four correct' per level from the parse scorer's JSONL."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import yaml
    from bodynav.scene_graph.graph import BodyNavSceneGraph
    from eval_open_vocab_parse import _rebuild, score

    path = REPO / "results" / "open_vocab" / f"parse_{backend}_{grounding}.jsonl"
    if not path.exists():
        return {}
    tasks = yaml.safe_load((REPO / "benchmarks/open_vocab/instruction_set.yaml").read_text())["tasks"]
    graphs = {s: BodyNavSceneGraph.load(str(REPO / f"results/{s}/scene_graph_nodes.json"))
              for s in ("scene1", "scene2", "labfloor")}
    recs = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        recs[(r["site"], r["level"], r["text"], r["repeat"])] = r
    repeats = sorted({k[3] for k in recs})
    out = {}
    for lv in LEVELS:
        per_rep = []
        for rep in repeats:
            ok = n = 0
            for t in tasks:
                text = t["structured"] if lv == "structured" else t["levels"][lv]
                r = recs.get((t["site"], lv, text, rep))
                if r is None:
                    continue
                n += 1
                ok += bool(r["ok"] and score(_rebuild(r, graphs[t["site"]]), t["truth"],
                                             graphs[t["site"]])["all"])
            if n:
                per_rep.append(100.0 * ok / n)
        if per_rep:
            out[lv] = {"mean": round(sum(per_rep) / len(per_rep), 1),
                       "min": round(min(per_rep), 1), "max": round(max(per_rep), 1),
                       "repeats": len(per_rep)}
    return out


def main() -> int:
    rng = random.Random(0)
    summary = {"metrics": {
        "ifs": "planned + safe + right goal category + stated margin kept (feasible tasks)",
        "psr": "path found (feasible tasks)", "scr": "path found and safe (feasible tasks)",
        "margin": "stated margin kept on the path (tasks that state one)",
        "goal": "grounded goal has the true category", "agent": "generated value-map code validated",
        "understanding": "parse+grounding all four correct (eval_open_vocab_parse.py)"},
        "configs": []}
    for backend, grounding, label, paid in CONFIGS:
        data = load(backend, grounding)
        if not data:
            continue
        und = understanding(backend, grounding)
        cfg = {"backend": backend, "grounding": grounding, "label": label, "paid": paid,
               "levels": {}}
        for lv in LEVELS:
            tasks = data.get(lv, {})
            row = {"tasks": len(tasks), "runs": sum(len(v) for v in tasks.values())}
            for m in METRICS:
                x = rate(tasks, m)
                lo, hi = boot_ci(tasks, m, rng)
                row[m] = None if x is None else {"rate": round(x, 1), "ci95": [
                    None if lo is None else round(lo, 1), None if hi is None else round(hi, 1)]}
            toks = [v["tokens"] for vs in tasks.values() for v in vs if v["tokens"]]
            lats = [v["latency"] for vs in tasks.values() for v in vs if v["latency"]]
            row["llm_tokens_mean"] = round(sum(toks) / len(toks)) if toks else None
            row["llm_latency_mean_s"] = round(sum(lats) / len(lats), 2) if lats else None
            row["understanding"] = und.get(lv)
            cfg["levels"][lv] = row
        summary["configs"].append(cfg)

    summary["understanding"] = []
    for backend, grounding, label in UNDERSTANDING:
        u = understanding(backend, grounding)
        if u:
            summary["understanding"].append({"backend": backend, "grounding": grounding,
                                             "label": label, "levels": u})
    OUT.write_text(json.dumps(summary, indent=1))
    print("\nUnderstanding (all four correct, mean over repeats; structured/para/free/implicit)")
    for u in summary["understanding"]:
        print(f"  {u['label']:28s} " + "  ".join(
            f"{u['levels'][lv]['mean']:5.1f}" if lv in u["levels"] else "  -  " for lv in LEVELS))
    fmt = lambda c: "  -  " if c is None else f"{c['rate']:5.1f}"
    for cfg in summary["configs"]:
        print(f"\n{cfg['label']}  [{cfg['backend']}/{cfg['grounding']}]")
        print(f"  {'level':11s}  under   IFS   PSR   SCR  margin goal  agent  tokens latency  runs")
        for lv in LEVELS:
            r = cfg["levels"][lv]
            u = r["understanding"]
            print(f"  {lv:11s}  {'  -  ' if not u else format(u['mean'], '5.1f')} "
                  + " ".join(fmt(r[m]) for m in METRICS)
                  + f"  {r['llm_tokens_mean'] or '-':>6} {r['llm_latency_mean_s'] or '-':>6}  {r['runs']:5d}")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
