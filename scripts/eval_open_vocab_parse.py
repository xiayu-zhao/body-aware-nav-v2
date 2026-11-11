"""Instruction understanding, scored against ground truth (no planning).

    PYTHONPATH=. .venv/bin/python scripts/eval_open_vocab_parse.py \
        --backend llama3.1-8b --grounding llm [--levels structured paraphrase ...]

For every task in ``benchmarks/open_vocab/instruction_set.yaml`` and every
wording level, run the agent's parse + grounding (Steps 1-2, the part the
LLM and the grounding mode decide) and score it against the task's truth:

  goal      the grounded goal node has the true category
  hazard    the hazards the instruction names resolve to exactly the true
            category (or to nothing, when the instruction names none)
  margin    the margin applied to that category is the stated one (+-2%);
            with no stated margin, the default is applied
  priority  the parsed priority is the true one
  all       all four

A text shared by several robots is sent once and scored for each task.
Results append to ``results/open_vocab/parse_<backend>_<grounding>.jsonl``
(one line per call, resumable), and a summary prints per level.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SET = REPO / "benchmarks" / "open_vocab" / "instruction_set.yaml"
LEVELS = ("structured", "paraphrase", "free_form", "implicit")
DEFAULT_MARGIN = 3.0
METRICS = ("goal", "hazard", "margin", "priority", "all")


def score(resolved, truth: dict, graph) -> dict:
    from bodynav.llm.grounding import resolve_reference

    task = resolved.task
    sem = lambda n: graph.attributes(n).semantic
    goal = bool(resolved.goal_ids) and sem(resolved.goal_ids[0]) == truth["goal_category"]

    named = set()
    for phrase in task.hazards:
        cat = task.hazard_categories.get(phrase)
        if cat is None:
            ids, _ = resolve_reference(phrase, graph)
            cat = sem(ids[0]) if ids else None
        if cat:
            named.add(cat)
    want = truth["hazard_category"]
    hazard = named == ({want} if want else set())

    if want:
        applied = {round(m, 3) for n, m in resolved.hazard_margins.items() if sem(n) == want}
        target = truth["margin_m"] if truth["margin_m"] is not None else DEFAULT_MARGIN
        margin = bool(applied) and all(abs(m - target) <= 0.02 * target for m in applied)
    else:
        margin = not task.safety_distances
    priority = task.priority.value == truth["priority"]
    return {"goal": goal, "hazard": hazard, "margin": margin, "priority": priority,
            "all": goal and hazard and margin and priority}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--grounding", choices=("lexical", "llm"), required=True)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--repeat", type=int, default=0, help="repeat index (for variation)")
    args = ap.parse_args()

    from bodynav.llm.agent import AgentConfig, ConstrainedAgent
    from bodynav.llm.backends import load_backend
    from bodynav.scene_graph.graph import BodyNavSceneGraph

    tasks = yaml.safe_load(SET.read_text())["tasks"]
    graphs = {s: BodyNavSceneGraph.load(str(REPO / f"results/{s}/scene_graph_nodes.json"))
              for s in {t["site"] for t in tasks}}
    out = REPO / "results" / "open_vocab" / f"parse_{args.backend}_{args.grounding}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[(r["site"], r["level"], r["text"], r["repeat"])] = r

    client = load_backend(args.backend)
    if client.name == "scripted":
        raise SystemExit(f"{args.backend}: no credentials -- refusing a scripted run")
    rows = defaultdict(lambda: defaultdict(list))
    for level in args.levels:
        for t in tasks:
            text = t["structured"] if level == "structured" else t["levels"][level]
            key = (t["site"], level, text, args.repeat)
            if key not in done:
                agent = ConstrainedAgent(AgentConfig(model=args.backend,
                                                     grounding=args.grounding), client)
                t0 = time.time()
                rec = {"site": t["site"], "level": level, "text": text,
                       "repeat": args.repeat, "backend": args.backend,
                       "model": client.model, "grounding": args.grounding}
                try:
                    resolved = agent.pre_ground(text, graphs[t["site"]])
                    task = resolved.task
                    rec.update(ok=True, goal=task.goal, hazards=task.hazards,
                               priority=task.priority.value,
                               safety_distances=task.safety_distances,
                               goal_category=task.goal_category,
                               hazard_categories=task.hazard_categories,
                               goal_ids=resolved.goal_ids,
                               hazard_margins=resolved.hazard_margins)
                except Exception as exc:
                    rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
                tokens = sum(e.payload.get("tokens", 0) for e in agent.trace
                             if e.kind == "parse_attempt")
                rec.update(latency_s=round(time.time() - t0, 2), tokens=tokens)
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[key] = rec
            rec = done[key]
            if rec["ok"]:
                s = score(_rebuild(rec, graphs[t["site"]]), t["truth"], graphs[t["site"]])
            else:
                s = dict.fromkeys(METRICS, False)
            for m in METRICS:
                rows[level][m].append(s[m])

    print(f"{args.backend} / {args.grounding}  (n = tasks per level)")
    print(f"  {'level':12s} " + " ".join(f"{m:>8s}" for m in METRICS) + "      n")
    for level in args.levels:
        r = rows[level]
        print(f"  {level:12s} " + " ".join(f"{100 * sum(r[m]) / len(r[m]):7.1f}%"
                                           for m in METRICS) + f"  {len(r['all']):5d}")
    return 0


def _rebuild(rec: dict, graph):
    """The resolved task, from a stored record (so scoring can be re-run)."""
    from bodynav.core.types import Priority, TaskSpec
    from bodynav.llm.grounding import ResolvedTask

    task = TaskSpec(start=None, goal=rec["goal"], hazards=rec["hazards"],
                    priority=Priority(rec["priority"]),
                    safety_distances=rec["safety_distances"],
                    goal_category=rec["goal_category"],
                    hazard_categories=rec["hazard_categories"])
    return ResolvedTask(task=task, start_id=None, goal_ids=rec["goal_ids"],
                        hazard_ids=list(rec["hazard_margins"]),
                        hazard_margins=rec["hazard_margins"])


if __name__ == "__main__":
    raise SystemExit(main())
