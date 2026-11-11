"""Build the open-vocabulary instruction set from the reviewed rewrites.

    PYTHONPATH=. .venv/bin/python scripts/build_open_vocab_set.py

Reads ``benchmarks/open_vocab/rewrites.yaml`` (one entry per distinct
template instruction) and the sampled tasks in
``results/<site>/batch_all.json``, and writes
``benchmarks/open_vocab/instruction_set.yaml``: one entry per (site,
robot, task) with its ground truth and the three rewrites, in the form
``run_batch.py --instruction-set`` reads.

It refuses to write the set if a check fails:
  * every task has a rewrite, and every rewrite is used;
  * every rewrite states the task's margin -- the same number of metres,
    whatever the unit -- and a task with no stated margin gets none;
  * no rewrite names a thing with a word from the lexical synonym table
    (that would let the lexical grounding pass without understanding);
  * only the quadrotor is told to fly.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "benchmarks" / "open_vocab" / "rewrites.yaml"
OUT = REPO / "benchmarks" / "open_vocab" / "instruction_set.yaml"
SITES = ("scene1", "scene2", "labfloor")
LEVELS = ("paraphrase", "free_form", "implicit")

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def stated_metres(text: str) -> list[float]:
    """Every distance the text states, converted to metres."""
    t = text.lower()
    out = []
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm|m|millimetres?|centimetres?|"
                                r"metres?|meters?)\b", t):
        scale = 1e-3 if unit.startswith(("mm", "milli")) else 1e-2 if unit.startswith(("cm", "centi")) else 1.0
        out.append(float(num) * scale)
    for word, half in re.findall(r"\b(one|two|three|four|five|six)( and a half)?\s+metres?\b", t):
        out.append(WORD_NUM[word] + (0.5 if half else 0.0))
    if re.search(r"\ba metre and a half\b", t):
        out.append(1.5)
    if re.search(r"\ba metre\b(?! and a half)|\ba metre's\b", t):
        out.append(1.0)
    return out


def truth_of(task: dict) -> dict:
    """What the instruction asks for, as the rewrite must preserve it."""
    text = task["text"]
    paired = bool(task.get("pair_id"))
    hazard = margin = None
    if paired and task["priority"] == "safety_first":
        hazard, margin = task["hazard_category"], task["hazard_margin_m"]
    elif not paired:  # probes: the hazard is named in the text only
        for phrase, cat in (("stored material", "stored_material"),
                            ("ductwork", "duct"), ("traffic cones", "cone")):
            if phrase in text:
                hazard = cat
        nums = stated_metres(text)
        margin = nums[0] if nums else None
    return {"priority": task["priority"], "goal_category": task["goal_category"],
            "hazard_category": hazard, "margin_m": margin}


def lexical_words() -> set[str]:
    sys.path.insert(0, str(REPO))
    from bodynav.llm.grounding import SITE_PROMPTS
    # words the lexical matcher maps, minus generic place words it treats as
    # "area" (aisle, room...) and words that only make sense as structure
    generic = {"corridor", "hallway", "room", "lab", "laboratory", "zone", "space",
               "floor", "ground", "ceiling", "wall", "walls"}
    return set(SITE_PROMPTS) - generic


def main() -> int:
    entries = yaml.safe_load(SRC.read_text())["entries"]
    by_key = {(e["site"], e["task_id"], e["structured"]): e for e in entries}
    used, problems, out = set(), [], []
    banned = lexical_words()

    for site in SITES:
        tasks = json.loads((REPO / f"results/{site}/batch_all.json").read_text())["tasks"]
        for t in tasks.values():
            key = (site, t["id"], t["text"])
            e = by_key.get(key)
            if e is None:
                problems.append(f"no rewrite for {site}/{t['robot']}/{t['id']}: {t['text']!r}")
                continue
            used.add(key)
            truth = truth_of(t)
            for level in LEVELS:
                text = e[level]
                got = stated_metres(text)
                want = truth["margin_m"]
                if want is None and got:
                    problems.append(f"{site}/{t['id']} {level}: states {got} m but the task has no margin")
                if want is not None and not any(abs(g - want) < 1e-6 for g in got):
                    problems.append(f"{site}/{t['id']} {level}: margin {want} m not stated ({got})")
                words = set(re.findall(r"[a-z]+", text.lower()))
                if words & banned:
                    problems.append(f"{site}/{t['id']} {level}: lexical-table word(s) {sorted(words & banned)}")
                if re.search(r"\bfly\b", text.lower()) and t["robot"] != "crazyflie21":
                    problems.append(f"{site}/{t['robot']}/{t['id']} {level}: 'fly' for a ground robot")
            out.append({"site": site, "robot": t["robot"], "task_id": t["id"],
                        "structured": t["text"], "truth": truth,
                        "levels": {lvl: e[lvl] for lvl in LEVELS}})

    for key in set(by_key) - used:
        problems.append(f"rewrite never used: {key}")
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s); {OUT.name} not written")
        return 1
    OUT.write_text(
        "# GENERATED by scripts/build_open_vocab_set.py from rewrites.yaml and\n"
        "# results/<site>/batch_all.json -- edit rewrites.yaml, then rebuild.\n"
        + yaml.safe_dump({"tasks": out}, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {OUT.relative_to(REPO)}: {len(out)} tasks x {len(LEVELS)} levels "
          f"from {len(entries)} rewrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
