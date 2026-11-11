"""Instruction schema and parsing (Sec. III-D.1, Sec. IV "Tasks").

Instructions arrive in a lightweight structured schema
(target / constraint / preference fields with free natural language
*within* fields) that bounds LLM output space — the enabler of reliable
deployment on smaller models (Sec. V-C). Semi-structured and free-form
inputs are also accepted; the schema-reminder retry safeguard fires on
noncompliant parses.

The parse target is T = {S, G, H, C, pi} (start, goal, hazards,
constraints, priority), including explicit modifiers such as an
instruction-provided safety distance. Underspecified parameters use
domain defaults (area centers as endpoints; 3 m hazard / 0.5 m
static-obstacle margins).
"""

from __future__ import annotations

from bodynav.core.types import Priority, TaskSpec

#: JSON schema the parse must satisfy (validated before grounding).
TASK_JSON_SCHEMA: dict = {
    "type": "object",
    "required": ["goal", "priority"],
    "properties": {
        "start": {"type": ["string", "null"]},
        "goal": {"type": "string"},
        "hazards": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "object"},
        "priority": {"enum": [p.value for p in Priority]},
        "safety_distances": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

#: Domain defaults for underspecified parameters (Sec. IV).
DEFAULTS = {
    "hazard_margin_m": 3.0,
    "static_obstacle_margin_m": 0.5,
}


#: The prompt is part of the method, not an implementation detail: the
#: schema is *shown* to the model, which is what bounds the output space
#: enough for a small model to comply (Sec. V-C).
PARSE_SYSTEM_PROMPT = """You translate a navigation instruction into JSON.

Return ONLY a JSON object, no prose and no code fences, with these keys:
  start              string or null -- where the robot begins, if stated
  goal               string -- REQUIRED, what to reach
  hazards            array of strings -- things to keep away from
  constraints        object -- other stated requirements
  priority           one of: safety_first, efficiency_first, area_preference
  safety_distances   object mapping a hazard name to metres, when the
                     instruction gives an explicit distance

Rules:
- Use the instruction's own words for entity names. Do not invent
  entities that are not mentioned.
- Omit what the instruction does not say rather than guessing; defaults
  are applied downstream.
- "safely", "carefully", "avoid" imply safety_first. "quickly",
  "shortest", "directly" imply efficiency_first. A stated preference for
  a region implies area_preference.
- A number with a unit IS an explicit distance. Put it in
  safety_distances, keyed by the hazard exactly as you named it in
  hazards. Missing it discards the one quantity the person specified,
  and a default is used in its place.

Example:
  "Go to the loading bay, staying at least 1.5 metres clear of the
   stored material."
  {"start": null, "goal": "the loading bay",
   "hazards": ["the stored material"], "constraints": {},
   "priority": "safety_first",
   "safety_distances": {"the stored material": 1.5}}
"""

#: One-line definitions of the segmenter's labels, shown to the model in
#: open-vocabulary grounding. They define each label; they are not a
#: synonym list (the lexical table in grounding.py is that).
CATEGORY_DEFINITIONS: dict[str, str] = {
    "stored_material": "goods or building materials placed in storage",
    "equipment": "machines and other equipment units",
    "furniture": "tables, desks, seating and cabinets",
    "clutter": "loose items and obstructions left on the floor",
    "cone": "traffic cones and similar temporary barriers",
    "duct": "overhead ductwork",
    "pipe": "exposed pipework",
}

#: Open-vocabulary parse (journal): the template prompt plus the scene's own
#: category inventory, so the model -- not a synonym table -- decides what
#: a phrase like "the pallets" or "where spare stock is kept" refers to.
OPEN_PARSE_ADDENDUM = """
The scene contains only these object categories:
{inventory}

Also return:
  goal_category      the category above that the goal refers to, or null
                     if the goal is not one of them
  hazard_categories  object mapping each name in hazards to the category
                     above it refers to (omit a hazard that matches none)

More rules:
- Choose categories by meaning, not by spelling: the person may use any
  words for these things.
- Give safety_distances in metres; convert other units (1 ft = 0.3048 m).
"""


def open_parse_prompt(categories) -> str:
    """The open-vocabulary parse prompt for a scene with ``categories``."""
    lines = [f"  {c:18s} {CATEGORY_DEFINITIONS.get(c, c.replace('_', ' '))}"
             for c in sorted(categories)]
    return PARSE_SYSTEM_PROMPT + OPEN_PARSE_ADDENDUM.format(inventory="\n".join(lines))


SCHEMA_REMINDER = """Your previous reply did not satisfy the schema:
{error}

Return ONLY the JSON object described above. No prose, no code fences."""


def _type_ok(value, spec) -> bool:
    """Minimal JSON-schema type check for the subset used here."""
    if isinstance(spec, list):
        return any(_type_ok(value, s) for s in spec)
    return {
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }[spec](value)


def check_schema(payload: dict, schema: dict = TASK_JSON_SCHEMA) -> None:
    """Validate against ``schema``; raise SchemaViolation with a message
    the retry can quote back to the model."""
    from bodynav.llm.validation import SchemaViolation

    if not isinstance(payload, dict):
        raise SchemaViolation(f"expected an object, got {type(payload).__name__}")
    for key in schema.get("required", []):
        if key not in payload:
            raise SchemaViolation(f"missing required key {key!r}")
    props = schema.get("properties", {})
    if not schema.get("additionalProperties", True):
        extra = sorted(set(payload) - set(props))
        if extra:
            raise SchemaViolation(f"unexpected keys: {extra}")
    for key, value in payload.items():
        spec = props.get(key)
        if spec is None:
            continue
        if "enum" in spec and value not in spec["enum"]:
            raise SchemaViolation(
                f"{key!r} must be one of {spec['enum']}, got {value!r}")
        if "type" in spec and not _type_ok(value, spec["type"]):
            raise SchemaViolation(
                f"{key!r} must be {spec['type']}, got {type(value).__name__}")
        if spec.get("type") == "array" and "items" in spec:
            for i, item in enumerate(value):
                if not _type_ok(item, spec["items"]["type"]):
                    raise SchemaViolation(
                        f"{key}[{i}] must be {spec['items']['type']}")
        if "additionalProperties" in spec and isinstance(value, dict):
            want = spec["additionalProperties"].get("type")
            for k, v in value.items():
                if want and not _type_ok(v, want):
                    raise SchemaViolation(f"{key}[{k!r}] must be {want}")


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply.

    Models wrap JSON in prose or fences even when told not to; that is a
    formatting habit, not a schema violation, so it is stripped here and
    the schema check judges the content.
    """
    import json
    import re

    from bodynav.llm.validation import SchemaViolation

    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise SchemaViolation("no JSON object in the reply")
        candidate = candidate[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"invalid JSON: {exc}") from exc


def validate_parse(payload: dict, raw_instruction: str = "",
                   categories=None) -> TaskSpec:
    """Check payload against TASK_JSON_SCHEMA and build a TaskSpec with
    defaults filled in.

    With ``categories`` (open-vocabulary parse), the category fields must
    name categories the scene actually has: an invented one is a schema
    violation and earns the same reminder-and-retry as malformed JSON.
    """
    from bodynav.llm.validation import SchemaViolation

    goal_category, hazard_categories = None, {}
    if categories is not None:
        payload = dict(payload)
        allowed = set(categories)
        goal_category = payload.pop("goal_category", None)
        if goal_category is not None and goal_category not in allowed:
            raise SchemaViolation(f"goal_category {goal_category!r} is not one of "
                                  f"{sorted(allowed)}")
        hazard_categories = payload.pop("hazard_categories", None) or {}
        if not isinstance(hazard_categories, dict):
            raise SchemaViolation("hazard_categories must be an object")
        bad = {k: v for k, v in hazard_categories.items() if v not in allowed}
        if bad:
            raise SchemaViolation(f"hazard_categories {bad} use categories not in "
                                  f"{sorted(allowed)}")
    check_schema(payload)
    constraints = dict(payload.get("constraints") or {})
    constraints.setdefault("hazard_margin_m", DEFAULTS["hazard_margin_m"])
    constraints.setdefault("static_obstacle_margin_m",
                           DEFAULTS["static_obstacle_margin_m"])
    return TaskSpec(
        start=payload.get("start"),
        goal=payload["goal"],
        hazards=list(payload.get("hazards") or []),
        constraints=constraints,
        priority=Priority(payload["priority"]),
        safety_distances={k: float(v)
                          for k, v in (payload.get("safety_distances") or {}).items()},
        raw_instruction=raw_instruction,
        goal_category=goal_category,
        hazard_categories={str(k): str(v) for k, v in hazard_categories.items()},
    )


def parse_instruction(raw: str, llm_client, *, max_retries: int = 2,
                      trace: list | None = None, categories=None) -> TaskSpec:
    """LLM parse of one instruction into a validated TaskSpec.

    ``categories`` switches to the open-vocabulary prompt, which also asks
    the model to map each entity to one of the scene's categories.

    Raises SchemaViolation (see bodynav.llm.validation) on noncompliant
    output after the schema-reminder retries are exhausted.
    """
    from bodynav.llm.validation import SchemaViolation

    system = PARSE_SYSTEM_PROMPT if categories is None else open_parse_prompt(categories)
    user = f"Instruction: {raw}"
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        reply = llm_client.complete(system, user, max_tokens=4000)
        if trace is not None:
            trace.append({"kind": "parse_attempt", "attempt": attempt,
                          "reply": reply.text, "tokens": reply.total_tokens,
                          "latency_s": round(reply.latency_s, 3)})
        try:
            return validate_parse(extract_json(reply.text), raw_instruction=raw,
                                  categories=categories)
        except SchemaViolation as exc:
            last = exc
            user = f"Instruction: {raw}\n\n" + SCHEMA_REMINDER.format(error=exc)
            if trace is not None:
                trace.append({"kind": "schema_retry", "attempt": attempt,
                              "error": str(exc)})
    raise last
