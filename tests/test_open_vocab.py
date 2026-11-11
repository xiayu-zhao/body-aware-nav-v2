"""Open-vocabulary grounding: the model maps phrases to scene categories.

Same scene as test_llm_stage (two areas, a pallet of stored material, two
cones); every reply is scripted, so nothing is spent.
"""
import json

import pytest

from bodynav.llm.agent import AgentConfig, ConstrainedAgent, scene_categories
from bodynav.llm.backends import ScriptedBackend
from bodynav.llm.grounding import ground, resolve_entities
from bodynav.llm.schema import open_parse_prompt, parse_instruction, validate_parse
from bodynav.llm.validation import SchemaViolation

from test_llm_stage import scene  # noqa: F401  (fixture)

CATS = ["cone", "stored_material"]


def test_inventory_lists_objects_not_structure(scene):
    graph, _ = scene
    assert scene_categories(graph) == CATS


def test_lexical_grounding_misses_unseen_words(scene):
    graph, grid = scene
    task = validate_parse({"goal": "the skids of spare stock", "priority": "safety_first"})
    assert ground(task, graph, grid).unresolved == ["the skids of spare stock"]


def test_category_grounding_resolves_them(scene):
    graph, grid = scene
    task = validate_parse({"goal": "the skids of spare stock", "priority": "safety_first",
                           "goal_category": "stored_material"}, categories=CATS)
    grounded = ground(task, graph, grid)
    assert grounded.goal_ids == ["spot_0_1_0"] and grounded.unresolved == []


def test_invented_category_is_a_schema_violation():
    with pytest.raises(SchemaViolation, match="not one of"):
        validate_parse({"goal": "x", "priority": "safety_first",
                        "goal_category": "helipad"}, categories=CATS)
    with pytest.raises(SchemaViolation, match="not in"):
        validate_parse({"goal": "x", "priority": "safety_first",
                        "hazards": ["the orange things"],
                        "hazard_categories": {"the orange things": "barrier"}},
                       categories=CATS)


def test_invented_category_is_retried():
    bad = json.dumps({"goal": "x", "priority": "safety_first", "goal_category": "helipad"})
    good = json.dumps({"goal": "x", "priority": "safety_first",
                       "goal_category": "stored_material"})
    replies = iter([bad, good])

    class Seq(ScriptedBackend):
        def complete(self, system, user, *, max_tokens=1200):
            out = super().complete(system, user, max_tokens=max_tokens)
            out.text = next(replies)
            return out

    task = parse_instruction("go there", Seq(), categories=CATS)
    assert task.goal_category == "stored_material"


def test_stated_distance_follows_the_category_to_the_cones(scene):
    """The cones' hazard class is exclusion_marker; the label is cone."""
    graph, _ = scene
    task = validate_parse({
        "goal": "the spare stock", "priority": "safety_first",
        "goal_category": "stored_material",
        "hazards": ["those orange markers"],
        "hazard_categories": {"those orange markers": "cone"},
        "safety_distances": {"those orange markers": 1.83}}, categories=CATS)
    resolved = resolve_entities(task, graph)
    assert {resolved.hazard_margins[n] for n in ("spot_0_0_1", "spot_0_0_2")} == {1.83}


def test_agent_shows_the_inventory_only_in_llm_mode(scene):
    graph, _ = scene
    base = {"goal": "the spare stock", "priority": "efficiency_first"}
    for mode, shown in (("lexical", False), ("llm", True)):
        # lexical keeps the template schema, which has no category keys
        reply = dict(base, goal_category="stored_material") if shown else base
        backend = ScriptedBackend(default=json.dumps(reply))
        agent = ConstrainedAgent(AgentConfig(grounding=mode), backend)
        resolved = agent.pre_ground("Hurry over to the spare stock.", graph)
        system = backend.calls[0][0]
        assert ("goods or building materials" in system) is shown
        if shown:
            assert resolved.goal_ids == ["spot_0_1_0"]


def test_open_prompt_converts_units():
    assert "0.3048" in open_parse_prompt(CATS)
