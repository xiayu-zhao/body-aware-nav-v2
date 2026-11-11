"""Stage 3: schema-bounded parsing, grounding, and constrained code-gen.

Every test runs against :class:`ScriptedBackend`, so the suite needs no
API key and spends nothing. The scripted replies are the interesting
cases: a model that returns prose, one that invents a key, one that
writes code reaching for the filesystem, one that repels from only the
first hazard. Each must produce a *recorded rejection*, which is the
manuscript's claim -- language flexibility bounded by checkable output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bodynav.core.types import GridArrays, Priority, TaskSpec
from bodynav.llm.agent import AgentConfig, ConstrainedAgent
from bodynav.llm.backends import Completion, ScriptedBackend, load_backend
from bodynav.llm.grounding import entity_to_mask, ground, resolve_reference
from bodynav.llm.sandbox import UnsafeCode, check_code, run
from bodynav.llm.schema import (
    check_schema,
    extract_json,
    parse_instruction,
    validate_parse,
)
from bodynav.llm.toolset import ApprovedToolset
from bodynav.llm.validation import MaskValidationError, SchemaViolation
from bodynav.robots import load_profile
from bodynav.scene_graph.attributes import Affordances, HazardParams, NodeAttributes
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "robots"

GOOD_CODE = """
V_A = gaussian_kernel(GOAL_MASK, SIGMA_GOAL)
V_H = -np.sum(np.array([w * gaussian_kernel(m, s) for (m, w, s) in HAZARD_MASKS]), axis=0)
V_C = np.zeros(SUPPORT)
V = LAMBDA_A * V_A + LAMBDA_H * V_H + LAMBDA_C * V_C
"""


def _attrs(centre, half, semantic, hazard=None):
    centre, half = np.asarray(centre, float), np.asarray(half, float)
    return NodeAttributes(
        centroid=centre, aabb_min=centre - half, aabb_max=centre + half,
        semantic=semantic, affordances=Affordances(hazardous=hazard is not None),
        hazard=hazard,
    )


@pytest.fixture
def scene():
    """Two areas, a pallet of stored material, and two cones."""
    cone = lambda: HazardParams(clearance_radius=0.8, risk_weight=0.8,
                                hazard_class="exclusion_marker")
    attrs = {
        "floor_0": _attrs([10, 5, 1.2], [10, 5, 1.2], "floor"),
        "area_0_0": _attrs([3, 5, 1.0], [3, 5, 1.0], "area"),
        "area_0_1": _attrs([17, 5, 1.0], [3, 5, 1.0], "area"),
        "spot_0_1_0": _attrs([17, 5, 0.5], [1, 1, 0.5], "stored_material"),
        "spot_0_0_1": _attrs([6, 3, 0.3], [0.2, 0.2, 0.3], "cone", cone()),
        "spot_0_0_2": _attrs([8, 7, 0.3], [0.2, 0.2, 0.3], "cone", cone()),
    }
    graph = BodyNavSceneGraph.build(LabeledPoints.from_xyz(np.zeros((1, 3))), attrs)
    xs, ys = np.meshgrid(np.arange(0.5, 20, 0.5), np.arange(0.5, 10, 0.5))
    centers = np.column_stack([xs.ravel(), ys.ravel(),
                               np.full(xs.size, 0.2)])
    n = len(centers)
    grid = GridArrays(
        origin=np.zeros(3), base_pitch=0.5,
        index=np.floor(centers / 0.5).astype(int), center=centers,
        resolution=np.full(n, 0.5), occupancy=np.ones(n, dtype=int),
        semantic_label=np.full(n, "free", dtype=object),
        spot_id=np.full(n, None, dtype=object),
        synthetic=np.zeros(n, dtype=bool), confidence=np.ones(n),
    )
    return graph, grid


# -- schema -------------------------------------------------------------
def test_parse_accepts_a_valid_reply():
    task = validate_parse({"goal": "the lab", "hazards": ["cones"],
                           "priority": "safety_first"}, "go to the lab")
    assert task.goal == "the lab" and task.priority is Priority.SAFETY_FIRST
    # domain defaults are filled rather than demanded of the model
    assert task.constraints["hazard_margin_m"] == 3.0


@pytest.mark.parametrize("payload,why", [
    ({"goal": "x"}, "missing priority"),
    ({"goal": "x", "priority": "fast"}, "priority not in the enum"),
    ({"goal": 3, "priority": "safety_first"}, "goal is not a string"),
    ({"goal": "x", "priority": "safety_first", "colour": "red"}, "extra key"),
    ({"goal": "x", "priority": "safety_first", "hazards": [1]}, "hazard not a string"),
])
def test_schema_rejects(payload, why):
    with pytest.raises(SchemaViolation):
        validate_parse(payload)


def test_extract_json_survives_fences_and_prose():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('The answer is {"a": 2} I think') == {"a": 2}
    with pytest.raises(SchemaViolation):
        extract_json("no object here")


def test_schema_reminder_retry_recovers():
    """First reply is prose, second is valid: the run should succeed."""
    backend = ScriptedBackend(
        # the retry quotes the error back, so it is what distinguishes the
        # second call from the first
        responses={"previous reply": '{"goal": "the bay", "priority": "safety_first"}'},
        default="I cannot do that.")
    trace: list[dict] = []
    task = parse_instruction("go to the bay", backend, trace=trace)
    assert task.goal == "the bay"
    assert any(e["kind"] == "schema_retry" for e in trace)


def test_parse_gives_up_after_the_retries():
    backend = ScriptedBackend(default="not json")
    with pytest.raises(SchemaViolation):
        parse_instruction("go", backend, max_retries=1)
    assert len(backend.calls) == 2


# -- grounding ----------------------------------------------------------
def test_site_prompts_bridge_everyday_words(scene):
    graph, _ = scene
    assert resolve_reference("the storage area", graph)[0] == ["spot_0_1_0"]
    assert set(resolve_reference("traffic cones", graph)[0]) == {"spot_0_0_1", "spot_0_0_2"}


def test_unresolved_reference_is_recorded_not_raised(scene):
    graph, grid = scene
    task = validate_parse({"goal": "the helipad", "priority": "safety_first"})
    grounded = ground(task, graph, grid)
    assert grounded.unresolved == ["the helipad"]
    assert grounded.goal_ids == []


def test_ambiguous_reference_resolves_by_proximity(scene):
    graph, _ = scene
    both, _ = resolve_reference("cone", graph)
    assert set(both) == {"spot_0_0_1", "spot_0_0_2"}
    # the tie breaks by proximity to the anchor
    near_first, _ = resolve_reference("cone", graph, near=np.array([6.0, 3.0]))
    near_second, _ = resolve_reference("cone", graph, near=np.array([8.0, 7.0]))
    assert near_first[0] == "spot_0_0_1"
    assert near_second[0] == "spot_0_0_2"


def test_ambiguity_is_recorded_on_the_grounded_task(scene):
    graph, grid = scene
    task = validate_parse({"goal": "cone", "priority": "safety_first"})
    grounded = ground(task, graph, grid)
    assert "cone" in grounded.ambiguous
    assert len(grounded.goal_ids) == 1


def test_entity_mask_falls_back_to_the_approach_space(scene):
    """A pallet is not free space, so its mask is the space beside it."""
    graph, grid = scene
    mask = entity_to_mask(["spot_0_1_0"], graph, grid)
    assert mask.any()
    reached = grid.center[mask]
    assert np.linalg.norm(reached.mean(axis=0)[:2] - np.array([17.0, 5.0])) < 2.0


def test_instruction_margin_overrides_the_default(scene):
    graph, grid = scene
    task = validate_parse({"goal": "the storage area", "priority": "safety_first",
                           "safety_distances": {"exclusion_marker": 5.0}})
    grounded = ground(task, graph, grid)
    assert set(grounded.hazard_margins.values()) == {5.0}


def test_margin_floor_is_enforced(scene):
    graph, grid = scene
    task = validate_parse({"goal": "the storage area", "priority": "safety_first",
                           "safety_distances": {"exclusion_marker": 0.1}})
    grounded = ground(task, graph, grid, hazard_floor={"exclusion_marker": 1.0})
    assert set(grounded.hazard_margins.values()) == {1.0}


# -- sandbox ------------------------------------------------------------
@pytest.mark.parametrize("source,reason", [
    ("import os", "imports"),
    ("from os import path", "imports"),
    ("V_A = open('/etc/passwd')", "unknown name"),
    ("V_A = __import__('os')", "dunder"),
    ("V_A = np.zeros(3).__class__", "dunder attribute"),
    ("V_A = eval('1')", "unknown name"),
    ("for i in range(3):\n    V_A = i", "statement loops"),
    ("def helper():\n    return 1", "definitions"),
    ("V_A = (lambda: 1)()", "lambda"),
    ("V_A = np._core", "private attribute"),
    ("with open('f') as fh:\n    V_A = 1", "context managers"),
    ("V_A = missing_name", "unknown name"),
])
def test_sandbox_rejects(source, reason):
    with pytest.raises(UnsafeCode):
        check_code(source, {"np", "V_A"})


def test_sandbox_allows_the_expressive_subset():
    ns = {"np": np, "gaussian_kernel": lambda m, s: np.ones(4),
          "HAZARD_MASKS": [(np.ones(4, bool), 1.0, 2.0)], "SUPPORT": 4,
          "len": len, "float": float}
    out = run(
        "V = -np.sum(np.array([w * gaussian_kernel(m, s) for (m, w, s) in HAZARD_MASKS]), axis=0)\n"
        "V = V.astype(float).clip(-1.0, 0.0)",
        ns, outputs=("V",))
    assert out["V"].shape == (4,)


def test_sandbox_runs_without_builtins():
    """Nothing is inherited: only what the toolset binds is callable."""
    with pytest.raises(UnsafeCode):
        run("V = getattr(np, 'zeros')(3)", {"np": np}, outputs=("V",))


# -- agent loop ---------------------------------------------------------
def _agent(scene, code_reply, parse_reply=None):
    parse_reply = parse_reply or json.dumps(
        {"goal": "the storage area", "hazards": ["cones"], "priority": "safety_first"})
    # the code prompt is the one that describes the robot; the parse
    # prompt is everything else
    backend = ScriptedBackend(responses={"Robot:": code_reply}, default=parse_reply)
    return ConstrainedAgent(AgentConfig(max_regeneration_retries=1), backend)


def test_agent_composes_maps_end_to_end(scene):
    graph, grid = scene
    agent = _agent(scene, GOOD_CODE)
    maps = agent.compose("go to the storage area", graph, grid,
                         load_profile("husky_ur5", config_dir=CONFIG_DIR))
    assert len(maps.fused) == len(grid)
    assert maps.fused.min() >= -1.0 and maps.fused.max() <= 1.0
    assert np.all(maps.hazard <= 1e-9)
    kinds = [e.kind for e in agent.trace]
    assert kinds[:3] == ["parse_attempt", "parse", "ground"]
    assert "maps" in kinds


def test_agent_rejects_code_that_ignores_hazards(scene):
    """The failure that shape checks miss: repel from only the first."""
    graph, grid = scene
    only_first = (
        "V_A = gaussian_kernel(GOAL_MASK, SIGMA_GOAL)\n"
        "V_H = -gaussian_kernel(HAZARD_MASKS[0][0], HAZARD_MASKS[0][2])\n"
        "V_C = np.zeros(SUPPORT)\n"
        "V = LAMBDA_A * V_A + LAMBDA_H * V_H + LAMBDA_C * V_C\n"
    )
    agent = _agent(scene, only_first)
    with pytest.raises(MaskValidationError, match="every entry of HAZARD_MASKS"):
        agent.compose("go to the storage area", graph, grid,
                      load_profile("husky_ur5", config_dir=CONFIG_DIR))
    assert sum(1 for e in agent.trace if e.kind == "retry") == 2


def test_agent_rejects_unsafe_code(scene):
    graph, grid = scene
    agent = _agent(scene, "V_A = open('/etc/passwd').read()")
    with pytest.raises(MaskValidationError):
        agent.compose("go", graph, grid,
                      load_profile("husky_ur5", config_dir=CONFIG_DIR))


def test_agent_reports_an_empty_reply_as_a_budget_problem(scene):
    graph, grid = scene
    agent = _agent(scene, "")
    with pytest.raises(MaskValidationError, match="max_tokens"):
        agent.compose("go", graph, grid,
                      load_profile("husky_ur5", config_dir=CONFIG_DIR))


def test_offline_backend_is_labelled_as_such():
    """A scripted run must never be mistaken for a model run."""
    backend = load_backend("llm-large", offline=True)
    assert backend.name == "scripted"
    assert "offline" in backend.model
    assert backend.complete("s", "u").backend == "scripted"
