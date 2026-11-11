"""Constrained LLM agent loop (Sec. III-D).

Fixed, tool-augmented pipeline over (instruction I, scene graph G, voxel
grid + world2voxel, robot profile):

    parse -> ground -> generate code -> execute-and-validate

The agent is restricted to the approved toolset with deterministic
decoding, must return voxel arrays for V_A, V_H, V_C and their fusion,
and every output passes structural validation before reaching the
planner. The full trace (prompt -> parse -> tool calls -> code -> maps)
is retained on the PlanResult.

Backends: the manuscript evaluates LLM-Large and LLM-Mini via the OpenAI
API (``llm`` extra); the client is injected so other backends can be
swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bodynav.core.types import TaskSpec, ValueMapSet, VoxelGrid
from bodynav.llm.grounding import GroundedTask
from bodynav.llm.toolset import ApprovedToolset
from bodynav.robots.profiles import RobotProfile
from bodynav.scene_graph.graph import BodyNavSceneGraph


#: Labels that are structure, not things a person would name as a goal or
#: hazard; left out of the open-vocabulary inventory.
_NON_OBJECT = frozenset({"area", "floor", "ceiling", "wall"})


def scene_categories(graph: BodyNavSceneGraph) -> list[str]:
    """The object categories present in ``graph``, for the open parse."""
    return sorted({graph.attributes(n).semantic for n in graph.g.nodes}
                  - _NON_OBJECT)


@dataclass
class AgentConfig:
    model: str = "llm-large"  # resolved via configs/llm.yaml
    temperature: float = 0.0  # deterministic decoding
    seed: int | None = 0
    max_schema_retries: int = 2
    max_regeneration_retries: int = 2
    #: completion budget; reasoning models consume much of it invisibly
    max_tokens: int = 6000
    #: "lexical" -- the original keyword system: entity phrases matched to scene labels
    #: through a curated synonym table. "llm" -- open-vocabulary: the parse
    #: prompt lists the scene's categories and the model maps each phrase.
    grounding: str = "lexical"


@dataclass
class AgentTraceEvent:
    kind: str  # "prompt" | "parse" | "tool_call" | "code" | "maps" | "retry"
    payload: dict = field(default_factory=dict)


CODE_SYSTEM_PROMPT = """You write a short Python program that composes value
maps for a robot navigation planner.

You may call ONLY these functions, and use `np` (numpy). No imports, no
loops, no function definitions, no attribute access except on `np`.

  entity_mask(node_ids)            -> bool array over the voxel support
  distance_transform(mask)         -> metres to the nearest True voxel
  gaussian_kernel(mask, sigma)     -> exp(-d^2 / 2 sigma^2) from a mask
  find_nodes(semantic=..., hazardous=..., level=...) -> list of node ids
  node_attributes(node_id)         -> attributes of one node
  adjacent(node_id)                -> neighbouring node ids
  resolve_reference(phrase)        -> candidate node ids
  world2voxel(points), voxel2world(indices)

These names are already bound for you:
  GOAL_MASK      bool array, the grounded goal
  HAZARD_MASKS   list of (bool array, weight, sigma_metres) per hazard.
                 It is usually LONG. Every entry must contribute to V_H --
                 sum over all of them (a list comprehension is allowed).
  SIGMA_GOAL     float, goal attraction spread in metres
  SUPPORT        int, the length every array must have
  LAMBDA_A, LAMBDA_H, LAMBDA_C   float fusion weights for this priority

Define exactly four float arrays of length SUPPORT:
  V_A  goal attraction, in [0, 1]
  V_H  hazard repulsion, <= 0 everywhere
  V_C  constraint penalty, <= 0 everywhere
  V    the fusion LAMBDA_A*V_A + LAMBDA_H*V_H + LAMBDA_C*V_C

Return ONLY the program. No prose, no code fences, no comments needed.
"""

CODE_REMINDER = """Your previous program was rejected:
{error}

Return ONLY a program obeying the constraints above."""

#: What the generated program must define.
REQUIRED_OUTPUTS = ("V_A", "V_H", "V_C", "V")


class ConstrainedAgent:
    """The fixed pipeline of Sec. III-D: parse, ground, generate, validate.

    Every step is bounded by something checkable -- a JSON schema, a
    scene-graph lookup, an AST allowlist, an array-shape validation -- so
    a model that misbehaves produces a rejected attempt and a recorded
    retry rather than a bad map reaching the planner.
    """

    def __init__(self, config: AgentConfig, llm_client=None):
        self.config = config
        self.llm = llm_client
        self.trace: list[AgentTraceEvent] = []

    def _record(self, kind: str, **payload) -> None:
        self.trace.append(AgentTraceEvent(kind=kind, payload=payload))

    # -- steps ----------------------------------------------------------
    def parse(self, instruction: str, graph: BodyNavSceneGraph | None = None) -> TaskSpec:
        """Step 1: schema-bounded parse (retry on SchemaViolation)."""
        from bodynav.llm.schema import parse_instruction

        categories = None
        if self.config.grounding == "llm":
            if graph is None:
                raise ValueError("open-vocabulary grounding needs the scene graph")
            categories = scene_categories(graph)
        events: list[dict] = []
        try:
            task = parse_instruction(instruction, self.llm,
                                     max_retries=self.config.max_schema_retries,
                                     trace=events, categories=categories)
        finally:
            for event in events:
                self._record(event.pop("kind"), **event)
        self._record("parse", goal=task.goal, hazards=task.hazards,
                     priority=task.priority.value,
                     safety_distances=dict(task.safety_distances),
                     goal_category=task.goal_category,
                     hazard_categories=dict(task.hazard_categories))
        return task

    def pre_ground(self, instruction: str, graph: BodyNavSceneGraph):
        """Steps 1-2 without a map: parse, then resolve to ids and metres.

        The abstraction stage needs the grounded margins *before* it can
        build a map, because a stated margin decides where the robot may
        end up and therefore where the map has to be accurate. Running
        this first is what closes the bidirectional loop of Sec. III-B.2;
        the result is handed back to :meth:`compose` so the instruction is
        parsed once and grounded once.
        """
        from bodynav.llm.grounding import resolve_entities

        task = self.parse(instruction, graph)
        resolved = resolve_entities(task, graph)
        self._record("pre_ground", goal_ids=resolved.goal_ids,
                     hazard_ids=list(resolved.hazard_ids),
                     hazard_margins=dict(resolved.hazard_margins))
        return resolved

    def ground_entities(self, task: TaskSpec, graph: BodyNavSceneGraph,
                        grid: VoxelGrid, resolved=None) -> GroundedTask:
        """Step 2: bind entities to nodes and voxel masks."""
        from bodynav.llm.grounding import ground

        grounded = ground(task, graph, grid, resolved=resolved)
        self._record("ground", goal_ids=grounded.goal_ids,
                     hazards=len(grounded.hazard_ids),
                     # the ids and the resolved margins, not just a count:
                     # a margin the instruction *states* is a constraint the
                     # planner can enforce, and it cannot enforce a number
                     # it was never told
                     hazard_ids=list(grounded.hazard_ids),
                     hazard_margins=dict(grounded.hazard_margins),
                     ambiguous=grounded.ambiguous, unresolved=grounded.unresolved)
        return grounded

    def generate_code(self, grounded: GroundedTask, robot: RobotProfile,
                      *, error: str | None = None) -> str:
        """Step 3: emit value-map composition code against the toolset API."""
        summary = (
            f"Robot: {robot.name}, height band {robot.height_band[0]:.2f}-"
            f"{robot.height_band[1]:.2f} m, needs {robot.min_lateral_clearance:.2f} m "
            f"lateral and {robot.required_vertical_clearance:.2f} m vertical clearance.\n"
            f"Instruction: {grounded.task.raw_instruction}\n"
            f"Goal: {grounded.goal_ids}\n"
            f"Hazards: {len(grounded.hazard_ids)} nodes, "
            f"margins {sorted(set(round(v, 2) for v in grounded.hazard_margins.values()))} m\n"
            f"Priority: {grounded.task.priority.value}"
        )
        user = summary if error is None else summary + "\n\n" + CODE_REMINDER.format(error=error)
        # Reasoning models spend completion budget on hidden reasoning
        # before emitting a token of content: a tight cap returns an empty
        # string, which looks like a model that ignored the instructions
        # and is really a truncated request.
        reply = self.llm.complete(CODE_SYSTEM_PROMPT, user, max_tokens=self.config.max_tokens)
        code = _strip_fences(reply.text)
        self._record("code", code=code, tokens=reply.total_tokens,
                     latency_s=round(reply.latency_s, 3), model=reply.model,
                     backend=reply.backend)
        return code

    def execute_and_validate(self, code: str, toolset: ApprovedToolset,
                             bindings: dict | None = None) -> ValueMapSet:
        """Step 4: run code in the toolset-only sandbox, validate output."""
        import numpy as np

        from bodynav.core.types import as_arrays
        from bodynav.llm.sandbox import UnsafeCode, run
        from bodynav.llm.validation import MaskValidationError

        namespace = toolset.namespace()
        namespace.update(bindings or {})
        support = toolset.support()

        if not code.strip():
            raise MaskValidationError(
                "the model returned no code (raise AgentConfig.max_tokens: "
                "a reasoning model can spend the whole budget before "
                "emitting content)")
        try:
            out = run(code, namespace, outputs=REQUIRED_OUTPUTS)
        except UnsafeCode as exc:
            raise MaskValidationError(str(exc)) from exc

        arrays = {}
        for name, value in out.items():
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.shape != (support,):
                raise MaskValidationError(
                    f"{name} has length {arr.shape[0]}, expected {support}")
            if not np.all(np.isfinite(arr)):
                raise MaskValidationError(f"{name} contains non-finite values")
            arrays[name] = arr
        if np.any(arrays["V_H"] > 1e-9):
            raise MaskValidationError("V_H must be <= 0 (it is a repulsion)")
        if not np.any(arrays["V_A"] > 0):
            raise MaskValidationError("V_A is empty: the goal attracts nowhere")

        # Every hazard must actually repel. A program that writes
        # HAZARD_MASKS[0] and stops is syntactically fine, passes every
        # shape check, and produces a map that ignores 154 of 155 hazards
        # -- a silent safety failure, and the reason this check exists
        # rather than trusting the generated code to iterate.
        # Each hazard's own cells must carry at least half of what that
        # hazard alone contributes at zero distance (-weight). Testing
        # merely for "non-zero" is not enough: hazards overlap, so a
        # neighbour's kernel can cover an ignored one and hide the bug.
        hazard_masks = (bindings or {}).get("HAZARD_MASKS") or []
        ignored = []
        for i, (mask, weight, _) in enumerate(hazard_masks):
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                continue
            if float(arrays["V_H"][mask].min()) > -0.5 * float(weight):
                ignored.append(i)
        if ignored:
            raise MaskValidationError(
                f"V_H is zero at {len(ignored)} of {len(hazard_masks)} hazards "
                f"(e.g. index {ignored[0]}): every entry of HAZARD_MASKS must "
                f"contribute, not just the first")

        grid_arrays = as_arrays(toolset.grid)
        scale = float(np.max(np.abs(arrays["V"])))
        fused = arrays["V"] / scale if scale > 0 else arrays["V"]
        maps = ValueMapSet(
            indices=grid_arrays.index, affordance=arrays["V_A"],
            hazard=arrays["V_H"], constraint=arrays["V_C"], fused=fused,
            metadata={"centers": grid_arrays.center},
        )
        maps.validate()
        self._record("maps", support=support,
                     fused_range=[round(float(fused.min()), 3),
                                  round(float(fused.max()), 3)])
        return maps

    # -- entry ----------------------------------------------------------
    def compose(
        self,
        instruction: str,
        graph: BodyNavSceneGraph,
        grid: VoxelGrid,
        robot: RobotProfile,
        resolved=None,
    ) -> ValueMapSet:
        """Full agent pass for one instruction.

        ``resolved`` is the output of :meth:`pre_ground` when the caller
        already grounded the instruction to build the map. Re-parsing here
        would spend a second LLM call and, worse, could ground the same
        words to different nodes than the map was built around.
        """
        from bodynav.llm.validation import MaskValidationError
        from bodynav.llm.value_maps import FUSION_WEIGHTS, ValueMapParams

        task = resolved.task if resolved is not None else self.parse(instruction, graph)
        grounded = self.ground_entities(task, graph, grid, resolved=resolved)
        toolset = ApprovedToolset(graph, grid)

        params = ValueMapParams()
        lam_a, lam_h, lam_c = FUSION_WEIGHTS[task.priority]
        hazard_masks = [
            (toolset.entity_mask([node_id]), 1.0, float(margin))
            for node_id, margin in grounded.hazard_margins.items()
        ]
        bindings = {
            "GOAL_MASK": grounded.goal_mask,
            "HAZARD_MASKS": hazard_masks,
            "SIGMA_GOAL": params.sigma_g,
            "SUPPORT": toolset.support(),
            "LAMBDA_A": lam_a, "LAMBDA_H": lam_h, "LAMBDA_C": lam_c,
        }

        error: str | None = None
        for attempt in range(self.config.max_regeneration_retries + 1):
            code = self.generate_code(grounded, robot, error=error)
            try:
                return self.execute_and_validate(code, toolset, bindings)
            except MaskValidationError as exc:
                error = str(exc)
                self._record("retry", attempt=attempt, error=error)
        raise MaskValidationError(
            f"code generation failed after "
            f"{self.config.max_regeneration_retries + 1} attempts: {error}")


def _strip_fences(text: str) -> str:
    """Models fence code even when told not to; that is formatting, not a
    violation, so it is removed before the allowlist judges the content."""
    import re

    match = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    return (match.group(1) if match else text).strip()
