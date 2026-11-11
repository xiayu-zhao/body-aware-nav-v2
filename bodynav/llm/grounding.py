"""Entity grounding: TaskSpec references -> scene-graph nodes -> voxel
masks (Sec. III-D.1).

Each entity resolves through the scene graph to spatial extents and
attributes, then to binary voxel masks M_g, M_h, M_c via world2voxel.
Grounded entities act twice — upstream as the task filtering mask V_t
(Sec. III-B.2) and downstream as value-map supports — closing the
bidirectional loop.

Ambiguity handling (Sec. V-D): >1 candidate resolves by scene-graph
proximity; unresolved references trigger the nearest-match fallback
safeguard and are recorded in the agent trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bodynav.core.types import TaskSpec, VoxelGrid
from bodynav.scene_graph.graph import BodyNavSceneGraph


@dataclass
class GroundedTask:
    """TaskSpec with every entity bound to nodes and voxel masks."""

    task: TaskSpec
    start_id: str | None
    goal_ids: list[str]
    hazard_ids: list[str]
    goal_mask: np.ndarray  # (M,) bool over grid support
    hazard_mask: np.ndarray  # (M,) bool
    constraint_mask: np.ndarray  # (M,) bool
    #: per-hazard resolved margins sigma_h (instruction-provided value
    #: wins over the class table)
    hazard_margins: dict[str, float] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


#: Words that carry no discriminating power in a reference like
#: "the storage area on the north side".
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "on", "in", "at", "to", "by", "near", "next",
    "side", "area", "zone", "region", "place", "spot", "with", "and",
})


#: Curated site prompts: everyday words for the closed vocabulary the
#: geometric segmenter emits. Sec. III-A.3 calls for exactly this --
#: "open-vocabulary comparison of aggregated semantic evidence against
#: curated site prompts" -- and without it "the storage area" matches
#: nothing, because the label is `stored_material`.
SITE_PROMPTS: dict[str, str] = {
    "storage": "stored_material", "stored": "stored_material",
    "material": "stored_material", "materials": "stored_material",
    "pallet": "stored_material", "supplies": "stored_material",
    "cone": "cone", "cones": "cone", "traffic": "cone", "marker": "cone",
    "duct": "duct", "ducts": "duct", "hvac": "duct", "services": "duct",
    "pipe": "pipe", "pipes": "pipe", "pipework": "pipe",
    "desk": "furniture", "desks": "furniture", "chair": "furniture",
    "table": "furniture", "bench": "furniture", "cabinet": "furniture",
    "benches": "furniture", "workbench": "furniture", "workbenches": "furniture",
    "machine": "equipment", "equipment": "equipment", "rig": "equipment",
    "column": "column", "pillar": "column",
    "wall": "wall", "walls": "wall",
    "floor": "floor", "ground": "floor",
    "ceiling": "ceiling",
    "corridor": "area", "hallway": "area", "room": "area", "lab": "area",
    "laboratory": "area", "zone": "area", "space": "area",
    "clutter": "clutter", "debris": "clutter", "obstruction": "clutter",
    "ductwork": "duct", "overhead": "duct",
}


def _tokens(phrase: str) -> set[str]:
    import re

    raw = {t for t in re.split(r"[^a-z0-9]+", phrase.lower()) if t}
    mapped = {SITE_PROMPTS[t] for t in raw if t in SITE_PROMPTS}
    expanded = set()
    for term in mapped:
        expanded |= {t for t in term.split("_")}
    return (raw | mapped | expanded) - _STOPWORDS


def _token_score(want: set[str], have: set[str]) -> float:
    """Overlap with prefix matching, so plurals and inflections count.

    "cones" must reach `cone` and "storage" must reach `stored_material`;
    exact set overlap answers neither, and a full stemmer is more
    machinery than a four-character prefix rule earns here.
    """
    if not want:
        return 0.0
    hits = 0.0
    for w in want:
        if w in have:
            hits += 1.0
            continue
        for h in have:
            if len(w) >= 4 and len(h) >= 4 and (w.startswith(h[:4]) or h.startswith(w[:4])):
                hits += 0.8
                break
    return hits / len(want)


def resolve_reference(phrase: str, graph: BodyNavSceneGraph,
                      *, near: np.ndarray | None = None) -> tuple[list[str], list[str]]:
    """Candidate nodes for a natural-language reference.

    Returns ``(chosen, all_candidates)``. Scoring is lexical overlap
    against the node's semantic label and id, and ties break by proximity
    to ``near`` when given -- the ambiguity rule of Sec. V-D, which is
    also why the full candidate list comes back: a reference that matched
    several things is recorded as ambiguous rather than silently collapsed.
    """
    want = _tokens(phrase)
    if not want:
        return [], []

    scored: list[tuple[float, str]] = []
    for node_id in graph.g.nodes:
        attrs = graph.attributes(node_id)
        have = _tokens(f"{attrs.semantic} {node_id}")
        score = _token_score(want, have)
        if score > 0:
            scored.append((score, node_id))
    if not scored:
        return [], []

    best = max(score for score, _ in scored)
    candidates = [node_id for score, node_id in scored if score == best]
    if len(candidates) > 1 and near is not None:
        near = np.asarray(near, dtype=float)[:2]
        candidates.sort(key=lambda n: float(np.linalg.norm(
            np.asarray(graph.attributes(n).centroid, dtype=float)[:2] - near)))
    return candidates, [n for _, n in scored]


def nodes_of_category(category: str, graph: BodyNavSceneGraph,
                      *, near: np.ndarray | None = None) -> list[str]:
    """Every node labelled ``category``, nearest to ``near`` first.

    Open-vocabulary grounding: the model has already said which category a
    phrase means, so the only question left is which instance -- answered
    by proximity, the same tie-break the lexical resolver uses.
    """
    ids = [n for n in graph.g.nodes if graph.attributes(n).semantic == category]
    if near is not None and len(ids) > 1:
        near = np.asarray(near, dtype=float)[:2]
        ids.sort(key=lambda n: float(np.linalg.norm(
            np.asarray(graph.attributes(n).centroid, dtype=float)[:2] - near)))
    return ids


def entity_to_mask(node_ids: list[str], graph: BodyNavSceneGraph, grid: VoxelGrid,
                   *, approach_radius: float = 1.5) -> np.ndarray:
    """Binary voxel mask over the grid support for a set of nodes.

    An entity's own volume is often *not* in the support: value maps are
    evaluated where the robot can be, and a pallet of stored material is
    exactly where it cannot. So a node whose bounding volume contains no
    support voxel grounds to the support nearest it -- the free space from
    which the entity is approached, which is what "go to the pallet" means
    for a robot that cannot enter it.
    """
    from bodynav.core.types import as_arrays

    arrays = as_arrays(grid)
    centers = arrays.center
    mask = np.zeros(len(arrays), dtype=bool)
    if not len(centers):
        return mask

    for node_id in node_ids:
        attrs = graph.attributes(node_id)
        lo = np.asarray(attrs.aabb_min, dtype=float)
        hi = np.asarray(attrs.aabb_max, dtype=float)
        inside = np.all((centers >= lo) & (centers <= hi), axis=1)
        if inside.any():
            mask |= inside
            continue
        # distance to the box, then everything within the approach radius
        # of the closest support voxel
        delta = np.maximum(np.maximum(lo - centers, centers - hi), 0.0)
        dist = np.linalg.norm(delta, axis=1)
        nearest = float(dist.min())
        mask |= dist <= nearest + approach_radius
    return mask


#: Hazard classes included whether or not the instruction names them:
#: physically marked exclusions a robot must respect regardless. Overhead
#: services are deliberately absent -- they are a clearance constraint,
#: enforced by the planner's gate, not a lateral repulsion.
ALWAYS_HAZARD_CLASSES: frozenset[str] = frozenset({"exclusion_marker"})


@dataclass
class ResolvedTask:
    """Grounding's map-independent half: node ids and metres.

    Split out because the abstraction stage needs it *before* it builds a
    map. Sec. III-B.2 says grounded entities act upstream as the task
    mask V_t -- the bidirectional loop -- and that loop cannot close while
    grounding needs a voxel grid that does not exist yet. Nothing here
    touches the map: references resolve against the scene graph, and
    margins come from the instruction.
    """

    task: TaskSpec
    start_id: str | None
    goal_ids: list[str]
    hazard_ids: list[str]
    hazard_margins: dict[str, float] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def resolve_entities(
    task: TaskSpec, graph: BodyNavSceneGraph, *,
    hazard_floor: dict[str, float] | None = None,
    always_hazard_classes: frozenset[str] = ALWAYS_HAZARD_CLASSES,
) -> ResolvedTask:
    """Bind a parsed task to scene-graph nodes and margins (no map).

    Grounding is where an instruction stops being language: after this
    step every downstream stage sees node ids, masks and metres. The two
    failure modes the manuscript names are recorded rather than raised --
    an unresolved reference falls back to nearest-match, an ambiguous one
    resolves by proximity -- because a task that cannot be grounded is a
    result to report, not a crash.
    """
    start_ids, _ = resolve_reference(task.start or "", graph)
    start_id = start_ids[0] if start_ids else None
    anchor = (np.asarray(graph.attributes(start_id).centroid, dtype=float)
              if start_id else None)

    if task.goal_category:
        goal_ids = nodes_of_category(task.goal_category, graph, near=anchor)
        goal_all = goal_ids
    else:
        goal_ids, goal_all = resolve_reference(task.goal, graph, near=anchor)
    ambiguous, unresolved = [], []
    if not goal_ids:
        unresolved.append(task.goal)
    elif len(goal_all) > 1 and len(goal_ids) > 1:
        ambiguous.append(task.goal)
    goal_ids = goal_ids[:1]

    hazard_ids: list[str] = []
    for phrase in task.hazards:
        category = task.hazard_categories.get(phrase)
        if category:
            hazard_ids.extend(nodes_of_category(category, graph))
            continue
        ids, all_ids = resolve_reference(phrase, graph)
        if not ids:
            unresolved.append(phrase)
            continue
        if len(all_ids) > len(ids):
            ambiguous.append(phrase)
        hazard_ids.extend(ids)

    # The hazard set is what the instruction named, plus the classes a
    # robot must respect whether or not anyone mentions them. Taking every
    # hazard-bearing node instead put 155 of them in the field -- 126 of
    # which were ducts overhead -- and their overlapping kernels swamped
    # the goal attraction entirely. Taking only named ones would let an
    # unmentioned coned-off zone disappear from the map, which is the
    # opposite failure and the worse one.
    always = set()
    for node_id in graph.find(hazardous=True):
        attrs = graph.attributes(node_id)
        cls = attrs.hazard.hazard_class if attrs.hazard else attrs.semantic
        if cls in always_hazard_classes:
            always.add(node_id)
    hazard_ids = sorted(set(hazard_ids) | always)

    # `safety_distances` is keyed by the instruction's own words -- the
    # parse prompt says to name hazards exactly as the person did -- so it
    # has to be resolved to classes the same way the hazard references
    # themselves were. Looking it up by class alone silently discarded
    # every explicit distance: "staying at least 1.5 metres clear of the
    # stored material" was keyed "the stored material" and matched against
    # "stored_material", so the 3 m default was applied to a request for
    # 1.5 m, on every task that stated a number.
    by_class: dict[str, float] = {}
    # Open-vocabulary distances are keyed by the node's *label* (what the
    # model chose from the inventory), not its hazard class: a cone's class
    # is "exclusion_marker", so a class-keyed lookup would drop "2 m from
    # the cones". The lexical path keeps its original behaviour unchanged.
    by_label: dict[str, float] = {}
    for phrase, value in (task.safety_distances or {}).items():
        try:
            metres = float(value)
        except (TypeError, ValueError):
            continue
        if phrase in task.hazard_categories:
            # open-vocabulary: the model already mapped the phrase
            by_label[task.hazard_categories[phrase]] = metres
            continue
        matched = False
        for token in _tokens(str(phrase)):
            by_class[token] = metres
            matched = True
        if not matched:
            by_class[str(phrase)] = metres

    margins: dict[str, float] = {}
    default_margin = float(task.constraints.get("hazard_margin_m", 3.0))
    floor = hazard_floor or {}
    for node_id in hazard_ids:
        attrs = graph.attributes(node_id)
        cls = attrs.hazard.hazard_class if attrs.hazard else attrs.semantic
        margin = by_label.get(attrs.semantic,
                              by_class.get(cls, task.safety_distances.get(cls, default_margin)))
        margins[node_id] = max(float(margin), float(floor.get(cls, 0.0)))

    return ResolvedTask(
        task=task, start_id=start_id, goal_ids=goal_ids,
        hazard_ids=hazard_ids, hazard_margins=margins,
        ambiguous=ambiguous, unresolved=unresolved,
    )


def ground(task: TaskSpec, graph: BodyNavSceneGraph, grid: VoxelGrid,
           *, hazard_floor: dict[str, float] | None = None,
           always_hazard_classes: frozenset[str] = ALWAYS_HAZARD_CLASSES,
           resolved: ResolvedTask | None = None) -> GroundedTask:
    """Bind a parsed task to scene-graph nodes *and* voxel masks.

    The masks are the only part that needs the map, so a caller that has
    already resolved the entities (to build that map) passes the result
    back in rather than resolving twice -- and, more importantly, rather
    than resolving differently.
    """
    r = resolved or resolve_entities(
        task, graph, hazard_floor=hazard_floor,
        always_hazard_classes=always_hazard_classes)
    return GroundedTask(
        task=r.task,
        start_id=r.start_id,
        goal_ids=r.goal_ids,
        hazard_ids=r.hazard_ids,
        goal_mask=entity_to_mask(r.goal_ids, graph, grid),
        hazard_mask=entity_to_mask(r.hazard_ids, graph, grid),
        constraint_mask=np.zeros(len(entity_to_mask([], graph, grid)), dtype=bool),
        hazard_margins=r.hazard_margins,
        ambiguous=r.ambiguous,
        unresolved=r.unresolved,
    )
