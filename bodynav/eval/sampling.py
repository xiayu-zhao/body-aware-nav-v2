"""Sampling evaluation tasks a platform could actually perform.

An instruction whose destination the robot cannot reach is not a test of
planning; it is a test of the building. Scoring those as planner failures
put PSR at 27.8% while almost none of the failures were the planner
failing something it could have solved -- 42 of 72 were "no feasible
path" because the goal lay outside the platform's reachable component.

So tasks are drawn **inside each platform's reachable component**, which
makes PSR mean what the manuscript says it means. Two things keep that
from being a way of quietly grading on a curve:

  * the reachable area itself is reported, per platform, as a result --
    it is the multi-embodiment finding, not something to hide;
  * infeasible instructions are still evaluated, as a separate
    correct-rejection metric. Refusing an impossible instruction is a
    success (Sec. V-D), and it is measured as one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

#: Everyday words for the closed vocabulary, and the phrasing a person
#: would actually use. Keyed by the semantic label the segmenter emits.
#:
#: Only floor-standing categories are destinations. This floor's 126 duct
#: and 16 pipe instances are at ceiling height: "go to the ductwork" is
#: not an instruction anyone gives a ground robot, and scoring the system
#: on it would measure nothing. They appear below as *hazards* instead,
#: which is the role they actually play -- and the role that makes the
#: H1's vertical clearance bind.
GOAL_PHRASES: dict[str, list[str]] = {
    "stored_material": ["the storage area", "the stored material",
                        "the pallet of supplies"],
    "equipment": ["the equipment", "the equipment bay", "the machine"],
    "furniture": ["the workbench", "the desks", "the benches"],
    "cone": ["the traffic cones", "the coned-off area"],
    "clutter": ["the clutter", "the obstruction in the aisle"],
}

#: Height *above the floor* at which a spot stops being somewhere a
#: ground robot could be sent; ducts hang from the soffit. Measured from
#: the floor, not from the world origin: Scene 1's floor sits at
#: z = 7.4 m in its stage, and an absolute test rejected every
#: floor-standing object on it as overhead.
GOAL_MAX_BASE_Z_M = 1.5

#: Instruction templates, one per priority, so the sampled set exercises
#: all three fusion modes rather than only the default.
TEMPLATES: list[tuple[str, str]] = [
    ("safety_first", "Go to {goal}, keeping well clear of {hazard}."),
    ("safety_first", "Move to {goal}, staying at least {margin} metres from {hazard}."),
    ("safety_first", "Navigate to {goal}, prioritising safety over speed."),
    ("efficiency_first", "Take the shortest route to {goal}."),
    ("efficiency_first", "Go to {goal} as quickly as possible."),
    ("area_preference", "Reach {goal} while staying in wide open space."),
    ("area_preference", "Head for {goal}, keeping to the main corridor."),
]

#: Hazard references, tagged with the category each names so a task never
#: tells the robot to approach a thing and avoid that same thing.
HAZARD_PHRASES: list[tuple[str, str]] = [
    ("cone", "the traffic cones"),
    ("stored_material", "the stored material"),
    ("clutter", "the clutter"),
    ("duct", "the overhead ductwork"),
    ("pipe", "the exposed pipes"),
]


def _hazard_boxes(graph, robot, floor_z: float = 0.0) -> dict:
    """In-band hazard instances per category, as (centre, aabb) pairs.

    Out-of-band instances are dropped because they are not hazards to
    *this* body: 126 of this floor's hazard nodes are ducts at ceiling
    height, which a ground robot cannot walk into and need not walk
    around. Telling it to keep 4 m from them asks for nothing.
    """
    band_lo = floor_z + float(robot.height_band[0])
    band_hi = floor_z + float(robot.height_band[1])
    out: dict[str, list] = {}
    for node_id in graph.nodes_by_level("spot"):
        a = graph.attributes(node_id)
        cls = a.hazard.hazard_class if a.hazard else a.semantic
        if a.aabb_min[2] > band_hi or a.aabb_max[2] < band_lo:
            continue
        out.setdefault(cls, []).append(
            (node_id, np.asarray(a.centroid, dtype=float)[:2]))
    return out


def _pick_hazard(rng, goal_category: str, *, boxes: dict | None = None,
                 start=None, goal=None,
                 near_m: float = 6.0) -> tuple[str, str, tuple] | None:
    """The hazard to name: in the robot's band, and near this route.

    A hazard 30 m off the path is avoided for free, so naming it makes
    the safety instruction indistinguishable from the efficiency one and
    the paired comparison measures nothing. Preference therefore goes to
    a category with an instance within ``near_m`` of the straight line
    between the endpoints -- somewhere the robot would otherwise go.

    The *specific instances* near the route are returned with it. "Keep
    clear of the clutter" cannot mean all 42 clutter piles on this floor
    -- corridors here are 2-4 m wide, so no path clears 3 m from every
    one of them and the metric would read 0% for every method including a
    perfect one. It means the clutter on the way, and that set is fixed by
    the task, identically for every method, before any of them plans.
    """
    choices = [(c, p) for c, p in HAZARD_PHRASES if c != goal_category]
    if boxes is None or start is None or goal is None:
        if not choices:
            return None
        cat, phrase = choices[int(rng.integers(len(choices)))]
        return cat, phrase, ()

    a, b = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]
    ab = b - a
    denom = float(ab @ ab) or 1.0
    scored = []
    for cat, phrase in choices:
        entries = boxes.get(cat)
        if not entries:
            continue
        ids = [e[0] for e in entries]
        pts = np.asarray([e[1] for e in entries])
        t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
        d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
        near = tuple(i for i, dist in zip(ids, d) if dist <= near_m)
        scored.append((len(near), cat, phrase, near))
    if not scored:
        return None
    scored.sort(key=lambda r: -r[0])
    hits = [r for r in scored if r[0] > 0] or scored
    pick = hits[int(rng.integers(len(hits)))]
    return pick[1], pick[2], pick[3]


@dataclass
class Task:
    """One sampled instruction, with the ground truth it was drawn from."""

    id: str
    text: str
    robot: str
    start_id: str
    goal_id: str
    goal_category: str
    priority: str
    separation_m: float
    start_xyz: tuple = ()
    goal_xyz: tuple = ()
    #: the hazard class the instruction names, and the margin it states
    #: (None when the wording is qualitative, e.g. "keeping well clear of")
    hazard_category: str | None = None
    hazard_margin_m: float | None = None
    #: the specific instances the instruction refers to -- those on the
    #: way. Fixed by the task, before any method plans.
    hazard_node_ids: tuple = ()
    #: tasks sharing a pair_id have the same start and goal and differ
    #: only in what the instruction asks for
    pair_id: str | None = None
    feasible_by_construction: bool = True
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "robot": self.robot,
            "start_id": self.start_id, "goal_id": self.goal_id,
            "goal_category": self.goal_category, "priority": self.priority,
            "separation_m": round(self.separation_m, 2),
            "hazard_category": self.hazard_category,
            "hazard_margin_m": self.hazard_margin_m,
            "hazard_node_ids": list(self.hazard_node_ids),
            "pair_id": self.pair_id,
            "start_xyz": [round(v, 3) for v in self.start_xyz],
            "goal_xyz": [round(v, 3) for v in self.goal_xyz],
            "feasible_by_construction": self.feasible_by_construction,
            **self.meta,
        }


#: How near the robot must be able to get for "go to X" to count as done.
#: The planner is given the same number as its goal-snap radius, so the
#: set of tasks called reachable and the set the planner accepts are the
#: same set by construction rather than by coincidence.
APPROACH_RADIUS_M = 3.0


def reachable_entities(graph, fields, planner, *,
                       max_approach_m: float = APPROACH_RADIUS_M):
    """Spots the platform can get to, and the areas it can start from.

    A spot counts as reachable when a cell of the platform's reachable
    component lies within ``max_approach_m`` of it -- the robot cannot
    enter a pallet, but it can stand beside one, and that is what "go to
    the pallet" means.
    """
    from scipy import ndimage

    layer_k, passable = planner.occupiable_layer(fields)
    labels, _ = ndimage.label(passable)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if not sizes.any():
        return {}, {}, np.zeros_like(passable), 0.0
    best = int(np.argmax(sizes))
    component = labels == best
    area_m2 = float(sizes[best]) * fields.pitch ** 2

    ci, cj = np.nonzero(component)
    cells = np.column_stack([ci, cj]).astype(float) * fields.pitch \
        + np.asarray(fields.origin)[:2] + fields.pitch / 2
    # the height the body actually occupies over each of those cells
    cell_z = (np.asarray(fields.origin)[2]
              + (layer_k[ci, cj] + 0.5) * fields.pitch)

    from scipy.spatial import cKDTree
    tree = cKDTree(cells)

    spots, areas = {}, {}
    for node_id in graph.g.nodes:
        attrs = graph.attributes(node_id)
        centre = np.asarray(attrs.centroid, dtype=float)[:2]
        d, idx = tree.query(centre, k=1)
        if d > max_approach_m:
            continue
        anchor = np.array([cells[idx][0], cells[idx][1], float(cell_z[idx])])
        record = {"node": node_id, "semantic": attrs.semantic,
                  "approach_m": float(d), "anchor": anchor}
        if node_id.startswith("spot_"):
            spots[node_id] = record
        elif node_id.startswith("area_"):
            areas[node_id] = record
    return spots, areas, component, area_m2


#: The paired instructions. Same destination, same named hazard, and the
#: only difference is what the person asked for -- which is the whole
#: variable BodyNav claims to respond to. A method that ignores the
#: instruction returns byte-identical paths for the two, and the
#: difference between them is therefore a direct measure of
#: instruction-following that needs no ground truth to score.
PAIR_TEMPLATES: list[tuple[str, str]] = [
    ("efficiency_first", "Take the shortest route to {goal}."),
    ("safety_first",
     "Go to {goal}, staying at least {margin} metres clear of {hazard}."),
]


#: Margins an instruction might state, in metres. The one used is chosen
#: per task, as the smallest that actually constrains the direct route.
MARGIN_LADDER = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)


def _as_boxes(hazards):
    """Accept hazard footprints as boxes, or points treated as zero-size.

    Boxes are what ``hazard_compliance`` measures against, so they are the
    form that keeps the sampler's feasibility test and the scoring test
    talking about the same geometry.
    """
    out = []
    for h in hazards if hazards is not None else ():
        arr = np.asarray(h, dtype=float)
        if arr.ndim == 2 and arr.shape == (2, 2):
            out.append((arr[0], arr[1]))
        else:
            xy = arr.reshape(-1)[:2]
            out.append((xy, xy))
    return out


def route_choice(planner, fields, start_xyz, goal_xyz, hazard_xy,
                 ladder=MARGIN_LADDER, offset: int = 0,
                 fixed_margin: float | None = None) -> dict | None:
    """Is there actually a decision to make on this task?

    An instruction can only change a route where more than one route
    exists. On a 10 m run down a single corridor the feasibility gate
    picks the path and every method returns the same one, whatever it was
    told -- which is what the first live pair showed: BodyNav and
    Pure-Planner agreed to the centimetre while the agent was correctly
    parsing "shortest" against "stay 1.5 m clear".

    A task is a test of instruction-following only if all three hold:
      * the direct route passes *inside* the stated margin, so the
        constraint bites;
      * a route honouring the margin exists, so compliance is achievable.

    The margin is therefore *chosen per task* rather than drawn blind, as
    the smallest rung of ``ladder`` that the direct route violates. A
    fixed margin cannot serve three sites of different openness: 1.5 m is
    hard in a 2 m laboratory corridor and free in a 60 x 30 m hall, where
    every candidate task was rejected because the shortest route already
    satisfied it. Calibrating to the route is what makes "keep clear"
    mean the same thing on each site; the value used is reported per
    task.

    Deliberately *not* required: that obeying be expensive. A compliant
    route costing 0.3 m more is the sharpest test there is -- obeying is
    nearly free, so a method that still walks past the hazard did so
    because it never read the instruction, not because it was trading
    safety against distance. Requiring an expensive detour rejected all
    120 candidate tasks on this floor; requiring only that the margin
    bite and be honourable keeps the ones that discriminate.

    Two knobs exist so the choice of margin can itself be swept, which is
    the answer to "were the margins tuned to favour the method".

    ``offset`` takes the *n*-th binding rung rather than the first, i.e. a
    demand n rungs stricter than the minimum that bites. There is no
    negative direction: every rung below the first is one the direct
    route already satisfies, so it would not bind at all.

    ``fixed_margin`` overrides the ladder with one value for every task
    and returns the record even when that value does not bind or cannot
    be honoured, flagging which. That is how a fixed margin is shown to
    be the wrong instrument rather than merely asserted to be.

    Returns the detour cost, or None when the task is not a test.
    """
    from scipy import ndimage

    direct = planner.search(None, None, fields, start_xyz, goal_xyz)
    if direct is None:
        return None
    d0 = float(np.linalg.norm(np.diff(direct, axis=0), axis=1).sum())

    seeds = np.ones(fields.obstacle_2d.shape, dtype=bool)
    origin, pitch = np.asarray(fields.origin, float), float(fields.pitch)
    marked = False
    for lo_xy, hi_xy in _as_boxes(hazard_xy):
        i0, j0 = np.floor((lo_xy - origin[:2]) / pitch).astype(int)
        i1, j1 = np.ceil((hi_xy - origin[:2]) / pitch).astype(int)
        i0, i1 = np.clip([i0, i1], 0, seeds.shape[0] - 1)
        j0, j1 = np.clip([j0, j1], 0, seeds.shape[1] - 1)
        seeds[i0:i1 + 1, j0:j1 + 1] = False
        marked = True
    if not marked:
        return None
    dist = ndimage.distance_transform_edt(seeds, sampling=pitch)

    # how close the unconstrained route already comes to this hazard set
    xy = np.clip(fields.index(direct)[:, :2], 0,
                 np.array(dist.shape) - 1)
    approach = float(dist[xy[:, 0], xy[:, 1]].min())

    def _record(margin_m, d1):
        return {"margin_m": float(margin_m),
                "direct_approach_m": round(approach, 2),
                "direct_m": round(d0, 2), "compliant_m": round(d1, 2),
                "detour_cost_m": round(d1 - d0, 2)}

    if fixed_margin is not None:
        binding = fixed_margin > approach + 1e-6
        near = dist < fixed_margin
        detour = planner.search(None, None, fields, start_xyz, goal_xyz,
                                forbidden=near)
        honourable = detour is not None
        d1 = (float(np.linalg.norm(np.diff(detour, axis=0), axis=1).sum())
              if honourable else float("nan"))
        rec = _record(fixed_margin, d1 if honourable else d0)
        rec.update(margin_binding=bool(binding),
                   margin_honourable=bool(honourable),
                   margin_source="fixed")
        return rec

    viable = []
    for margin_m in ladder:
        if margin_m <= approach + 1e-6:
            continue  # this rung costs nothing here
        near = dist < margin_m
        detour = planner.search(None, None, fields, start_xyz, goal_xyz,
                                forbidden=near)
        if detour is None:
            break  # nothing looser can be honoured either; stop climbing
        d1 = float(np.linalg.norm(np.diff(detour, axis=0), axis=1).sum())
        if d1 <= d0 + 0.05:
            continue  # same route; try a tighter demand
        viable.append((margin_m, d1))
        if len(viable) >= 4:
            break  # four strictness levels is more than any sweep needs
    if not viable:
        return None
    if offset >= len(viable):
        return None  # this task has no rung that strict and honourable
    margin_m, d1 = viable[offset]
    rec = _record(margin_m, d1)
    rec.update(margin_binding=True, margin_honourable=True,
               margin_source=f"ladder+{offset}", ladder_rungs_viable=len(viable))
    return rec


def sample_tasks(graph, fields, planner, robot, *, n: int = 12, seed: int = 0,
                 min_separation_m: float = 6.0, paired: bool = True,
                 require_choice: bool = True,
                 infeasible_fraction: float = 0.25,
                 margin_offset: int = 0,
                 fixed_margin: float | None = None) -> list[Task]:
    """Draw ``n`` tasks inside the platform's reachable component.

    A quarter of the set is deliberately drawn *outside* it: those are the
    instructions the system is supposed to refuse, and refusing them is
    scored as correct rejection rather than as failure. Sampling only
    feasible tasks would measure planning while quietly retiring the
    question of whether the system knows when to say no.
    """
    rng = np.random.default_rng(seed)
    spots, areas, component, area_m2 = reachable_entities(graph, fields, planner)
    if len(areas) < 1 or len(spots) < 1:
        return []

    floor_z = float((fields.meta or {}).get("floor_z", 0.0))
    usable = {k: v for k, v in spots.items()
              if v["semantic"] in GOAL_PHRASES
              and float(graph.attributes(k).aabb_min[2]) - floor_z
              <= GOAL_MAX_BASE_Z_M}
    if not usable:
        return []

    # unreachable spots of the same categories, for the rejection cases
    unreachable = [
        node_id for node_id in graph.nodes_by_level("spot")
        if node_id not in spots
        and graph.attributes(node_id).semantic in GOAL_PHRASES
        and float(graph.attributes(node_id).aabb_min[2]) - floor_z
        <= GOAL_MAX_BASE_Z_M
    ]

    # A minimum separation has to be read against the region the platform
    # can actually use. Eight metres is a short trip across the Husky's
    # 70 m2; it is most of the H1's 20 m2, which left only five feasible
    # tasks -- too few to say anything with. The floor is the smaller of
    # the stated distance and a fraction of the component's own extent, so
    # a task still crosses the reachable region rather than a corner of
    # it, on whatever size of region the embodiment turns out to have.
    ci, cj = np.nonzero(component)
    extent = float(np.hypot(np.ptp(ci), np.ptp(cj))) * fields.pitch if len(ci) else 0.0
    separation_floor = min(min_separation_m, 0.35 * extent)

    hazard_boxes = _hazard_boxes(graph, robot, floor_z)
    area_ids = sorted(areas)
    # Sample the *category* first, then an instance within it. Drawing
    # instances uniformly would hand two thirds of the tasks to `clutter`,
    # which has 78 instances against the cones' 3, and the benchmark would
    # then be a clutter benchmark wearing a category label.
    by_category: dict[str, list[str]] = {}
    for node_id, rec in sorted(usable.items()):
        by_category.setdefault(rec["semantic"], []).append(node_id)
    categories = sorted(by_category)
    n_infeasible = int(round(n * infeasible_fraction)) if unreachable else 0
    tasks: list[Task] = []
    attempts = 0

    while len(tasks) < n - n_infeasible and attempts < 2000:
        attempts += 1
        start_id = area_ids[int(rng.integers(len(area_ids)))]
        category = categories[len(tasks) % len(categories)]
        pool = by_category[category]
        goal_id = pool[int(rng.integers(len(pool)))]
        start = areas[start_id]["anchor"]
        goal = usable[goal_id]["anchor"]
        separation = float(np.linalg.norm(goal[:2] - start[:2]))
        if separation < separation_floor:
            continue
        if any(t.start_id == start_id and t.goal_id == goal_id for t in tasks):
            continue
        phrases = GOAL_PHRASES[category]
        goal_phrase = phrases[int(rng.integers(len(phrases)))]
        picked = _pick_hazard(rng, category, boxes=hazard_boxes,
                              start=start, goal=goal)
        if picked is None:
            continue
        haz_cat, haz_phrase, haz_ids = picked
        # Margins a 2-4 m corridor can actually honour. 3-5 m read 0%
        # compliance for every method, which measures the building.
        margin = float(rng.choice([1.5, 2.0, 2.5]))
        if require_choice:
            floor_z = float((fields.meta or {}).get("floor_z", 0.0))
            band_lo = floor_z + float(robot.height_band[0])
            band_hi = floor_z + float(robot.height_band[1])
            haz_xy = []
            for h in haz_ids or ():
                at = graph.attributes(h)
                # The judge drops instances outside the body's band -- a
                # duct overhead is headroom, not something to walk around
                # -- so the feasibility test must drop them too.
                if at.aabb_min[2] > band_hi or at.aabb_max[2] < band_lo:
                    continue
                haz_xy.append((np.asarray(at.aabb_min, float)[:2],
                               np.asarray(at.aabb_max, float)[:2]))
            if not haz_xy:
                continue
            choice = route_choice(planner, fields, start, goal, haz_xy,
                                  offset=margin_offset,
                                  fixed_margin=fixed_margin)
            if choice is None:
                continue
            margin = choice["margin_m"]
        else:
            margin = float(rng.choice([1.5, 2.0, 2.5]))
        variants = (PAIR_TEMPLATES if paired
                    else [TEMPLATES[len(tasks) % len(TEMPLATES)]])
        pair_id = f"p{len(tasks) // max(len(variants), 1) + 1:02d}"
        for suffix, (priority, template) in zip("ab", variants):
            tasks.append(Task(
                id=f"t{len(tasks) + 1:02d}{suffix if paired else ''}",
                text=template.format(goal=goal_phrase, hazard=haz_phrase,
                                     margin=f"{margin:g}"),
                robot=robot.name,
                start_id=start_id, goal_id=goal_id, goal_category=category,
                priority=priority, separation_m=separation,
                # Both members carry the category, because both paths are
                # measured against it; only the safety variant *states* a
                # margin, which is what makes the difference between them
                # attributable to the instruction.
                hazard_category=haz_cat,
                hazard_node_ids=haz_ids,
                hazard_margin_m=(float(margin) if "{margin}" in template
                                 else None),
                pair_id=pair_id if paired else None,
                start_xyz=tuple(float(v) for v in start),
                goal_xyz=tuple(float(v) for v in goal),
                meta={"approach_m": round(usable[goal_id]["approach_m"], 2),
                      **(choice if require_choice else {})}))

    for k in range(n_infeasible):
        start_id = area_ids[int(rng.integers(len(area_ids)))]
        goal_id = unreachable[int(rng.integers(len(unreachable)))]
        category = graph.attributes(goal_id).semantic
        phrases = GOAL_PHRASES[category]
        priority, template = TEMPLATES[k % len(TEMPLATES)]
        picked = _pick_hazard(rng, category) or ("clutter", "the clutter", ())
        haz_cat, haz_phrase, _ = picked
        margin = float(rng.choice([1.5, 2.0, 2.5]))
        text = template.format(
            goal=phrases[int(rng.integers(len(phrases)))],
            hazard=haz_phrase, margin=f"{margin:g}")
        centre = np.asarray(graph.attributes(goal_id).centroid, dtype=float)
        start = areas[start_id]["anchor"]
        tasks.append(Task(
            id=f"x{k + 1:02d}", text=text, robot=robot.name,
            start_id=start_id, goal_id=goal_id, goal_category=category,
            priority=priority,
            separation_m=float(np.linalg.norm(centre[:2] - start[:2])),
            start_xyz=tuple(float(v) for v in start),
            goal_xyz=tuple(float(v) for v in centre),
            feasible_by_construction=False))

    for t in tasks:
        t.meta["reachable_area_m2"] = round(area_m2, 1)
        t.meta["separation_floor_m"] = round(separation_floor, 1)
    return tasks
