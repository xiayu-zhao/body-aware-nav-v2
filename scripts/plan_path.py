#!/usr/bin/env python3
"""Stage 4: obstacles, value maps, value-guided A*, post-processing.

    PYTHONPATH=. .venv/bin/python scripts/plan_path.py \
        --graph results/labfloor/scene_graph_nodes.json \
        --grid data/labfloor/grid_husky_ur5.npz --robot husky_ur5

Stage 3 does not exist yet, so the value maps are built from the scene
graph directly -- the goal entity, the hazard nodes with their resolved
clearance radii, and the slope and corridor width the obstacle model
measures. That is exactly the argument list stage 3's agent will emit as
schema-validated code; the fusion and the search below do not change when
it lands. Building it this way is what lets stage 4 be measured now.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

from bodynav.core.types import GridArrays, Priority
from bodynav.io import load_grid_arrays
from bodynav.llm.grounding import ALWAYS_HAZARD_CLASSES
from bodynav.llm.value_maps import ValueMapParams, build_value_maps
from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
from bodynav.planning.obstacles import build_collision_primitives, compute_clearance_fields
from bodynav.planning.postprocess import _segment_ok, revalidate, shortcut, smooth_bspline
from bodynav.robots import load_profile
from bodynav.scene_graph.graph import BodyNavSceneGraph

REPO = Path(__file__).resolve().parents[1]


def compose_with_agent(instruction: str, graph, lattice, robot, *,
                       backend: str | None = None, offline: bool = False,
                       resolved=None, grounding: str = "lexical"):
    """Stage 3: instruction -> parse -> ground -> generated code -> maps.

    The agent composes the same value maps stage 4 already consumes, so
    the only thing that changes when the LLM is in the loop is where the
    parameters come from -- which is the point of the split.
    """
    from bodynav.llm.agent import AgentConfig, ConstrainedAgent
    from bodynav.llm.backends import load_backend

    client = load_backend(backend, offline=offline)
    agent = ConstrainedAgent(AgentConfig(model=backend or "llm-large",
                                         grounding=grounding), client)
    print(f"agent  {client.name}/{client.model}: {instruction!r}")
    maps = agent.compose(instruction, graph, lattice, robot, resolved=resolved)

    events = [{"kind": e.kind, **e.payload} for e in agent.trace]
    tokens = sum(int(e.get("tokens", 0) or 0) for e in events)
    latency = sum(float(e.get("latency_s", 0) or 0) for e in events)
    retries = sum(1 for e in events if e["kind"] in ("schema_retry", "retry"))
    code = next((e["code"] for e in reversed(events) if e["kind"] == "code"), "")
    parse = next((e for e in events if e["kind"] == "parse"), {})
    ground = next((e for e in events if e["kind"] == "ground"), {})
    print(f"  parse   goal={parse.get('goal')!r} priority={parse.get('priority')} "
          f"hazards={parse.get('hazards')}")
    print(f"  ground  {ground.get('goal_ids')} | {ground.get('hazards')} hazard nodes"
          f" | ambiguous {ground.get('ambiguous')} unresolved {ground.get('unresolved')}")
    print(f"  code    {len(code.splitlines())} lines, {retries} retries, "
          f"{tokens} tokens, {latency:.1f} s")
    info = {
        "instruction": instruction,
        "backend": client.name, "model": client.model,
        "parse": parse, "ground": ground,
        "code": code, "retries": retries,
        "llm_tokens": tokens, "llm_latency_s": round(latency, 2),
        "trace": events,
    }
    return maps, info


def compose_controlled(instruction, graph, lattice, robot, fields, lattice_ijk,
                       resolved, config, *, replay_code=None, deterministic=False):
    """Controlled stage 3 for the revision ablations (no LLM call).

    Both paths use the *same* parse and grounding BodyNav used (``resolved``)
    and the same masks, margins, and priority, so only field composition
    differs:

    - ``replay_code``: re-execute the program a stored BodyNav run generated,
      through the same sandbox and validator.
    - ``deterministic``: build the fields directly from Eqs. (5)-(7):
      V_A a Gaussian of distance to the grounded goal cells, V_H one
      Gaussian per grounded hazard instance with sigma equal to its stated
      margin, V_C the reference slope/width penalty, fused with the
      priority's weights and the reference normalisation.
    """
    from bodynav.core.types import ValueMapSet
    from bodynav.llm.agent import ConstrainedAgent, AgentConfig
    from bodynav.llm.toolset import ApprovedToolset
    from bodynav.llm.value_maps import (FUSION_WEIGHTS, ValueMapParams,
                                     constraint_map, fuse)

    task = resolved.task
    agent = ConstrainedAgent(AgentConfig(), None)
    grounded = agent.ground_entities(task, graph, lattice, resolved=resolved)
    toolset = ApprovedToolset(graph, lattice)
    params = ValueMapParams()
    hazard_masks = [(toolset.entity_mask([node_id]), 1.0, float(margin))
                    for node_id, margin in grounded.hazard_margins.items()]
    if replay_code is not None:
        lam_a, lam_h, lam_c = FUSION_WEIGHTS[task.priority]
        bindings = {"GOAL_MASK": grounded.goal_mask, "HAZARD_MASKS": hazard_masks,
                    "SIGMA_GOAL": params.sigma_g, "SUPPORT": toolset.support(),
                    "LAMBDA_A": lam_a, "LAMBDA_H": lam_h, "LAMBDA_C": lam_c}
        maps = agent.execute_and_validate(replay_code, toolset, bindings)
        mode = "replay"
    else:
        vm = config["value_maps"]
        v_a = np.clip(toolset.gaussian_kernel(grounded.goal_mask, params.sigma_g), 0.0, 1.0)
        v_h = np.zeros(toolset.support())
        for mask, weight, sigma in hazard_masks:
            if np.asarray(mask, bool).any():
                v_h -= weight * toolset.gaussian_kernel(mask, sigma)
        ijk = lattice_ijk
        v_c = constraint_map(fields.slope_grid[ijk[:, 0], ijk[:, 1]],
                             fields.lateral_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * 2.0,
                             robot.min_lateral_clearance,
                             ValueMapParams(sigma_g=vm["sigma_g_m"], theta_safe=vm["theta_safe"],
                                            alpha_s=vm["alpha_s"], alpha_w=vm["alpha_w"]))
        fused = fuse(v_a, v_h, v_c, task.priority, smooth_sigma=vm["smooth_sigma"])
        maps = ValueMapSet(indices=lattice.index, affordance=v_a, hazard=v_h,
                           constraint=v_c, fused=fused,
                           metadata={"centers": lattice.center,
                                     "priority": np.array([task.priority.value])})
        maps.validate()
        mode = "deterministic"
    ground = next((e.payload for e in agent.trace if e.kind == "ground"), {})
    info = {"instruction": instruction, "backend": mode, "model": None,
            "priority": task.priority.value,
            "parse": {"goal": task.goal, "hazards": task.hazards,
                      "priority": task.priority.value},
            "ground": ground, "code": replay_code or "", "retries": 0,
            "llm_tokens": 0, "llm_latency_s": 0.0, "trace": []}
    return maps, info


def reachable_areas(planner, fields, graph) -> dict:
    """The largest region this platform can actually occupy, and the area
    nodes inside it."""
    from scipy import ndimage

    layer_k, passable = planner.occupiable_layer(fields)
    lab, _ = ndimage.label(passable)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if not sizes.any():
        return {"areas": [], "area_m2": 0.0, "components": 0}
    best = int(np.argmax(sizes))

    # Membership is decided by where the planner would actually *start*
    # from that area -- its snapped cell -- not by whether the component
    # passes near the centroid. A window test in cells silently changes
    # meaning when the lattice pitch changes, and picked anchors the
    # search then could not reach.
    component = lab == best
    inside, anchors = [], {}
    for node_id in graph.nodes_by_level("area"):
        c = np.asarray(graph.attributes(node_id).centroid, dtype=float)
        snapped = planner._snap_ground(c, fields, component)
        if snapped is None:
            continue
        inside.append(node_id)
        # Keep the cell, not the centroid. Handing the planner a centroid
        # lets its own snap pick a nearer cell in a *different* component,
        # and the search then fails on a task that was chosen as reachable.
        k = int(np.clip(layer_k[snapped], 0, fields.occupied.shape[2] - 1))
        anchors[node_id] = fields.world(np.array([[snapped[0], snapped[1], k]]))[0]
    return {"areas": inside, "anchors": anchors,
            "area_m2": float(sizes[best]) * fields.pitch ** 2,
            "components": int((sizes > 0).sum())}


def farthest_pair(graph, area_ids: list[str], anchors: dict | None = None):
    """The most separated pair among reachable areas -- a route that
    crosses the reachable region rather than a corner of it."""
    if len(area_ids) < 2:
        return None
    anchors = anchors or {}
    xy = np.array([np.asarray(anchors.get(a, graph.attributes(a).centroid))[:2]
                   for a in area_ids])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return area_ids[i], area_ids[j]


def subgoal_chain(graph, start_id: str, goal_id: str) -> list[str]:
    """Places to pass through on the way, at the scene graph's own level.

    Adjacency edges only join nodes of the same level, so asking for a
    route from an area straight to a spot finds nothing and silently
    returns the two endpoints -- which turns a hierarchical baseline into
    the flat planner it was supposed to be compared against. The chain is
    therefore built between *areas*, hopping to the goal's container
    through its containment edge, and the goal is appended last.
    """
    from bodynav.abstraction.task_filter import route_between

    container = goal_id
    if not goal_id.startswith("area_"):
        parents = [v for v, d in graph.g[goal_id].items()
                   if d.get("kind") == "containment" and v.startswith("area_")]
        container = parents[0] if parents else None
    if container is None or container == start_id:
        return [start_id, goal_id]
    chain = route_between(graph, start_id, container)
    if chain[-1] != container:  # no adjacency route; fall back to endpoints
        return [start_id, goal_id]
    return chain + ([goal_id] if chain[-1] != goal_id else [])


def path_metrics(path: np.ndarray, fields, straight: float) -> dict:
    """Length, clearance and hazard statistics on the accepted trajectory."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    ijk = np.clip(fields.index(path), 0, np.array(fields.occupied.shape) - 1)
    lat = fields.lateral_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    haz = fields.hazard_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    finite_haz = haz[np.isfinite(haz)]
    return {
        "length_m": round(float(seg.sum()), 2),
        "straight_line_m": round(float(straight), 2),
        "normalized_length": round(float(seg.sum() / straight), 3) if straight else None,
        "waypoints": int(len(path)),
        "min_clearance_m": round(float(lat.min()), 3),
        "mean_clearance_m": round(float(lat.mean()), 3),
        "min_hazard_distance_m": (round(float(finite_haz.min()), 3)
                                  if len(finite_haz) else None),
    }


def margin_constraint(fields, graph, hazard_ids, hazard_margins, robot,
                     floor_z: float = 0.0, start_xyz=None, goal_xyz=None,
                     corridor_m: float = 8.0):
    """Cells the instruction's stated margins put out of bounds.

    "Stay at least 1.5 metres clear of the stored material" states a
    minimum, and a soft repulsion cannot deliver a minimum: with eta at
    0.2 the value map can bias the cost of a step by a fifth, which bends
    a route toward safety without ever guaranteeing the number the person
    asked for. Measured, that is a 0.16 m improvement against a 1.5 m
    request.

    So a stated margin is enforced where one is stated -- which is what
    "feasibility-gated" means in Sec. III-E -- and the soft field still
    does the work for hazards named without a number. The caller falls
    back to the unconstrained search when no compliant route exists,
    because refusing to move is not a safer answer than routing past a
    pallet at 1.2 m, and records that it did.
    """
    from scipy import ndimage

    band_lo = floor_z + float(robot.height_band[0])
    band_hi = floor_z + float(robot.height_band[1])
    shape = fields.obstacle_2d.shape
    forbidden = np.zeros(shape, dtype=bool)
    applied = {}
    seg = None
    if start_xyz is not None and goal_xyz is not None:
        seg = (np.asarray(start_xyz, float)[:2], np.asarray(goal_xyz, float)[:2])
    for node_id in hazard_ids or []:
        margin = hazard_margins.get(node_id)
        if not margin:
            continue
        attrs = graph.attributes(node_id)
        if attrs.aabb_min[2] > band_hi or attrs.aabb_max[2] < band_lo:
            continue  # overhead: a clearance constraint, not one to walk around
        if seg is not None:
            # "keep clear of the stored material" constrains the piles on
            # the way, not all thirteen on the floor. Excluding a 3 m disc
            # around every instance of a class closes the building and the
            # search falls back, enforcing nothing at all -- which is what
            # it did before this test existed.
            a, b = seg
            ab = b - a
            c = np.asarray(attrs.centroid, dtype=float)[:2]
            t = float(np.clip((c - a) @ ab / (float(ab @ ab) or 1.0), 0.0, 1.0))
            if float(np.linalg.norm(c - (a + t * ab))) > corridor_m:
                continue
        a = fields.index(np.asarray(attrs.aabb_min, dtype=float))[0][:2]
        b = fields.index(np.asarray(attrs.aabb_max, dtype=float))[0][:2]
        seed = np.ones(shape, dtype=bool)
        i0, i1 = np.clip([a[0], b[0] + 1], 0, shape[0])
        j0, j1 = np.clip([a[1], b[1] + 1], 0, shape[1])
        if i0 >= i1 or j0 >= j1:
            continue
        seed[i0:i1, j0:j1] = False
        # Same allowance the body-clearance gate makes, for the same
        # reason: distance is sampled at cell centres, so a point in an
        # admitted cell can be half a cell diagonal nearer the hazard than
        # the sample says. Without it a 2.0 m request was met at 1.987 m
        # -- 13 mm short, enforced correctly, and still a violation.
        forbidden |= ndimage.distance_transform_edt(
            seed, sampling=fields.pitch) < (float(margin)
                                            + 0.5 * fields.pitch * np.sqrt(2))
        applied[node_id] = float(margin)
    return (forbidden, applied) if applied else (None, {})


def plan_once(robot, grid, graph, prior, config, *, instruction=None,
              backend=None, offline=False, pitch=0.1, priority="safety_first",
              start_id=None, goal_id=None, start_xyz=None, goal_xyz=None,
              eta=None, smooth=True, value_maps="bodynav", seed=0, verbose=True,
              fields=None, true_fields=None, cache=None,
              hazard_category=None, hazard_margin_m=None, hazard_node_ids=(),
              agent_pre=None, replay_code=None, deterministic_fields=False,
              hard_gate=True, gate_margin_offset=0.0, grounding="lexical"):
    """One instruction end to end on cached stage-1/2 artefacts.

    Returns a result dict; raises nothing for an infeasible instruction --
    "no path" is an answer the evaluation needs, not an error.

    ``value_maps`` selects what guides the search, which is how the
    baselines are run through exactly the same code as BodyNav:
      "bodynav"    the reference value maps (or the agent's, with an
                instruction)
      "none"    no guidance at all -- Pure-Planner
      "subgoals" unguided A* run leg by leg between the scene graph's
                area chain -- the hierarchical decomposition baseline
      "straight" no search: the grounded start and goal joined directly,
                then checked against the same gate

    ``fields`` is the map the method plans on and post-processes with --
    all it knows. ``true_fields`` is the embodiment model the result is
    *scored* against, and defaults to ``fields``. They differ only for the
    degraded-map baselines, and that difference is the whole point: a
    planar costmap can route a humanoid under a duct and never know.
    """
    import time

    # the caller pinned the destination; the agent must not move it
    goal_pinned = goal_id is not None
    # the reachable set and search lattice depend only on (fields, graph,
    # robot), so a batch builds them once per platform
    cache = {} if cache is None else cache
    t0 = time.time()
    if fields is None:
        # the obstacle model depends only on (grid, robot), so a batch can
        # build it once per platform instead of once per run
        primitives = build_collision_primitives(graph, grid, robot)
        fields = compute_clearance_fields(
            grid, primitives, robot, pitch=pitch,
            floor_z=float(prior["floor_z"]) if prior else None, free_prior=prior)
    t_fields = time.time() - t0

    planner = ValueGuidedAStar(
        PlannerParams(eta=float(config["planning"]["eta"] if eta is None else eta)),
        robot)
    # The reachable set and the search lattice depend only on (fields,
    # robot), so a batch of 400 runs should build them twice, not 400
    # times. Keyed by the map, because the baselines plan on different
    # ones and must not inherit BodyNav's.
    ck = (id(fields), robot.name)
    if ("reach", ck) in cache:
        reach = cache[("reach", ck)]
    else:
        reach = cache[("reach", ck)] = reachable_areas(planner, fields, graph)
    anchors = reach.get("anchors", {})
    if start_id is None or goal_id is None:
        picked = farthest_pair(graph, reach["areas"], anchors)
        if picked is None:
            return {"robot": robot.name, "success": False,
                    "failure_reason": "no two reachable areas",
                    "reachable_m2": round(reach["area_m2"], 1)}
        start_id = start_id or picked[0]
        goal_id = goal_id or picked[1]
    # An explicit endpoint wins over the node's centroid. A sampled task
    # names a *spot* -- a pallet, a bench -- and the robot cannot stand in
    # one; the sampler already resolved the cell it would stand in, and
    # re-deriving that here from the centroid would pick a different cell
    # than the one the task was certified reachable from.
    start_xyz = (np.asarray(start_xyz, dtype=float) if start_xyz is not None
                 else np.asarray(anchors.get(start_id,
                                             graph.attributes(start_id).centroid),
                                 dtype=float))
    goal_xyz = (np.asarray(goal_xyz, dtype=float) if goal_xyz is not None
                else np.asarray(anchors.get(goal_id,
                                            graph.attributes(goal_id).centroid),
                                dtype=float))
    true_fields = fields if true_fields is None else true_fields

    # value field on the search lattice (see the note in the CLI below)
    if ("lattice", ck) in cache:
        lattice, lattice_ijk = cache[("lattice", ck)]
    else:
        stand_k, passable = planner.occupiable_layer(fields)
        pi, pj = np.nonzero(passable)
        pk = np.clip(stand_k, 0, fields.occupied.shape[2] - 1)[pi, pj]
        lattice_ijk = np.column_stack([pi, pj, pk])
        centers = fields.world(lattice_ijk)
        lattice = GridArrays(
            origin=fields.origin, base_pitch=fields.pitch, index=lattice_ijk,
            center=centers, resolution=np.full(len(centers), fields.pitch),
            occupancy=np.ones(len(centers), dtype=int),
            semantic_label=np.full(len(centers), "free", dtype=object),
            spot_id=np.full(len(centers), None, dtype=object),
            synthetic=np.zeros(len(centers), dtype=bool),
            confidence=np.ones(len(centers)))
        cache[("lattice", ck)] = (lattice, lattice_ijk)

    vm = config["value_maps"]
    agent_info, maps, agent_failed, grounded_goal = None, None, None, []
    t1 = time.time()
    if value_maps == "none":
        maps = None
    elif instruction:
        try:
            if (replay_code is not None or deterministic_fields) and agent_pre is not None:
                maps, agent_info = compose_controlled(
                    instruction, graph, lattice, robot, fields, lattice_ijk, agent_pre,
                    config, replay_code=replay_code, deterministic=deterministic_fields)
            else:
                maps, agent_info = compose_with_agent(
                    instruction, graph, lattice, robot,
                    backend=backend, offline=offline, resolved=agent_pre,
                    grounding=grounding)
        except Exception as exc:
            # The agent failing to emit schema-valid code is a *stage 3*
            # failure, not evidence that no path exists. Scoring it as a
            # planning failure conflates two things the paper reports
            # separately, and understates PSR by exactly the agent's error
            # rate. The run continues on the reference maps, and the
            # substitution is recorded so the agent's success rate can be
            # reported as its own number.
            agent_failed = f"{type(exc).__name__}: {exc}"[:200]
            maps, agent_info = None, None
        # The instruction has to decide *where the robot goes*, not merely
        # shape the field on the way to a default destination. Without
        # this the twelve instructions all planned the same route and the
        # evaluation measured nothing about instruction-following.
        if agent_info is not None:
            grounded_goal = (agent_info.get("ground", {}) or {}).get("goal_ids") or []
        elif agent_pre is not None:
            # Code generation failed, but grounding (Steps 1-2) had already
            # succeeded and the map and gate use it. Score the grounding it
            # did; agent_ok/agent_error record the code-generation failure.
            grounded_goal = list(agent_pre.goal_ids)
        if grounded_goal and not goal_pinned:
            centroid = np.asarray(graph.attributes(grounded_goal[0]).centroid,
                                  dtype=float)
            layer_k, passable = planner.occupiable_layer(fields)
            snapped = planner._snap_ground(centroid, fields, passable,
                                           radius_m=8.0)
            if snapped is not None:
                k = int(np.clip(layer_k[snapped], 0,
                                fields.occupied.shape[2] - 1))
                goal_xyz = fields.world(np.array([[snapped[0], snapped[1], k]]))[0]
                goal_id = grounded_goal[0]
                # the goal moved, so the attraction must follow it
                maps.metadata["grounded_goal"] = np.array([goal_id], dtype=object)
    if maps is None and value_maps != "none":
        floor_z = float(prior["floor_z"]) if prior else 0.0
        band_lo, band_hi = floor_z + robot.height_band[0], floor_z + robot.height_band[1]
        sigma_lo, sigma_hi = vm["sigma_h_range_m"]
        hazards = []
        for node_id in graph.find(hazardous=True):
            attrs = graph.attributes(node_id)
            cls = attrs.hazard.hazard_class if attrs.hazard else attrs.semantic
            if cls not in ALWAYS_HAZARD_CLASSES:
                continue
            if attrs.aabb_min[2] > band_hi or attrs.aabb_max[2] < band_lo:
                continue
            hazards.append((np.asarray(attrs.centroid, dtype=float),
                            float(attrs.hazard.risk_weight),
                            float(np.clip(attrs.hazard.clearance_radius,
                                          sigma_lo, sigma_hi))))
        ijk = lattice_ijk
        maps = build_value_maps(
            lattice, goal_points=goal_xyz.reshape(1, 3), hazards=hazards,
            slope=fields.slope_grid[ijk[:, 0], ijk[:, 1]],
            width=fields.lateral_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * 2.0,
            priority=Priority(priority),
            params=ValueMapParams(sigma_g=vm["sigma_g_m"], theta_safe=vm["theta_safe"],
                                  alpha_s=vm["alpha_s"], alpha_w=vm["alpha_w"]),
            w_min=robot.min_lateral_clearance, smooth_sigma=vm["smooth_sigma"])
    t_values = time.time() - t1

    t2 = time.time()
    result_margin: dict = {}
    # Defined for every branch: the subgoal baseline never sets them,
    # and the post-processing gate below reads them unconditionally.
    forbidden, applied, result_margin = None, {}, {}

    if value_maps == "straight":
        # LLM-as-Planner: the agent proposes waypoints and straight
        # segments join them. Joining only start to goal would be a straw
        # man -- a 30 m straight line across a building is blocked by
        # construction -- so the waypoints are the scene graph's own route
        # through connected areas, which is the best case for the method.
        chain = subgoal_chain(graph, start_id, goal_id)
        pts = [start_xyz] + [np.asarray(anchors.get(
            n, graph.attributes(n).centroid), dtype=float)
            for n in chain[1:-1]] + [goal_xyz]
        raw = np.vstack(pts)
        blocked = [i for i, (a, b) in enumerate(zip(raw[:-1], raw[1:]))
                   if not _segment_ok(a, b, fields, planner.feasible)]
        planner.stats = {"expansions": 0,
                         "reason": f"{len(blocked)} of {len(raw) - 1} straight "
                                   f"segments blocked",
                         "waypoints_proposed": len(raw)}
        raw = None if blocked else raw
    elif value_maps == "subgoals":
        # D3 -- subgoal decomposition. The instruction is turned into a
        # chain of places (the scene graph's own area route to the
        # grounded destination) and an *unguided* A* connects consecutive
        # ones. This is the standard hierarchical arrangement: language
        # chooses waypoints, geometry connects them. What it cannot do is
        # let the instruction shape the route *between* waypoints, which
        # is the thing the value maps are for.
        chain = subgoal_chain(graph, start_id, goal_id)
        pts = [start_xyz] + [np.asarray(anchors.get(
            n, graph.attributes(n).centroid), dtype=float)
            for n in chain[1:-1]] + [goal_xyz]
        legs, expansions, failed_leg = [], 0, None
        for li, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            leg = planner.search(grid, None, fields, a, b)
            expansions += int(planner.stats.get("expansions", 0))
            if leg is None:
                failed_leg = li
                break
            legs.append(leg if not legs else leg[1:])
        if failed_leg is None and legs:
            raw = np.vstack(legs)
            planner.stats = {"expansions": expansions,
                             "subgoals": len(pts) - 2}
        else:
            raw = None
            planner.stats = {"expansions": expansions,
                             "subgoals": len(pts) - 2,
                             "reason": f"leg {failed_leg} of {len(pts) - 1} "
                                       f"has no feasible path"}
    else:
        # A stated margin is a constraint; enforce it, and fall back to the
        # soft field only when nothing satisfies it.
        # The margins come from grounding. When the agent's value-map code
        # is rejected, grounding (Steps 1-2) has still happened -- agent_pre
        # holds its ids and metres -- and a stated minimum must not be
        # dropped because a later, independent step failed. (Journal fix,
        # 2026-11-05: the earlier code gated only on a completed agent run.)
        margin_source, g_ = None, {}
        if agent_info is not None:
            margin_source, g_ = "agent", agent_info.get("ground", {}) or {}
        elif agent_pre is not None:
            margin_source = "pre_ground"
            g_ = {"hazard_ids": list(agent_pre.hazard_ids),
                  "hazard_margins": dict(agent_pre.hazard_margins)}
        if margin_source is not None and hard_gate:
            forbidden, applied = margin_constraint(
                fields, graph, g_.get("hazard_ids"),
                {k: float(v) + float(gate_margin_offset)
                 for k, v in (g_.get("hazard_margins") or {}).items()}, robot,
                float(prior["floor_z"]) if prior else 0.0,
                start_xyz=start_xyz, goal_xyz=goal_xyz)
        if forbidden is not None:
            raw = planner.search(grid, maps, fields, start_xyz, goal_xyz,
                                 forbidden=forbidden)
            result_margin = {"margins_enforced": applied,
                             "margin_fallback": raw is None,
                             "margin_source": margin_source}
            if raw is None:
                raw = planner.search(grid, maps, fields, start_xyz, goal_xyz)
        else:
            result_margin = {}
            raw = planner.search(grid, maps, fields, start_xyz, goal_xyz)
    t_search = time.time() - t2

    straight = float(np.linalg.norm(goal_xyz - start_xyz))
    result = {**result_margin, "hard_gate": bool(hard_gate),
        "gate_margin_offset_m": float(gate_margin_offset),
        "robot": robot.name, "start": start_id, "goal": goal_id,
        "priority": priority, "value_maps": value_maps, "seed": seed,
        "instruction": instruction,
        "reachable_m2": round(reach["area_m2"], 1),
        "straight_line_m": round(straight, 2),
        "t_fields_s": round(t_fields, 2), "t_values_s": round(t_values, 2),
        "t_plan_s": round(t_search, 3),
    }
    if instruction and value_maps != "none":
        # Reported separately from PSR: whether stage 3 produced usable
        # code, and whether it grounded the destination the task meant.
        result["agent_ok"] = agent_failed is None
        if agent_failed:
            result["agent_error"] = agent_failed
        result["grounded_goal"] = grounded_goal[0] if grounded_goal else None
        # Class, not instance. "Go to the clutter" does not identify one of
        # 78 clutter piles, so demanding the agent name the exact node the
        # sampler drew scores correct grounding as failure -- it read 5.6%
        # while the agent was resolving the right *kind* of thing almost
        # every time. The distance to the intended instance is recorded
        # alongside, which is the part that is genuinely a grounding error.
        if grounded_goal:
            ga = graph.attributes(grounded_goal[0])
            ta = graph.attributes(goal_id)
            result["goal_match"] = ga.semantic == ta.semantic
            result["goal_offset_m"] = round(float(np.linalg.norm(
                np.asarray(ga.centroid, float)[:2]
                - np.asarray(ta.centroid, float)[:2])), 2)
        else:
            result["goal_match"] = False
    if raw is None:
        result.update({"success": False,
                       "failure_reason": planner.stats.get("reason", "no path"),
                       "expansions": planner.stats.get("expansions", 0)})
        if agent_info:
            result["agent"] = {k: v for k, v in agent_info.items() if k != "trace"}
        return result

    # The search honoured the stated margin by refusing forbidden columns.
    # Line-of-sight shortcutting must honour it too: gated on clearance
    # alone it will happily replace a compliant detour with a straight
    # segment through the exclusion, and the result is a route that
    # violates the instruction its own planner satisfied.
    post_gate = planner.feasible
    if forbidden is not None and not result_margin.get("margin_fallback"):
        _forbidden = forbidden

        def post_gate(index, fields, _base=planner.feasible):
            i, j = int(index[0]), int(index[1])
            if (0 <= i < _forbidden.shape[0] and 0 <= j < _forbidden.shape[1]
                    and _forbidden[i, j]):
                return False
            return _base(index, fields)

    final = shortcut(raw, grid, fields, gate=post_gate)
    if smooth:
        smoothed = smooth_bspline(final, max_curvature=1.0 / max(robot.r_safety, 0.1))
        final = revalidate(smoothed, final, grid, fields, robot, gate=post_gate)
    metrics = path_metrics(final, true_fields, straight)

    # Safety compliance is scored on the *true* embodiment model, which
    # for BodyNav is the same model it planned on and for a degraded-map
    # baseline is not. A method is allowed to plan with less than the
    # truth; it is not allowed to be graded with less than the truth.
    from bodynav.eval.baselines import hazard_compliance, judge

    if hazard_category:
        # Did the path honour the hazard the instruction named? Reported
        # for every method, including those with no instruction input --
        # that contrast is the point.
        result["hazard_task"] = hazard_compliance(
            final, graph, robot, category=hazard_category,
            margin_m=hazard_margin_m, node_ids=hazard_node_ids,
            floor_z=float(prior["floor_z"]) if prior else 0.0)

    result.update({
        "success": True, "final": metrics,
        "expansions": planner.stats.get("expansions", 0),
        "waypoints": int(len(final)),
        "path": final.tolist(),
        **judge(final, true_fields, planner),
    })
    if agent_info:
        result["agent"] = {k: v for k, v in agent_info.items() if k != "trace"}
        result["agent_trace"] = agent_info.get("trace")
    if verbose:
        print(f"    {robot.name:11s} {value_maps:8s} {start_id} -> {goal_id}: "
              f"{metrics['length_m']:6.2f} m ({metrics['normalized_length']}x), "
              f"clear {metrics['min_clearance_m']:.2f} m, "
              f"{result['expansions']:,} exp, {t_search:.2f} s")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--robot", default="husky_ur5")
    ap.add_argument("--start", default=None, help="area node id (default: from stage 2)")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--priority", default="safety_first",
                    choices=[p.value for p in Priority])
    ap.add_argument("--pitch", type=float, default=0.1, help="planning lattice (m)")
    ap.add_argument("--out", type=Path, default=None, help="plan JSON")
    ap.add_argument("--path-out", type=Path, default=None, help="waypoints .npy")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--config-dir", type=Path, default=REPO / "configs" / "robots")
    ap.add_argument("--free-space", type=Path,
                    default=REPO / "data" / "labfloor" / "free_space.npz",
                    help="stage 1 traversable map; gates where the robot may stand")
    ap.add_argument("--instruction", default=None,
                    help="natural-language instruction; runs the stage-3 agent "
                         "instead of composing the value maps directly")
    ap.add_argument("--backend", default=None, help="llm-large | llm-mini")
    ap.add_argument("--offline", action="store_true",
                    help="force the scripted backend (no API calls)")
    ap.add_argument("--grounding", choices=("lexical", "llm"), default="lexical",
                    help="lexical: synonym table + token overlap; llm: the model maps "
                         "each phrase onto one of the scene's categories")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    robot = load_profile(args.robot, config_dir=args.config_dir)
    graph = BodyNavSceneGraph.load(str(args.graph))
    grid = load_grid_arrays(args.grid)
    print(f"grid {len(grid):,} voxels | graph {graph.g.number_of_nodes()} nodes "
          f"| robot {robot.name}")

    t0 = time.time()
    prior = None
    if args.free_space and args.free_space.exists():
        with np.load(args.free_space) as fp:
            prior = {k: fp[k] for k in fp.files}
        print(f"free-space prior {prior['free'].sum() * float(prior['cell']) ** 2:.0f} m2"
              f" from stage 1")
    primitives = build_collision_primitives(graph, grid, robot)
    fields = compute_clearance_fields(grid, primitives, robot, pitch=args.pitch,
                                      floor_z=float(prior["floor_z"]) if prior else None,
                                      free_prior=prior)
    t_fields = time.time() - t0
    print(f"obstacles  {len(primitives)} primitives, "
          f"{fields.meta['obstacle_cells']:,} blocked cells of "
          f"{np.prod(fields.meta['dims']):,} ({t_fields:.1f}s)")

    planner = ValueGuidedAStar(PlannerParams(eta=config["planning"]["eta"]), robot)

    # Which of the floor is reachable *for this body*. The free space is
    # one connected 500 m2 component, but a corridor obstructed to 0.50 m
    # by stored material admits neither platform, so the reachable region
    # is smaller and platform-specific. Tasks are drawn from it, and its
    # size is reported: a start/goal pair across the obstruction is
    # infeasible by construction, and calling that a planner failure would
    # be a measurement error.
    reach = reachable_areas(planner, fields, graph)
    print(f"reachable  {reach['area_m2']:.0f} m2 of "
          f"{prior['free'].sum() * float(prior['cell']) ** 2:.0f} m2 free "
          f"({len(reach['areas'])} of {len(graph.nodes_by_level('area'))} areas)")

    start_id, goal_id = args.start, args.goal
    if start_id is None or goal_id is None:
        picked = farthest_pair(graph, reach["areas"], reach.get("anchors"))
        if picked is None:
            raise SystemExit("no two reachable areas -- nothing to plan between")
        start_id = start_id or picked[0]
        goal_id = goal_id or picked[1]
    anchors = reach.get("anchors", {})
    start_xyz = np.asarray(anchors.get(
        start_id, graph.attributes(start_id).centroid), dtype=float)
    goal_xyz = np.asarray(anchors.get(
        goal_id, graph.attributes(goal_id).centroid), dtype=float)
    print(f"task  {start_id} -> {goal_id}  "
          f"({np.linalg.norm(goal_xyz - start_xyz):.1f} m apart)")


    # value maps: goal entity, hazard nodes, and the measured fields
    vm = config["value_maps"]
    params = ValueMapParams(sigma_g=vm["sigma_g_m"], theta_safe=vm["theta_safe"],
                            alpha_s=vm["alpha_s"], alpha_w=vm["alpha_w"])
    sigma_lo, sigma_hi = vm["sigma_h_range_m"]
    # A hazard repels laterally only where the body actually meets it.
    # A duct at 2.3 m is a clearance constraint for a 2.0 m humanoid and
    # nothing at all for a 1.25 m wheeled base -- and with 126 ducts on
    # this floor, counting them all as lateral hazards drove every cell of
    # V* negative and drowned the goal attraction entirely.
    floor_z = float(prior["floor_z"]) if prior else 0.0
    band_lo = floor_z + robot.height_band[0]
    band_hi = floor_z + robot.height_band[1]
    from bodynav.llm.grounding import ALWAYS_HAZARD_CLASSES

    hazards, skipped = [], 0
    for node_id in graph.find(hazardous=True):
        attrs = graph.attributes(node_id)
        cls = attrs.hazard.hazard_class if attrs.hazard else attrs.semantic
        # without an instruction there is nothing named, so the field
        # carries the always-on classes only -- the same policy the agent
        # applies, minus the names it would have added
        if cls not in ALWAYS_HAZARD_CLASSES:
            skipped += 1
            continue
        if attrs.aabb_min[2] > band_hi or attrs.aabb_max[2] < band_lo:
            skipped += 1
            continue
        weight = float(attrs.hazard.risk_weight)
        sigma = float(np.clip(attrs.hazard.clearance_radius, sigma_lo, sigma_hi))
        hazards.append((np.asarray(attrs.centroid, dtype=float), weight, sigma))

    # The value field is evaluated on the *search lattice*, not only on
    # occupied voxels. The maps are defined over the grid, but a grid of
    # surface voxels carries no value in the free space the robot actually
    # travels through -- sampled there, V* is zero almost everywhere and
    # the guidance in Eq. 10 does nothing.
    stand_k, passable = planner.occupiable_layer(fields)
    pi, pj = np.nonzero(passable)
    pk = np.clip(stand_k, 0, fields.occupied.shape[2] - 1)[pi, pj]
    lattice_ijk = np.column_stack([pi, pj, pk])
    centers = fields.world(lattice_ijk)
    lattice = GridArrays(
        origin=fields.origin, base_pitch=fields.pitch, index=lattice_ijk,
        center=centers, resolution=np.full(len(centers), fields.pitch),
        occupancy=np.ones(len(centers), dtype=int),
        semantic_label=np.full(len(centers), "free", dtype=object),
        spot_id=np.full(len(centers), None, dtype=object),
        synthetic=np.zeros(len(centers), dtype=bool),
        confidence=np.ones(len(centers)),
    )
    ijk = lattice_ijk
    t1 = time.time()
    agent_info = None
    if args.instruction:
        maps, agent_info = compose_with_agent(
            args.instruction, graph, lattice, robot,
            backend=args.backend, offline=args.offline, grounding=args.grounding)
    else:
        maps = build_value_maps(
        lattice,
        goal_points=goal_xyz.reshape(1, 3),
        hazards=hazards,
        slope=fields.slope_grid[ijk[:, 0], ijk[:, 1]],
        width=fields.lateral_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * 2.0,
        priority=Priority(args.priority),
        params=params,
        w_min=robot.min_lateral_clearance,
        smooth_sigma=vm["smooth_sigma"],
        )
    t_values = time.time() - t1
    print(f"value maps {len(hazards)} hazards in the {robot.height_band[0]:.2f}-"
          f"{robot.height_band[1]:.2f} m band ({skipped} above it), V* in "
          f"[{maps.fused.min():.2f}, {maps.fused.max():.2f}] ({t_values:.1f}s)")

    t2 = time.time()
    raw = planner.search(grid, maps, fields, start_xyz, goal_xyz)
    t_search = time.time() - t2

    if raw is None:
        print(f"NO PATH ({planner.stats.get('reason')}, "
              f"{planner.stats.get('expansions', 0):,} expansions, {t_search:.1f}s)")
        result = {"robot": robot.name, "success": False,
                  "failure_reason": planner.stats.get("reason"),
                  "expansions": planner.stats.get("expansions", 0),
                  "t_plan_s": round(t_search, 3)}
    else:
        straight = float(np.linalg.norm(goal_xyz - start_xyz))
        before = path_metrics(raw, fields, straight)
        gate = planner.feasible
        short = shortcut(raw, grid, fields, gate=gate)
        final = short
        if not args.no_smooth:
            smoothed = smooth_bspline(short, max_curvature=1.0 / max(robot.r_safety, 0.1))
            final = revalidate(smoothed, short, grid, fields, robot, gate=gate)
        after = path_metrics(final, fields, straight)
        t_total = time.time() - t0
        print(f"path found  {planner.stats['expansions']:,} expansions, "
              f"{t_search:.1f}s search, {t_total:.1f}s total")
        print(f"  raw       {before['length_m']} m, {before['waypoints']} waypoints, "
              f"min clearance {before['min_clearance_m']} m")
        print(f"  final     {after['length_m']} m, {after['waypoints']} waypoints, "
              f"min clearance {after['min_clearance_m']} m, "
              f"normalized {after['normalized_length']}")
        result = {
            "robot": robot.name, "success": True,
            "start": start_id, "goal": goal_id, "priority": args.priority,
            "raw": before, "final": after,
            "smoothed": bool(not args.no_smooth and len(final) != len(short)),
            "expansions": planner.stats["expansions"],
            "search_cost": round(planner.stats["cost"], 2),
            "t_fields_s": round(t_fields, 2),
            "t_values_s": round(t_values, 2),
            "t_plan_s": round(t_search, 3),
            "t_total_s": round(t_total, 2),
            "lattice_pitch_m": args.pitch,
            "reachable_m2": round(reach["area_m2"], 1),
            "reachable_areas": len(reach["areas"]),
            "n_primitives": len(primitives),
            "n_hazards": len(hazards),
            "n_hazards_above_band": skipped,
            "agent": ({k: v for k, v in agent_info.items() if k != "trace"}
                      if agent_info else None),
        }
        result.update({f"final_{k}": v for k, v in after.items()})
        if args.path_out:
            args.path_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.path_out, final)
            # the field the planner minimised over, and where it lives
            np.save(args.path_out.with_name(f"values_{robot.name}.npy"), maps.fused)
            np.save(args.path_out.with_name(f"value_centers_{robot.name}.npy"), centers)
            np.savez_compressed(
                args.path_out.with_name(f"value_components_{robot.name}.npz"),
                affordance=maps.affordance, hazard=maps.hazard,
                constraint=maps.constraint, fused=maps.fused, centers=centers,
                instruction=np.array(args.instruction or "", dtype=object),
            )
            print(f"wrote {args.path_out}")

    if args.out and agent_info:
        trace_path = args.out.with_name(args.out.stem + "_trace.json")
        trace_path.write_text(json.dumps(agent_info, indent=1) + "\n")
        print(f"wrote {trace_path}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
