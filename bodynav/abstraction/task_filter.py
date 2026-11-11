"""Task-guided filtering (Sec. III-B.2).

Grounded entities (start, goal, waypoints, hazards) define:
  * a task corridor — centerline s with width w(s) that expands near
    hazard/equipment nodes and contracts in open space;
  * hazard exclusion volumes from recorded clearance attributes.

Their union is the mask V_t over the sliced cloud — the concrete
interface between semantics and resolution: grounded semantics decide
*where* density must be preserved, adaptive voxelization decides *how
finely*. A priority merge keeps 1--2 cm spacing in safety-critical /
task-essential regions and downsamples static background to 5% density;
scene-graph labels stay attached per point so geometry remains
semantically addressable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints


@dataclass
class TaskCorridor:
    centerline: np.ndarray  # (S, 3) polyline through grounded waypoints
    width: np.ndarray  # (S,) w(s), meters


def _node_xy(graph: BodyNavSceneGraph, node_id: str) -> np.ndarray:
    return np.asarray(graph.attributes(node_id).centroid, dtype=float)[:2]


def route_between(graph: BodyNavSceneGraph, start_id: str, goal_id: str) -> list[str]:
    """Shortest chain of same-level nodes following adjacency edges.

    A straight line between two grounded entities runs through whatever
    happens to lie between them -- on a corridor loop, that is the building
    itself. Following adjacency keeps the corridor inside space the scene
    graph says is connected, which is what makes V_t a statement about the
    route rather than about the bounding box of the route.
    """
    import networkx as nx

    keep = [
        (u, v) for u, v, d in graph.g.edges(data=True)
        if d.get("kind") == "adjacency"
    ]
    sub = nx.Graph()
    sub.add_nodes_from(graph.g.nodes)
    for u, v in keep:
        cu = graph.attributes(u).centroid
        cv = graph.attributes(v).centroid
        sub.add_edge(u, v, weight=float(np.linalg.norm(cu - cv)))
    try:
        return nx.shortest_path(sub, start_id, goal_id, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # Adjacency edges only join nodes of the same level, so an area-to-spot
    # request finds nothing and used to fall back to the straight line
    # between them -- which on a corridor loop runs through the building.
    # The corridor decides where the map stays dense, so putting it through
    # walls leaves the real route coarse: three of nine sampled tasks lost
    # their route to that. Route between areas instead, hopping to the
    # goal's container through its containment edge.
    def container(node_id: str) -> str | None:
        if node_id.startswith("area_"):
            return node_id
        parents = [v for v, d in graph.g[node_id].items()
                   if d.get("kind") == "containment" and v.startswith("area_")]
        return parents[0] if parents else None

    a, b = container(start_id), container(goal_id)
    if a is None or b is None:
        return [start_id, goal_id]
    try:
        mid = nx.shortest_path(sub, a, b, weight="weight") if a != b else [a]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [start_id, goal_id]
    chain = ([start_id] if start_id != a else []) + mid \
        + ([goal_id] if goal_id != b else [])
    return chain


def build_corridor(
    graph: BodyNavSceneGraph, start_id: str, goal_id: str, waypoint_ids: list[str],
    *, base_width: float = 2.0, hazard_expand: float = 1.5,
    samples_per_m: float = 2.0, route: bool = True,
) -> TaskCorridor:
    """Polyline through the grounded entities, widened near hazards.

    Width is a function of *proximity to hazard nodes*, not of curvature
    or of free space: the corridor is a statement about where the map must
    stay dense, and the places that must stay dense are the places where
    getting the geometry wrong is dangerous. In open, benign stretches it
    contracts to ``base_width`` and the background is free to coarsen.
    """
    anchors = [start_id, *waypoint_ids, goal_id]
    if route:
        chained: list[str] = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            leg = route_between(graph, a, b)
            chained.extend(leg[:-1])
        chained.append(anchors[-1])
        anchors = chained
    pts = np.array([graph.attributes(a).centroid for a in anchors], dtype=float)

    # resample the polyline so width varies smoothly along it
    segs = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.linalg.norm(b - a) * samples_per_m), 1)
        segs.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    line = np.asarray(segs, dtype=float)

    hazards = [
        _node_xy(graph, n) for n in graph.find(hazardous=True)
    ]
    width = np.full(len(line), float(base_width))
    if hazards:
        haz = np.asarray(hazards)
        for i, p in enumerate(line):
            d = float(np.linalg.norm(haz - p[:2], axis=1).min())
            # expand smoothly within 3 m of a hazard
            width[i] += hazard_expand * max(0.0, 1.0 - d / 3.0)
    return TaskCorridor(centerline=line, width=width)


def task_mask(
    points: LabeledPoints, corridor: TaskCorridor, hazard_ids: list[str],
    graph: BodyNavSceneGraph,
) -> np.ndarray:
    """Boolean V_t over the sliced cloud (corridor u hazard volumes)."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(points.xyz, dtype=float)
    mask = np.zeros(len(xyz), dtype=bool)
    if not len(xyz):
        return mask

    # corridor: distance to the centreline in plan, against local width
    line = corridor.centerline
    tree = cKDTree(line[:, :2])
    dist, idx = tree.query(xyz[:, :2], k=1)
    mask |= dist <= (corridor.width[idx] / 2.0)

    # hazard volumes: the recorded clearance radius of each hazard node
    for node_id in hazard_ids:
        attrs = graph.attributes(node_id)
        if attrs.hazard is None:
            continue
        centre = np.asarray(attrs.centroid, dtype=float)
        r = float(attrs.hazard.clearance_radius)
        in_plan = np.linalg.norm(xyz[:, :2] - centre[:2], axis=1) <= r
        in_z = (xyz[:, 2] >= attrs.aabb_min[2] - r) & (xyz[:, 2] <= attrs.aabb_max[2] + r)
        mask |= in_plan & in_z
    return mask


def priority_merge(
    points: LabeledPoints, mask: np.ndarray,
    *, critical_spacing: float = 0.02, background_spacing: float = 0.05,
    seed: int = 0,
) -> LabeledPoints:
    """Keep fine spacing inside V_t, thin the background, preserve labels.

    Both regions are thinned by *voxel-grid decimation* -- one
    representative per cell -- inside the mask at ``critical_spacing`` and
    outside it at the coarser ``background_spacing``. Surfaces keep their
    shape in both.

    This used to keep a random 5% of background points instead, and the
    difference is not a detail. Ground height and slope are estimated from
    the floor, and a random twentieth of a floor is not a floor: it has
    holes, its local minima jump between samples, and the gate reads that
    as terrain. On the lab floor it cost the H1 every route it had --
    columns failing on lateral clearance, headroom and slope at once --
    while a uniform map of the same cloud kept them. Random subsampling
    thins a surface everywhere except where you need it.
    """
    xyz = np.asarray(points.xyz, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    # `seed` is retained for API compatibility: with both regions thinned by
    # grid decimation the result no longer depends on it.
    del seed

    keep = np.zeros(len(xyz), dtype=bool)

    inside = np.flatnonzero(mask)
    if len(inside):
        # one representative per critical-spacing cell
        cell = np.floor((xyz[inside] - xyz[inside].min(axis=0)) / critical_spacing)
        _, first = np.unique(cell, axis=0, return_index=True)
        keep[inside[first]] = True

    outside = np.flatnonzero(~mask)
    if len(outside):
        cell = np.floor((xyz[outside] - xyz[outside].min(axis=0)) / background_spacing)
        _, first = np.unique(cell, axis=0, return_index=True)
        keep[outside[first]] = True

    out = points.select(keep)
    out.extras = dict(out.extras)
    out.extras["v_t"] = mask[keep]
    return out
