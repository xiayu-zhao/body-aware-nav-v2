"""Degraded maps for the comparison methods (D1, D2).

Every method in the evaluation runs through the *same* planner and is
judged by the *same* gate. What separates them is the map they are given
to plan on, which is the variable the paper is actually about. Building
the baselines as degraded copies of BodyNav's own obstacle model -- rather
than as separate implementations -- is what makes that a controlled
comparison: a baseline cannot lose because its A* was worse, because it
is not a different A*.

The judging is deliberately asymmetric. A method plans on its own map
and post-processes with its own gate, because that is all it knows; the
resulting trajectory is then scored against the true embodiment model.
A 2D costmap that routes a 1.8 m humanoid under a 1.5 m duct has
succeeded by its own lights and failed in the building, and the metric
has to be able to say so.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from bodynav.planning.obstacles import ClearanceFields

#: Mounting band of a horizontal 2D lidar on a mobile base. This is what
#: makes the costmap baseline a *2D* method rather than a differently
#: parameterised 3D one: it sees a horizontal slab and nothing else.
SENSOR_BAND_M = (0.15, 0.55)


def costmap_2d_fields(fields: ClearanceFields, *,
                      sensor_band_m: tuple[float, float] = SENSOR_BAND_M,
                      min_voxel_evidence: int = 2) -> ClearanceFields:
    """D1 -- planar costmap from a horizontal sensor slice.

    The classical stack: project what a horizontal scanner returns onto
    the floor plane, inflate it by the robot's radius, search. It is given
    BodyNav's own inflation radius and BodyNav's free-space prior, so it is not
    handicapped on either; the single thing it lacks is the third
    dimension. Overhead structure is therefore invisible (``vertical`` is
    unbounded) and so is terrain (``slope`` is flat) -- the two things
    Eq. 2's height band exists to supply.
    """
    from scipy import ndimage

    lo, pitch = np.asarray(fields.origin, float), float(fields.pitch)
    dims = fields.occupied.shape
    floor_z = float((fields.meta or {}).get("floor_z", lo[2]))

    k_lo = int(np.clip((floor_z + sensor_band_m[0] - lo[2]) / pitch, 0, dims[2] - 1))
    k_hi = int(np.clip((floor_z + sensor_band_m[1] - lo[2]) / pitch, 0, dims[2] - 1))
    counts = fields.occupied[:, :, k_lo:k_hi + 1].sum(axis=2)
    obstacle_2d = counts >= max(int(min_voxel_evidence), 1)
    # Walls: a 2D SLAM run maps these too, so withholding them would be a
    # straw man rather than a baseline.
    obstacle_2d = obstacle_2d | _walls(fields)

    lateral_2d = ndimage.distance_transform_edt(~obstacle_2d, sampling=pitch)
    meta = dict(fields.meta or {})
    meta.update(map_model="costmap_2d", sensor_band_m=list(sensor_band_m),
                lateral_cells_free=int((~obstacle_2d).sum()))
    return dataclasses.replace(
        fields,
        obstacle_2d=obstacle_2d,
        lateral_grid=np.repeat(lateral_2d[:, :, None], dims[2], axis=2),
        vertical_grid=np.full(dims, np.inf),
        slope_grid=np.zeros(dims[:2]),
        meta=meta,
    )


def judge(path: np.ndarray, true_fields: ClearanceFields, planner) -> dict:
    """Score a trajectory against the true embodiment model.

    ``clearance_ok`` and ``hazard_ok`` are the manuscript's SCR terms.
    ``vertical_ok`` is the third one the height band buys, and it is
    reported separately because it is the term a planar map cannot even
    evaluate -- for BodyNav it is true by construction, since the gate
    enforced it before the cell entered the open set.
    """
    ijk = np.clip(true_fields.index(path), 0,
                  np.array(true_fields.occupied.shape) - 1)
    lat = true_fields.lateral_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    vert = true_fields.vertical_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    haz = true_fields.hazard_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    required = planner.clearance_required(true_fields)
    need_v = planner.robot.required_vertical_clearance
    return {
        "clearance_ok": bool(np.all(lat >= required - 1e-6)),
        "vertical_ok": bool(np.all(vert >= need_v - 1e-6)),
        "hazard_ok": bool(np.all(haz[np.isfinite(haz)] >= 0.0)),
        "worst_lateral_m": round(float(lat.min()), 3),
        "worst_vertical_m": (round(float(vert.min()), 3)
                             if np.isfinite(vert).any() else None),
        "clearance_deficit_m": round(float(max(required - lat.min(), 0.0)), 3),
        "vertical_deficit_m": round(float(max(need_v - vert.min(), 0.0)), 3),
    }



def _densify(xy: np.ndarray, step: float = 0.02) -> np.ndarray:
    """Resample a polyline at a fixed spacing, vertices preserved.

    ``step`` is well below the 0.1 m lattice, so the sampled minimum is
    the segment minimum to within a millimetre.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return xy
    si = np.arange(0.0, s[-1] + step, step)
    return np.column_stack([np.interp(si, s, xy[:, 0]),
                            np.interp(si, s, xy[:, 1])])


def hazard_compliance(path: np.ndarray, graph, robot, *, category: str,
                      margin_m: float | None, node_ids=(),
                      floor_z: float = 0.0) -> dict:
    """Did the trajectory honour the hazard the *instruction* named?

    PSR and SCR are geometric: they ask whether a route exists and whether
    a body fits along it. Neither asks the question BodyNav exists to answer
    -- whether the robot did what it was told. A 2D costmap and a uniform
    voxel map have no instruction input at all, so on this metric they can
    only score by accident, and that is the honest way to show what the
    value maps buy.

    Distance is measured to the hazard instances' bounding boxes, not
    their centroids, and only to those intersecting the robot's own band:
    a duct 2.4 m up is a headroom constraint, not something to walk
    around, and counting it would make every path non-compliant for
    reasons that have nothing to do with the instruction.

    ``min_distance_m`` is the graded number and the one to report; the
    binary ``compliant`` flag is only meaningful where the corridor is
    wide enough to honour the margin at all, so it is reported alongside
    the distance rather than instead of it.
    """
    band_lo = floor_z + float(robot.height_band[0])
    band_hi = floor_z + float(robot.height_band[1])

    # The instances the task named -- those on the way -- not every
    # instance of the class on the floor. See `_pick_hazard`.
    candidates = list(node_ids) or [
        n for n in graph.nodes_by_level("spot")
        if (graph.attributes(n).hazard.hazard_class
            if graph.attributes(n).hazard else graph.attributes(n).semantic)
        == category]
    boxes = []
    for node_id in candidates:
        attrs = graph.attributes(node_id)
        if attrs.aabb_min[2] > band_hi or attrs.aabb_max[2] < band_lo:
            continue
        boxes.append((np.asarray(attrs.aabb_min, dtype=float)[:2],
                      np.asarray(attrs.aabb_max, dtype=float)[:2]))
    if not boxes:
        return {"hazard_category": category, "hazard_instances": 0,
                "min_distance_m": None, "compliant": None}

    # Densify before measuring. The accepted trajectory is shortcut and
    # smoothed down to a handful of vertices -- 2 to 8 across 20-35 m --
    # and measuring only at those asks where the robot *stopped*, not
    # where it went. A route can pass a hazard at 4 m between two
    # waypoints that are both 8 m away, and scoring it compliant is a
    # property of the sampling, not of the behaviour.
    xy = _densify(np.asarray(path, dtype=float)[:, :2], step=0.02)
    best = np.full(len(xy), np.inf)
    for lo, hi in boxes:
        # distance from each point to the axis-aligned box, 0 inside
        d = np.maximum(np.maximum(lo - xy, xy - hi), 0.0)
        best = np.minimum(best, np.linalg.norm(d, axis=1))
    nearest = float(best.min())
    return {
        "hazard_category": category,
        "hazard_instances": len(boxes),
        "min_distance_m": round(nearest, 3),
        # 1 mm, not 1 um: the lattice is 0.1 m, so a path that clears
        # 1.4999 m of a 1.5 m request has met it to any precision this
        # map can express, and scoring it as a violation measures
        # floating point rather than behaviour.
        "compliant": (bool(nearest >= margin_m - 1e-3)
                      if margin_m is not None else None),
        "hazard_margin_m": margin_m,
    }


#: Inflation radius of a generic small vehicle, in metres. The point of
#: this baseline is that it is *not* told which body will fly it.
GENERIC_INFLATION_M = 0.30


def _walls(fields: ClearanceFields) -> np.ndarray:
    """Columns no body may occupy, independent of which body is asking.

    Prefer stage 1's traversable map. Falling back to ``ground_k < 0``
    is right only for a body that stands on the floor: an aerial map's
    slice starts above it, so its floor index is negative everywhere and
    that test would call the whole site solid.
    """
    if getattr(fields, "prior_free", None) is not None:
        return ~np.asarray(fields.prior_free, dtype=bool)
    return fields.ground_k < 0


def costmap_3d_fields(fields: ClearanceFields, *,
                      inflation_m: float = GENERIC_INFLATION_M
                      ) -> ClearanceFields:
    """D1 for a body that flies -- an embodiment-blind 3D occupancy grid.

    The planar costmap ablates the third dimension, which is a fair
    question to ask of a ground robot and a meaningless one to ask of an
    aircraft: flying *is* the third dimension, and the 2D baseline simply
    returns nothing on two of the three sites. The conventional aerial
    stack is instead a 3D occupancy grid -- OctoMap and its descendants --
    inflated by a fixed radius and searched directly.

    So this map keeps all the geometry and all of the height, and drops
    only the conditioning: obstacles are taken over the whole column
    rather than a robot's operational band, clearance is measured against
    a generic vehicle envelope rather than the body that will fly it, and
    there is no terrain model. What remains is exactly what
    Sections~III-B and III-E add.
    """
    from scipy import ndimage

    pitch = float(fields.pitch)
    dims = fields.occupied.shape
    slab = max(int(round(inflation_m / pitch)), 1)

    # The structural walls, which the occlusion prior supplies and no
    # single voxel column does. Withholding them would not make this a
    # weaker map, it would make it a straw man: an occupancy grid built
    # from the same scan sees the same walls.
    walls = _walls(fields)

    lateral = np.empty(tuple(dims), dtype=float)
    for k in range(dims[2]):
        lo, hi = max(k - slab, 0), min(k + slab + 1, dims[2])
        near = fields.occupied[:, :, lo:hi].any(axis=2) | walls
        lateral[:, :, k] = ndimage.distance_transform_edt(~near, sampling=pitch)

    meta = dict(fields.meta or {})
    meta.update(map_model="costmap_3d", inflation_m=inflation_m)
    return dataclasses.replace(
        fields,
        lateral_grid=lateral,
        slope_grid=np.zeros(dims[:2]),   # an occupancy grid has no terrain
        meta=meta,
    )
