"""Path post-processing (Sec. III-E.2).

1. Line-of-sight shortcutting — a segment replaces intermediate waypoints
   only if every traversed voxel remains feasible.
2. Curvature-bounded cubic B-spline smoothing.
3. Re-validation — the smoothed trajectory is re-sampled at voxel
   resolution against the same masks and margins; rejected operations
   restore the original feasible sub-path.

All path-length and clearance metrics are computed on the accepted
post-processed trajectory.
"""

from __future__ import annotations

import numpy as np

from bodynav.core.types import VoxelGrid
from bodynav.planning.obstacles import ClearanceFields
from bodynav.robots.profiles import RobotProfile


def _walk(a: np.ndarray, b: np.ndarray, fields: ClearanceFields) -> np.ndarray:
    """Every lattice cell the segment a -> b passes through.

    A 3-D DDA rather than point sampling. Sampling a segment at some fixed
    interval skips cells it clips the corner of, so a shortcut can be
    accepted whose swept path crosses an obstacle no sample landed in --
    the check then certifies a path that is not feasible, which is the one
    failure mode post-processing exists to prevent.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    start = fields.index(a.reshape(1, 3))[0]
    end = fields.index(b.reshape(1, 3))[0]

    direction = b - a
    cell = np.array(start, dtype=int)
    cells = [cell.copy()]
    step = np.where(direction >= 0, 1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_delta = np.abs(fields.pitch / direction)
        # distance along the ray to the first boundary in each axis
        nxt = fields.origin + (cell + (step > 0)) * fields.pitch
        t_max = np.abs((nxt - a) / direction)
    t_delta[~np.isfinite(t_delta)] = np.inf
    t_max[~np.isfinite(t_max)] = np.inf

    guard = int(np.sum(np.abs(end - start))) + 8
    for _ in range(guard):
        if np.array_equal(cell, end):
            break
        axis = int(np.argmin(t_max))
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        cells.append(cell.copy())
    return np.array(cells)


def _segment_ok(a, b, fields, gate) -> bool:
    return all(gate(tuple(int(v) for v in c), fields) for c in _walk(a, b, fields))


def shortcut(path: np.ndarray, grid: VoxelGrid, fields: ClearanceFields,
             gate=None) -> np.ndarray:
    """Line-of-sight shortcutting: drop waypoints a straight segment can
    replace, but only where every traversed cell still passes the gate.

    Greedy from each anchor to the furthest reachable waypoint, so one pass
    removes the staircase that 26-connected search leaves on open floor.
    """
    path = np.asarray(path, dtype=float).reshape(-1, 3)
    if gate is None or len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _segment_ok(path[i], path[j], fields, gate):
            j -= 1
        out.append(path[j])
        i = j
    return np.asarray(out)


def smooth_bspline(path: np.ndarray, max_curvature: float) -> np.ndarray:
    """Curvature-bounded cubic B-spline through the waypoints.

    Returns the input unchanged when there are too few points to fit, or
    when the fit would exceed ``max_curvature`` -- the caller then keeps
    the piecewise-linear path, which is feasible by construction.
    """
    from scipy.interpolate import splev, splprep

    path = np.asarray(path, dtype=float).reshape(-1, 3)
    if len(path) < 4:
        return path
    try:
        tck, _ = splprep(path.T, s=0.0, k=3)
    except (TypeError, ValueError):
        return path
    u = np.linspace(0.0, 1.0, max(len(path) * 6, 24))
    smooth = np.asarray(splev(u, tck)).T

    d1 = np.asarray(splev(u, tck, der=1)).T
    d2 = np.asarray(splev(u, tck, der=2)).T
    speed = np.linalg.norm(d1, axis=1)
    ok = speed > 1e-9
    curv = np.zeros(len(u))
    curv[ok] = (np.linalg.norm(np.cross(d1[ok], d2[ok]), axis=1)
                / speed[ok] ** 3)
    if float(np.max(curv)) > max_curvature:
        return path
    return smooth


def revalidate(
    smoothed: np.ndarray,
    original: np.ndarray,
    grid: VoxelGrid,
    fields: ClearanceFields,
    robot: RobotProfile,
    gate=None,
) -> np.ndarray:
    """Accept the smoothed trajectory only where it stays feasible;
    restore original sub-paths elsewhere."""
    smoothed = np.asarray(smoothed, dtype=float).reshape(-1, 3)
    if gate is None or len(smoothed) < 2:
        return np.asarray(original, dtype=float).reshape(-1, 3)
    for a, b in zip(smoothed[:-1], smoothed[1:]):
        if not _segment_ok(a, b, fields, gate):
            return np.asarray(original, dtype=float).reshape(-1, 3)
    return smoothed
