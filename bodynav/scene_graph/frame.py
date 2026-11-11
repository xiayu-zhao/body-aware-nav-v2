"""The building frame: axes along the structure, not along the world.

The lab-floor scan arrives rotated ~137 deg to the world axes, which costs
twice. Plan views render as diamonds that waste most of the page, and --
less obviously but more seriously -- every bird's-eye grid rasterizes the
walls as staircases. A diagonal wall on an axis-aligned lattice is a
4-connected barrier with 8-connected holes, so free space leaks through
walls or corridors pinch shut depending on the connectivity rule, and
neither answer is about the building.

Working in the building's own frame removes the artifact rather than
tuning around it. The transform is a rotation about the world Z axis, so
it is exactly invertible: results computed here map back to the USD stage
for Isaac Sim by :func:`from_building_frame`.
"""

from __future__ import annotations

import numpy as np


def principal_axis(
    xy: np.ndarray, *, max_points: int = 200_000, percentile: float = 0.5
) -> float:
    """Orientation of the building's dominant axis, in degrees.

    The angle that minimises the area of the enclosing rectangle: for a
    rectilinear building that rectangle is smallest when its sides lie
    along the walls.

    Two details decide whether this works on a real scan. Extents are
    taken at ``percentile`` rather than as min/max, so a single stray
    return through a window does not define the building; and the search
    is a sweep rather than the textbook rotating-calipers pass over the
    convex hull, because the hull is built from exactly the outliers the
    percentile is there to reject. Feed it floor points.
    """
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    if len(xy) > max_points:
        xy = xy[:: -(-len(xy) // max_points)]
    if len(xy) < 3:
        return 0.0

    def area_at(angle: float) -> float:
        t = np.radians(-angle)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        p = xy @ rot.T
        lo, hi = np.percentile(p, [percentile, 100.0 - percentile], axis=0)
        return float((hi[0] - lo[0]) * (hi[1] - lo[1]))

    coarse = min(np.arange(0.0, 90.0, 1.0), key=area_at)
    fine = min(np.arange(coarse - 1.0, coarse + 1.0, 0.05), key=area_at)
    # the rectangle's four sides are equivalent: report the smallest
    # rotation that aligns the building.
    return float((fine + 45.0) % 90.0 - 45.0)


def rotation(angle_deg: float) -> np.ndarray:
    """2x2 matrix taking world coordinates into the building frame."""
    t = np.radians(-angle_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def to_building_frame(xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (N, 3) world points about Z into the building frame."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.column_stack([xyz[:, :2] @ rotation(angle_deg).T, xyz[:, 2]])


def from_building_frame(xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Inverse of :func:`to_building_frame` -- back to the USD stage."""
    return to_building_frame(xyz, -angle_deg)
