"""Robot-specific extraction (Sec. III-B.1).

Vertical slice (Eq. 2): keep points with
    h_min <= p_z - z_gnd(p) <= h_max
where z_gnd is the *local* ground height (handles slopes). Unlike uniform
downsampling this yields 60--80% reduction while keeping full resolution
inside the navigation envelope; overhead structure is retained sparsely
for global awareness.

Workspace kernel (Eq. 3): full resolution within
    ||p_xy - p_rob,xy|| <= r_reach + r_safety
where manipulation-time collision checking matters.
"""

from __future__ import annotations

import numpy as np

from bodynav.robots.profiles import RobotProfile
from bodynav.scene_graph.hierarchy import LabeledPoints


def _cell_keys(xy: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearized 2D cell id per point, plus the grid dimensions."""
    idx = np.floor((xy - xy.min(axis=0)) / cell).astype(np.int64)
    dims = idx.max(axis=0) + 1
    return idx[:, 0] * dims[1] + idx[:, 1], dims


def local_ground_height(
    xyz: np.ndarray, cell: float = 0.5, *, quantile: float = 0.0
) -> np.ndarray:
    """Per-point local ground elevation z_gnd via 2D grid minima.

    Args:
        cell: footprint of the ground-estimation cell (m). Must be small
            enough to track slope, large enough to contain floor returns.
        quantile: within-cell height quantile used as ground; 0 is the
            strict minimum of Eq. 2. Any positive value (e.g. 0.02)
            rejects the isolated below-floor returns that LiDAR produces
            on reflective surfaces, which would otherwise drag the local
            ground down and lift the whole slice off the floor. The rank
            is rounded *up*, so a positive quantile always discards at
            least the single lowest return in a cell — rounding down would
            silently do nothing in the sparse cells where a lone outlier
            does the most damage.

    Returns:
        (N,) array of ground heights, one per input point.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        return np.empty(0, dtype=np.float64)
    if not 0.0 <= quantile < 1.0:
        raise ValueError(f"quantile must be in [0, 1), got {quantile}")

    keys, _ = _cell_keys(xyz[:, :2], cell)
    z = xyz[:, 2]

    # Sort by (cell, z) so each cell's heights are contiguous and ascending.
    order = np.lexsort((z, keys))
    keys_sorted = keys[order]
    z_sorted = z[order]

    starts = np.flatnonzero(
        np.concatenate(([True], keys_sorted[1:] != keys_sorted[:-1]))
    )
    counts = np.diff(np.concatenate((starts, [len(keys_sorted)])))
    offsets = np.minimum(
        np.ceil(quantile * (counts - 1)).astype(np.int64), counts - 1
    )
    ground_per_cell = z_sorted[starts + offsets]

    # Map each point back to its cell's ground height.
    cell_ids = keys_sorted[starts]
    return ground_per_cell[np.searchsorted(cell_ids, keys)]


def height_slice(
    points: LabeledPoints,
    robot: RobotProfile,
    *,
    overhead_keep_ratio: float = 0.05,
    ground_cell: float = 0.5,
    ground_quantile: float = 0.02,
    seed: int = 0,
) -> LabeledPoints:
    """Apply Eq. 2; keeps a sparse sample of overhead structure.

    Points inside the robot's vertical band are kept in full; points above
    it are subsampled to ``overhead_keep_ratio`` so ceilings, pipes, and
    cable trays still register for global awareness without paying full
    density. Points below the band (below-floor noise) are dropped.

    The per-point height above local ground is cached in
    ``extras['z_rel']`` because the voxelizer's phi_R needs it again.
    """
    xyz = np.asarray(points.xyz, dtype=np.float64)
    if len(xyz) == 0:
        return points.select(np.zeros(0, dtype=int))
    if not 0.0 <= overhead_keep_ratio <= 1.0:
        raise ValueError(
            f"overhead_keep_ratio must be in [0, 1], got {overhead_keep_ratio}"
        )

    z_gnd = local_ground_height(xyz, cell=ground_cell, quantile=ground_quantile)
    z_rel = xyz[:, 2] - z_gnd

    h_min, h_max = robot.height_band
    in_band = (z_rel >= h_min) & (z_rel <= h_max)

    keep = in_band.copy()
    if overhead_keep_ratio > 0.0:
        overhead = z_rel > h_max
        if overhead.any():
            rng = np.random.default_rng(seed)
            sample = rng.random(len(z_rel)) < overhead_keep_ratio
            keep |= overhead & sample

    sliced = points.select(keep)
    sliced.extras = dict(sliced.extras)
    sliced.extras["z_rel"] = z_rel[keep]
    sliced.extras["in_band"] = in_band[keep]
    return sliced


def workspace_kernel_mask(
    xyz: np.ndarray, robot_xy: np.ndarray, robot: RobotProfile,
    *, extra_radius: float = 0.0
) -> np.ndarray:
    """Boolean mask of points inside the workspace kernel (Eq. 3).

    ``robot_xy`` may be a single (2,) position or a (K, 2) trajectory of
    poses, in which case the kernel is the union over poses.

    ``extra_radius`` widens the kernel by a distance the *instruction* can
    displace the robot -- a stated hazard margin. Eq. 3's radius answers
    "how far from its path can this body reach", which is the right
    question only while the path is the one geometry would have chosen.
    Once an instruction says "stay 2 m clear", the robot may end up 2 m
    off that path, and the map has to be accurate where it actually goes.
    Without this the kernel clips the task mask to a 1.25 m tube around
    the nominal route and widening the corridor changes nothing at all.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        return np.zeros(0, dtype=bool)

    radius = robot.r_reach + robot.r_safety + max(float(extra_radius), 0.0)
    poses = np.atleast_2d(np.asarray(robot_xy, dtype=np.float64))[:, :2]
    if radius <= 0.0:
        # Platforms without a manipulator (the survey UGV, r_reach = 0)
        # still carry a safety envelope; a zero radius means no kernel.
        return np.zeros(len(xyz), dtype=bool)

    mask = np.zeros(len(xyz), dtype=bool)
    for pose in poses:
        d2 = np.sum((xyz[:, :2] - pose) ** 2, axis=1)
        mask |= d2 <= radius * radius
    return mask
