"""Obstacle and collision model (Sec. III-E.1).

Obstacles are occupied voxels whose label is in the non-traversable set
(structural, equipment, temporary structure, hazard classes); each uses
its parent spot's axis-aligned bounding volume as a conservative
collision primitive, inflated by r_safety and — for hazards — the
resolved clearance radius.

A voxel is infeasible if lateral clearance r_lat(v) (free-space distance
transform to the nearest inflated volume) < R + m, or vertical clearance
h_clear(v) < H_req.

Definitions (make violations well-defined functions of the voxel set):
  * safety violation — an executed step whose clearance to a hazard
    boundary is below that hazard's resolved margin;
  * incursion — a step entering the inflated volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodynav.core.types import VoxelGrid, as_arrays
from bodynav.robots.profiles import RobotProfile
from bodynav.scene_graph.graph import BodyNavSceneGraph

#: Semantic labels treated as non-traversable.
NON_TRAVERSABLE: frozenset[str] = frozenset(
    {"structural", "equipment", "temporary_structure", "hazard"}
)


#: Scene-graph labels whose spots become collision primitives. The
#: geometric vocabulary of the as-built pipeline is more specific than the
#: manuscript's four classes, so both spellings are accepted.
NON_TRAVERSABLE_LABELS: frozenset[str] = NON_TRAVERSABLE | frozenset(
    {"wall", "column", "equipment", "furniture", "stored_material", "clutter", "cone"}
)


@dataclass
class ClearanceFields:
    """Precomputed gate attributes.

    The declared per-voxel arrays (``indices`` and friends) are the
    contract the planner's gate reads. They are computed on a dense
    lattice, which is also kept: distance transforms are dense operations,
    and A* needs neighbour lookups that a sparse dict cannot answer in
    constant time. The lattice is the robot's planning resolution, not the
    voxel grid's -- adaptive voxels are the *map*, the lattice is the
    search space over it.
    """

    indices: np.ndarray  # (M, 3)
    lateral: np.ndarray  # (M,) r_lat, meters
    vertical: np.ndarray  # (M,) h_clear, meters
    slope: np.ndarray  # (M,) local height-gradient angle
    hazard_distance: np.ndarray  # (M,) distance to nearest hazard boundary

    # dense lattice (search space)
    obstacle_2d: np.ndarray = None  # (I, J) bool, what the gate blocks on
    band_k: tuple[int, int] = (0, 0)  # lattice layers of the robot's band
    origin: np.ndarray = None  # (3,) world coords of lattice corner
    pitch: float = 0.0
    occupied: np.ndarray = None  # (I, J, K) bool
    lateral_grid: np.ndarray = None  # (I, J, K) meters
    vertical_grid: np.ndarray = None  # (I, J, K) meters
    hazard_grid: np.ndarray = None  # (I, J, K) meters
    slope_grid: np.ndarray = None  # (I, J) radians
    ground_k: np.ndarray = None  # (I, J) int, standing layer; -1 = no floor
    #: (I, J) bool -- stage 1's traversable map resampled onto this
    #: lattice. Kept because it is the only record of where the *walls*
    #: are that does not depend on a body: `ground_k < 0` reads as "wall"
    #: for a ground robot and as "everywhere" for an aerial one, whose
    #: slice begins above the floor and whose floor index is negative.
    prior_free: np.ndarray = None
    meta: dict = None

    def world(self, ijk: np.ndarray) -> np.ndarray:
        """Lattice indices -> world-frame cell centres."""
        ijk = np.atleast_2d(np.asarray(ijk, dtype=float))
        return self.origin + (ijk + 0.5) * self.pitch

    def index(self, xyz: np.ndarray) -> np.ndarray:
        """World coords -> lattice indices (unclipped)."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        return np.floor((xyz - self.origin) / self.pitch).astype(int)


def _resample_prior(free_prior: dict, lo: np.ndarray, pitch: float,
                    dims: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resample of stage 1's free map onto the lattice."""
    free = np.asarray(free_prior["free"], dtype=bool)
    origin = np.asarray(free_prior["origin"], dtype=float)
    cell = float(free_prior["cell"])
    ii, jj = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
    world_x = lo[0] + (ii + 0.5) * pitch
    world_y = lo[1] + (jj + 0.5) * pitch
    si = np.floor((world_x - origin[0]) / cell).astype(int)
    sj = np.floor((world_y - origin[1]) / cell).astype(int)
    inside = ((si >= 0) & (si < free.shape[0]) & (sj >= 0) & (sj < free.shape[1]))
    out = np.zeros(tuple(dims[:2]), dtype=bool)
    out[inside] = free[si[inside], sj[inside]]
    return out


def _rasterize(centers: np.ndarray, resolution: np.ndarray, lo: np.ndarray,
               pitch: float, dims: np.ndarray) -> np.ndarray:
    """Mark every lattice cell a voxel's *extent* covers.

    A voxel stands for a cube of side r, and marking only the cell its
    centre falls in throws that away: at a 0.1 m lattice a 0.4 m voxel
    occupies 64 cells and was being recorded as one. The map that
    coarsens the most therefore looked like the map with the fewest
    obstacles, which inverted the comparison the paper is about -- the
    adaptive map called 105.9 m2 reachable for the H1 where a 2.5 cm map
    of the same cloud called it 29.8 m2, and the difference was almost
    entirely this.

    It also removes an accidental resolution dependence from the evidence
    threshold below: cells are the same size whatever tiled them, so
    counting them means the same thing in every map.
    """
    occupied = np.zeros(tuple(dims), dtype=bool)
    if not len(centers):
        return occupied
    for r in np.unique(resolution):
        sel = resolution == r
        half = float(r) / 2.0
        a = np.floor((centers[sel] - half - lo) / pitch).astype(int)
        b = np.ceil((centers[sel] + half - lo) / pitch).astype(int) - 1
        b = np.maximum(b, a)
        span = (b - a + 1).max(axis=0)
        for di in range(int(span[0])):
            for dj in range(int(span[1])):
                for dk in range(int(span[2])):
                    off = np.array((di, dj, dk))
                    # only the voxels whose own extent reaches this far --
                    # a group maximum applied to every member of the group
                    # would inflate the small ones to the size of the
                    # largest, which is the opposite of the bug being fixed
                    keep = np.all(a + off <= b, axis=1)
                    if not keep.any():
                        continue
                    c = np.clip(a[keep] + off, 0, dims - 1)
                    occupied[c[:, 0], c[:, 1], c[:, 2]] = True
    return occupied


def build_collision_primitives(
    graph: BodyNavSceneGraph, grid: VoxelGrid, robot: RobotProfile
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inflated AABBs (min, max) for every non-traversable spot.

    Each spot contributes its own bounding volume inflated by
    ``r_safety``; hazard-bearing spots are inflated by their *resolved*
    clearance radius instead, which is the attribute the scene graph
    carries for exactly this purpose. Using the spot's volume rather than
    its member voxels is deliberately conservative: a partially observed
    object still blocks the space it occupies.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for node_id in graph.nodes_by_level("spot"):
        attrs = graph.attributes(node_id)
        if attrs.hazard is not None:
            margin = float(attrs.hazard.clearance_radius)
        elif attrs.semantic in NON_TRAVERSABLE_LABELS:
            margin = float(robot.r_safety)
        else:
            continue
        lo = np.asarray(attrs.aabb_min, dtype=float) - margin
        hi = np.asarray(attrs.aabb_max, dtype=float) + margin
        out.append((lo, hi))
    return out


def compute_clearance_fields(
    grid: VoxelGrid,
    primitives,
    robot: RobotProfile,
    *,
    pitch: float = 0.1,
    graph: BodyNavSceneGraph | None = None,
    floor_z: float | None = None,
    free_prior: dict | None = None,
    min_voxel_evidence: int = 2,
) -> ClearanceFields:
    """Free-space distance transforms + slope field for feasibility gating.

    ``free_prior`` is stage 1's traversable map -- the occlusion-based free
    space, with shut doors opened. When supplied it defines lateral
    clearance and where the robot may stand. That consistency matters:
    re-deriving traversability here from individual voxel hits treats one
    stray return as a wall, and broke a continuous corridor into 30 m2
    fragments. The voxel grid still supplies everything vertical -- what
    is overhead, and how far the hazards are.
    """
    from scipy import ndimage

    arrays = as_arrays(grid)
    centers = arrays.center
    if not len(centers):
        raise ValueError("empty voxel grid")
    labels = arrays.semantic_label

    lo = centers.min(axis=0) - 2 * pitch
    hi = centers.max(axis=0) + 2 * pitch
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)

    ijk = np.clip(np.floor((centers - lo) / pitch).astype(int), 0, dims - 1)
    occupied = _rasterize(centers, arrays.resolution, lo, pitch, dims)

    # the floor is support, not an obstacle: a robot standing on it is not
    # in collision with it
    if floor_z is None:
        floor_z = float(np.percentile(centers[:, 2], 1.0))
    floor_layer = int(np.floor((floor_z + 0.06 - lo[2]) / pitch))
    ground_mask = labels == "floor"
    is_floor = (_rasterize(centers[ground_mask], arrays.resolution[ground_mask],
                           lo, pitch, dims) if ground_mask.any()
                else np.zeros(tuple(dims), dtype=bool))
    # The floor is support, not an obstacle, at every layer it appears in:
    # a 2.5 cm slab spans two cells of a 20 cm lattice, and treating the
    # upper one as an obstacle puts the robot inside a wall.
    obstacles = occupied & ~is_floor

    # hazard and non-traversable AABBs painted into the lattice
    hazard_vol = np.zeros(tuple(dims), dtype=bool)
    for prim_lo, prim_hi in primitives or []:
        a = np.clip(np.floor((prim_lo - lo) / pitch).astype(int), 0, dims - 1)
        b = np.clip(np.ceil((prim_hi - lo) / pitch).astype(int), 0, dims - 1)
        hazard_vol[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = True

    # r_lat is a *lateral* clearance, so it is measured in plan over the
    # obstacles that lie inside the robot's own band. A 3-D distance
    # transform answers a different question and answers it uselessly: the
    # nearest occupied cell to a robot standing on a floor is the floor,
    # one cell below, so every cell in the building reports ~0.2 m and the
    # gate rejects the entire floor.
    band_lo = float(robot.height_band[0]) + 0.10
    band_hi = float(robot.height_band[1])
    prior_2d = None
    if free_prior is not None:
        prior_2d = _resample_prior(free_prior, lo, pitch, dims)
    # The band is measured from the floor *under each column*, not from a
    # single plane through the building. A flat-plane band drifts wherever
    # the floor does, and then disagrees with the abstraction stage, which
    # measures its own band from local ground: structure the voxelizer
    # called "above the band" (and so tiled coarsely) lands inside the
    # band here, as a phantom obstacle at head height. On this floor that
    # cost three of nine tasks their route.
    ground_k = np.full(dims[:2], -1, dtype=int)
    ground_z_meas = np.full(dims[:2], -np.inf)
    if ground_mask.any():
        g = ijk[ground_mask]
        np.maximum.at(ground_k, (g[:, 0], g[:, 1]), g[:, 2])
        # The floor's *measured* height, not the index of the cell holding
        # it. Slope is computed from this below, and a height quantised to
        # the lattice cannot express an angle: two adjacent columns one
        # 0.1 m cell apart, 0.1 m apart in plan, are a 45 degree ramp. The
        # field was therefore 0 or 45 degrees and nothing between, which
        # put the H1's 20 degree limit inside the quantisation noise and
        # cost it three quarters of the floor.
        np.maximum.at(ground_z_meas, (g[:, 0], g[:, 1]),
                      centers[ground_mask][:, 2])
    ground_z_col = np.where(ground_k >= 0, lo[2] + (ground_k + 0.5) * pitch,
                            float(floor_z))
    k_lo = np.clip(((ground_z_col + band_lo - lo[2]) / pitch).astype(int), 0, dims[2] - 1)
    k_hi = np.clip(((ground_z_col + band_hi - lo[2]) / pitch).astype(int), 0, dims[2] - 1)
    k_axis = np.arange(dims[2])[None, None, :]
    in_band = (k_axis >= k_lo[:, :, None]) & (k_axis <= k_hi[:, :, None])
    # A single 2.5 cm voxel in a 10 cm cell is a stray return, not a
    # wall. Requiring two changes the Husky's reachable area from 24 to
    # 96 m2, and three or five change it no further -- the difference is
    # noise, and treating noise as obstacle closed most of the building.
    band_counts = (obstacles & in_band).sum(axis=2)
    obstacle_2d = band_counts >= max(int(min_voxel_evidence), 1)
    if prior_2d is not None:
        # Both, not either. The prior fills in where abstraction left the
        # floor sparse; the voxels are evidence that something is there.
        # Replacing the voxels with the prior let paths pass through
        # geometry that exists -- three waypoints ended up inside the
        # collision boxes the simulator was about to enforce.
        obstacle_2d = obstacle_2d | ~prior_2d
    lateral_2d = ndimage.distance_transform_edt(~obstacle_2d, sampling=pitch)

    if getattr(robot, "kind", "") == "aerial":
        # A flying body is thin. Collapsing the whole band into one plane
        # is right for a robot whose torso spans it -- a duct at 2.2 m is
        # a wall to a 1.85 m humanoid -- and wrong for one that can fly
        # under the duct: it would make the aerial platform strictly worse
        # than the ground ones, for a reason that is an artefact of the
        # representation rather than of the aircraft. So lateral clearance
        # is computed per height, over a slab the depth of the airframe
        # plus its required vertical margin.
        slab = max(int(round((robot.required_vertical_clearance)
                             / pitch)), 1)
        lateral_grid = np.empty(tuple(dims), dtype=float)
        for k in range(dims[2]):
            lo_k, hi_k = max(k - slab, 0), min(k + slab + 1, dims[2])
            near = obstacles[:, :, lo_k:hi_k].any(axis=2)
            if prior_2d is not None:
                near = near | ~prior_2d
            lateral_grid[:, :, k] = ndimage.distance_transform_edt(
                ~near, sampling=pitch)
    else:
        lateral_grid = np.repeat(lateral_2d[:, :, None], dims[2], axis=2)
    hazard_grid = (ndimage.distance_transform_edt(~hazard_vol, sampling=pitch)
                   if hazard_vol.any() else np.full(tuple(dims), np.inf))

    # h_clear: height to the first obstacle above each cell
    vertical_grid = np.empty(tuple(dims), dtype=float)
    height_above = np.full(dims[:2], dims[2], dtype=float)
    for k in range(dims[2] - 1, -1, -1):
        vertical_grid[:, :, k] = height_above * pitch
        height_above = np.where(obstacles[:, :, k], 0.0, height_above + 1.0)

    if prior_2d is not None:
        # a column the prior calls traversable but where no floor voxel
        # survived abstraction still has a floor: fill from the plane
        missing = prior_2d & (ground_k < 0)
        ground_k[missing] = int(np.floor((floor_z - lo[2]) / pitch))
        ground_k[~prior_2d] = -1
    # A column's ground is the highest floor voxel in it, so at 2.5 cm
    # voxels on a 10 cm lattice a single stray return lifts one column two
    # cells above its neighbours -- a 20 cm phantom step that the gate
    # then refuses to walk over. A median over the 8-neighbourhood removes
    # the spike without touching a real change in level, which is
    # continuous across many columns.
    from scipy.ndimage import median_filter

    valid = ground_k >= 0
    if valid.any():
        smoothed = median_filter(ground_k, size=3, mode="nearest")
        spike = valid & (np.abs(smoothed - ground_k) >= 2)
        ground_k = np.where(spike, smoothed, ground_k)
        ground_k[~valid] = -1

    # Slope from the measured floor height, filled from the plane where no
    # floor voxel survived abstraction, and median-filtered before
    # differencing: a gradient amplifies exactly the single-column
    # outliers a scan is full of, so smoothing after differencing is too
    # late.
    ground_z = np.where(np.isfinite(ground_z_meas) & (ground_k >= 0),
                        ground_z_meas, float(floor_z))
    ground_z = median_filter(ground_z, size=3, mode="nearest")
    gy, gx = np.gradient(ground_z, pitch)
    slope_grid = np.arctan(np.hypot(gx, gy))

    idx = arrays.index
    sample = (ijk[:, 0], ijk[:, 1], ijk[:, 2])
    fields = ClearanceFields(
        obstacle_2d=obstacle_2d,
        band_k=(int(k_lo.min()), int(k_hi.max())),
        indices=idx,
        lateral=lateral_2d[ijk[:, 0], ijk[:, 1]],
        vertical=vertical_grid[sample],
        slope=slope_grid[ijk[:, 0], ijk[:, 1]],
        hazard_distance=hazard_grid[sample],
        origin=lo,
        pitch=pitch,
        occupied=obstacles,
        lateral_grid=lateral_grid,
        vertical_grid=vertical_grid,
        hazard_grid=hazard_grid,
        slope_grid=slope_grid,
        ground_k=ground_k,
        prior_free=prior_2d,
        meta={
            "dims": dims.tolist(),
            "band_m": [band_lo, band_hi],
            "has_prior": prior_2d is not None,
            "min_voxel_evidence": int(min_voxel_evidence),
            "clearance_margin_m": round(float(0.5 * pitch * np.sqrt(2)), 4),
            "lateral_cells_free": int((~obstacle_2d).sum()),
            "floor_z": float(floor_z),
            "n_primitives": len(primitives or []),
            "obstacle_cells": int(obstacles.sum()),
            "robot": robot.name,
        },
    )
    return fields
