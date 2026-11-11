"""Adaptive sparse voxelization (Sec. III-B.3, Eq. 4 extended by Sec. III-C).

Edge length per voxel:

    r(v) = r0 * (1 - alpha * phi_R(v) - beta * phi_T(v) - gamma * phi_E(v))

  phi_R : robot cues — fine within the locomotion band and within the
          foothold buffer of expected footholds;
  phi_T : LLM-interpreted task priority — fine at hazards, goals,
          corridor centerlines;
  phi_E : emphasis cue from semantics-guided completion (Sec. III-C) —
          fine at completed and safety-relevant spots;
  bounds keep r(v) in [r_min, r0] (2.5 cm floor captures step edges and
  small hazards while staying above depth noise; bounds prevent global
  collapse when all cues are high).

Each voxel stores occupancy, distance metrics, and a semantic label by
priority-weighted voting (hazard classes weighted highest so mixed
voxels stay risk-aware).

Implementation note: continuous r(v) is snapped to a dyadic ladder
r_min * 2^k so that every resolution nests on the base lattice and
``VoxelGrid.world2voxel`` stays a single affine map (see
``docs/architecture.md``). The top of the ladder is the largest such
level not exceeding r0, i.e. 0.4 m for the manuscript's r0 = 0.5 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bodynav.core.types import Voxel, VoxelGrid
from bodynav.robots.profiles import RobotProfile
from bodynav.scene_graph.attributes import DEFAULT_HAZARD_WEIGHTS
from bodynav.scene_graph.hierarchy import LabeledPoints

#: Vote weight for hazard-class labels — highest, so a voxel mixing a
#: hazard with background geometry still reports the hazard.
HAZARD_VOTE_WEIGHT = 4.0
#: Vote weight for safety-relevant but non-hazard categories.
SAFETY_VOTE_WEIGHT = 2.0
#: Categories that are safety-relevant on a construction/lab site.
SAFETY_LABELS: frozenset[str] = frozenset(
    {"safety_sign", "cone", "barrel", "stairs", "ladder", "door", "pipe"}
)


def default_label_weights() -> dict[str, float]:
    """Priority weights w_s used by the semantic vote."""
    weights = {label: SAFETY_VOTE_WEIGHT for label in SAFETY_LABELS}
    weights.update({label: HAZARD_VOTE_WEIGHT for label in DEFAULT_HAZARD_WEIGHTS})
    return weights


@dataclass
class VoxelizerParams:
    r0: float = 0.5  # background resolution (m)
    r_min: float = 0.025  # finest resolution (m)
    alpha: float = 0.6  # robot-cue gain, in [0.5, 0.7]
    beta: float = 0.6  # task-cue gain, in [0.5, 0.7]
    gamma: float = 0.3  # emphasis-cue gain (Sec. III-C)
    foothold_buffer: float = 0.3  # m, conservative placement uncertainty
    #: phi_R plateau inside the locomotion band above the ground layer
    band_weight: float = 0.5
    #: phi_R decays from ``band_weight`` to 0 over this height above the band
    band_taper: float = 0.5
    #: phi_T decays from 1 to 0 over this distance outside the task mask
    task_falloff: float = 1.0
    #: Cap on resolution inside the robot's own band, as a fraction of the
    #: lateral clearance the planner's gate will test.
    #:
    #: Being robot-aware has to mean more than putting fine cells near the
    #: ground: it has to mean cells fine enough to answer the question the
    #: gate asks. A voxel quantises the obstacle boundary by its own size,
    #: so a 0.2 m cell -- what alpha = 0.6 alone yields in the band -- puts
    #: 0.2 m of error into a 0.425 m half-width test, and inflates walls
    #: until corridors close. On the lab floor that cost BodyNav two thirds
    #: of its reachable area against a uniform map of the same memory.
    #:
    #: It costs almost nothing to fix, because the background has already
    #: been decimated to ``background_spacing`` (5 cm) by the
    #: priority merge: coarsening it further re-quantises points that are
    #: already sparse, buying no memory and losing accuracy. Set to None
    #: to recover the uncapped Eq. 4 behaviour.
    band_clearance_fraction: float | None = 0.1


@dataclass
class AbstractionStats:
    """Reduction bookkeeping for the evaluation tables (Sec. V, VI)."""

    n_points: int = 0
    n_voxels: int = 0
    voxels_by_resolution: dict[float, int] = field(default_factory=dict)
    fine_share: float = 0.0  # fraction of voxels at <= 0.25 m
    synthetic_share: float = 0.0
    mean_resolution: float = 0.0

    @property
    def reduction(self) -> float:
        """1 - voxels/points; the '60-80% reduction' claim in Sec. III-B.1."""
        return 1.0 - self.n_voxels / self.n_points if self.n_points else 0.0


class AdaptiveVoxelizer:
    def __init__(self, params: VoxelizerParams, robot: RobotProfile):
        self.params = params
        self.robot = robot
        #: populated by :meth:`voxelize`; read by the evaluation scripts
        self.last_stats: AbstractionStats | None = None

    # -- resolution ladder ----------------------------------------------
    def levels(self) -> np.ndarray:
        """Dyadic resolution ladder r_min * 2^k, capped at r0."""
        p = self.params
        if p.r_min <= 0 or p.r0 < p.r_min:
            raise ValueError(f"need 0 < r_min <= r0, got {p.r_min}, {p.r0}")
        n = int(np.floor(np.log2(p.r0 / p.r_min))) + 1
        return p.r_min * 2.0 ** np.arange(n)

    # -- refinement cues ------------------------------------------------
    def _z_rel(self, points: LabeledPoints) -> np.ndarray:
        """Height above local ground, reusing the slice's cached value."""
        cached = points.extras.get("z_rel")
        if cached is not None and len(cached) == len(points.xyz):
            return np.asarray(cached, dtype=np.float64)
        from bodynav.abstraction.robot_filter import local_ground_height

        xyz = np.asarray(points.xyz, dtype=np.float64)
        return xyz[:, 2] - local_ground_height(xyz)

    def phi_robot(self, points: LabeledPoints) -> np.ndarray:
        """phi_R in [0, 1]: locomotion band + foothold-band proximity.

        Peaks at 1 in the near-ground layer where contact geometry decides
        traversability (the foothold band widened by the placement buffer,
        or the wheel-contact layer for platforms without footholds), holds
        at ``band_weight`` through the rest of the body sweep, and tapers
        to 0 above it so overhead structure stays coarse.
        """
        p = self.params
        z_rel = self._z_rel(points)
        band_lo, band_hi = self.robot.height_band
        ground_layer = (self.robot.foothold_band or 0.0) + p.foothold_buffer

        phi = np.zeros(len(z_rel), dtype=np.float64)
        in_band = (z_rel >= band_lo) & (z_rel <= band_hi)
        phi[in_band] = p.band_weight

        near_ground = in_band & (z_rel <= band_lo + ground_layer)
        phi[near_ground] = 1.0

        if p.band_taper > 0:
            above = z_rel > band_hi
            decay = 1.0 - (z_rel[above] - band_hi) / p.band_taper
            phi[above] = p.band_weight * np.clip(decay, 0.0, 1.0)

        return np.clip(phi, 0.0, 1.0)

    def phi_task(
        self,
        points: LabeledPoints,
        task_mask: np.ndarray,
        *,
        hazard_labels: frozenset[str] | None = None,
    ) -> np.ndarray:
        """phi_T in [0, 1]: hazards, goals, corridor centerlines.

        1 inside the grounded task mask V_t and on hazard-class points
        regardless of the mask, decaying linearly to 0 over
        ``params.task_falloff`` metres outside it, so resolution does not
        step discontinuously at the corridor boundary.
        """
        xyz = np.asarray(points.xyz, dtype=np.float64)
        phi = np.zeros(len(xyz), dtype=np.float64)
        if len(xyz) == 0:
            return phi

        if task_mask is not None:
            task_mask = np.asarray(task_mask)
            if task_mask.shape != (len(xyz),):
                raise ValueError(
                    f"task_mask has shape {task_mask.shape}, expected ({len(xyz)},)"
                )
            phi[task_mask.astype(bool)] = 1.0

        labels = hazard_labels or frozenset(DEFAULT_HAZARD_WEIGHTS)
        if labels and points.semantic is not None:
            phi[np.isin(points.semantic.astype(str), list(labels))] = 1.0

        falloff = self.params.task_falloff
        core = phi >= 1.0
        if falloff > 0 and core.any() and not core.all():
            from scipy.spatial import cKDTree

            from bodynav.io.pointcloud import voxel_downsample

            # The tree only needs to resolve distance to within a fraction
            # of the falloff, so thin the core before building it.
            seeds, _ = voxel_downsample(xyz[core], max(falloff / 4.0, 1e-3))
            tree = cKDTree(seeds)
            outside = ~core
            dist, _ = tree.query(
                xyz[outside], k=1, distance_upper_bound=falloff
            )
            with np.errstate(invalid="ignore"):
                phi[outside] = np.clip(1.0 - dist / falloff, 0.0, 1.0)

        return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    # -- main -----------------------------------------------------------
    def voxelize(
        self,
        points: LabeledPoints,
        task_mask: np.ndarray,
        emphasis: np.ndarray | None = None,
    ) -> VoxelGrid:
        """Build the sparse multi-resolution grid.

        Args:
            emphasis: optional per-point phi_E from
                ``bodynav.abstraction.completion`` (zeros when absent).
        """
        p = self.params
        xyz = np.asarray(points.xyz, dtype=np.float64)
        levels = self.levels()
        base_pitch = float(levels[0])

        if len(xyz) == 0:
            self.last_stats = AbstractionStats()
            return VoxelGrid(origin=np.zeros(3), base_pitch=base_pitch)

        phi_r = self.phi_robot(points)
        phi_t = self.phi_task(points, task_mask)
        if emphasis is None:
            phi_e = np.zeros(len(xyz), dtype=np.float64)
        else:
            phi_e = np.clip(np.asarray(emphasis, dtype=np.float64), 0.0, 1.0)
            if phi_e.shape != (len(xyz),):
                raise ValueError(
                    f"emphasis has shape {phi_e.shape}, expected ({len(xyz)},)"
                )

        # Eq. 4, clipped to the admissible band.
        r = p.r0 * (1.0 - p.alpha * phi_r - p.beta * phi_t - p.gamma * phi_e)
        r = np.clip(r, p.r_min, p.r0)
        if p.band_clearance_fraction is not None:
            # No coarser than the gate can afford, anywhere a voxel's own
            # extent could reach into the band the robot sweeps.
            #
            # The test is on the voxel's *extent*, not its centre. A duct
            # 0.15 m above the band, tiled at 0.4 m, occupies 0.2 m below
            # its centre and so lands inside the band -- a phantom
            # obstacle at head height for a vehicle 1.25 m tall. Keying on
            # the centre (or on phi_R, which is 0 both above the band and
            # below local ground) leaves exactly those in place.
            cap = max(p.band_clearance_fraction
                      * float(self.robot.min_lateral_clearance), p.r_min)
            z_rel = self._z_rel(points)
            band_hi = float(self.robot.height_band[1])
            reaches_band = (z_rel - 0.5 * r) <= band_hi
            r = np.where(reaches_band, np.minimum(r, cap), r)

        # Snap onto the dyadic ladder (largest level not exceeding r).
        level = np.clip(np.searchsorted(levels, r, side="right") - 1, 0, len(levels) - 1)

        # Origin on the coarsest lattice so every level nests exactly.
        top = float(levels[-1])
        origin = np.floor(xyz.min(axis=0) / top) * top
        grid = VoxelGrid(origin=origin, base_pitch=base_pitch)

        codes, code_names = self._encode(points.semantic, len(xyz))
        vote_w = self._vote_weights(code_names)
        synthetic = self._per_point_flag(points, "synthetic", len(xyz), False)
        confidence = self._per_point_flag(points, "confidence", len(xyz), 1.0)

        stats = AbstractionStats(n_points=len(xyz))

        # Coarse levels first so that, on the rare boundary where a coarse
        # cell also contains finer points, the finer voxel is the one kept.
        span = xyz.max(axis=0) - origin
        for li in range(len(levels) - 1, -1, -1):
            sel = np.flatnonzero(level == li)
            if sel.size == 0:
                continue
            pitch = float(levels[li])
            # Scene extent in cells at this pitch; shared by every subset
            # packed below so their keys are directly comparable.
            dims = np.floor(span / pitch).astype(np.int64) + 2
            cells = np.floor((xyz[sel] - origin) / pitch).astype(np.int64)
            keys = self._pack(cells, dims)

            if li > 0:
                finer = np.flatnonzero(level < li)
                if finer.size:
                    finer_cells = np.floor((xyz[finer] - origin) / pitch).astype(np.int64)
                    occupied_finer = np.unique(self._pack(finer_cells, dims))
                    keep = ~np.isin(keys, occupied_finer)
                    if not keep.all():
                        # This cell also holds points that want a finer
                        # resolution. Promote *its* points down a level
                        # rather than dropping the cell: dropping would
                        # leave them uncovered, since the finer voxels only
                        # span where the finer points are. Refining is also
                        # the safe direction for a planner.
                        level[sel[~keep]] = li - 1
                        sel, keys, cells = sel[keep], keys[keep], cells[keep]
                        if sel.size == 0:
                            continue

            uniq_keys, first, inv = np.unique(
                keys, return_index=True, return_inverse=True
            )
            inv = inv.reshape(-1)
            n_cells = len(uniq_keys)

            best = _weighted_majority(
                inv, n_cells, codes[sel], vote_w[codes[sel]], len(code_names)
            )
            spot = _majority(inv, n_cells, points.instance_id[sel])
            syn = np.zeros(n_cells, dtype=bool)
            np.logical_or.at(syn, inv, synthetic[sel])
            conf = np.ones(n_cells, dtype=np.float64)
            np.minimum.at(conf, inv, confidence[sel])

            cell_coords = cells[first]
            scale = int(round(pitch / base_pitch))
            centers = origin + (cell_coords + 0.5) * pitch
            base_idx = cell_coords * scale

            for c in range(n_cells):
                grid.add(
                    Voxel(
                        index=(int(base_idx[c, 0]), int(base_idx[c, 1]),
                               int(base_idx[c, 2])),
                        center=centers[c],
                        resolution=pitch,
                        occupancy=1,
                        semantic_label=code_names[best[c]],
                        spot_id=None if spot[c] < 0 else f"spot_{int(spot[c])}",
                        synthetic=bool(syn[c]),
                        confidence=float(conf[c]),
                    )
                )

        self.last_stats = self._finalize_stats(stats, grid)
        return grid

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _pack(cells: np.ndarray, dims: np.ndarray) -> np.ndarray:
        """Collision-free int64 key for (i, j, k) cell coordinates.

        ``dims`` must come from the *scene* extent at this pitch, not from
        the array being packed, so keys stay comparable between two
        different point subsets at the same level.
        """
        return (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]

    @staticmethod
    def _encode(semantic, n: int) -> tuple[np.ndarray, list[str]]:
        if semantic is None:
            return np.zeros(n, dtype=np.int64), ["unknown"]
        names, codes = np.unique(np.asarray(semantic).astype(str), return_inverse=True)
        return codes.astype(np.int64), list(names)

    @staticmethod
    def _vote_weights(code_names: list[str]) -> np.ndarray:
        table = default_label_weights()
        return np.array([table.get(name, 1.0) for name in code_names], dtype=np.float64)

    @staticmethod
    def _per_point_flag(points: LabeledPoints, key: str, n: int, default):
        value = points.extras.get(key)
        if value is None or len(np.atleast_1d(value)) != n:
            dtype = bool if isinstance(default, bool) else np.float64
            return np.full(n, default, dtype=dtype)
        return np.asarray(value)

    def _finalize_stats(
        self, stats: AbstractionStats, grid: VoxelGrid
    ) -> AbstractionStats:
        stats.n_voxels = len(grid)
        if stats.n_voxels:
            res = np.array([v.resolution for v in grid], dtype=np.float64)
            stats.mean_resolution = float(res.mean())
            stats.fine_share = float(np.mean(res <= 0.25))
            stats.synthetic_share = float(np.mean([v.synthetic for v in grid]))
            # Counted off the grid rather than accumulated per level, so the
            # numbers survive any cross-level index collision.
            uniq, counts = np.unique(res, return_counts=True)
            stats.voxels_by_resolution = {
                float(r): int(c) for r, c in zip(uniq, counts)
            }
        return stats

    @staticmethod
    def vote_label(labels: np.ndarray, weights: dict[str, float]) -> str:
        """Priority-weighted majority label:
        s(v) = argmax_s sum_i w_s(s_i) * 1[s_i = s]."""
        labels = np.asarray(labels).astype(str)
        if labels.size == 0:
            return "unknown"
        names, inv = np.unique(labels, return_inverse=True)
        w = np.array([weights.get(n, 1.0) for n in names], dtype=np.float64)
        totals = np.bincount(inv, minlength=len(names)) * w
        return str(names[int(np.argmax(totals))])


# ----------------------------------------------------------------------
def _weighted_majority(
    cells: np.ndarray, n_cells: int, codes: np.ndarray,
    weights: np.ndarray, n_codes: int,
) -> np.ndarray:
    """Per-cell argmax of summed weights, in O(N log N) time and O(N) space."""
    out = np.zeros(n_cells, dtype=np.int64)
    if len(cells) == 0:
        return out

    comb = cells.astype(np.int64) * n_codes + codes
    order = np.argsort(comb, kind="stable")
    c, w = comb[order], weights[order]

    starts = np.flatnonzero(np.concatenate(([True], c[1:] != c[:-1])))
    sums = np.add.reduceat(w, starts)
    g_cell, g_code = np.divmod(c[starts], n_codes)

    # Ascending by (cell, weight) means the last row of each cell wins.
    o2 = np.lexsort((sums, g_cell))
    gc, gk = g_cell[o2], g_code[o2]
    last = np.flatnonzero(np.concatenate((gc[1:] != gc[:-1], [True])))
    out[gc[last]] = gk[last]
    return out


def _majority(cells: np.ndarray, n_cells: int, values: np.ndarray) -> np.ndarray:
    """Per-cell plain majority of an integer array (used for spot ids)."""
    values = np.asarray(values, dtype=np.int64)
    if len(cells) == 0:
        return np.full(n_cells, -1, dtype=np.int64)
    names, codes = np.unique(values, return_inverse=True)
    best = _weighted_majority(
        cells, n_cells, codes, np.ones(len(codes), dtype=np.float64), len(names)
    )
    return names[best]
