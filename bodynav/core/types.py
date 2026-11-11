"""Core data models shared by every BodyNav stage.

These types define the contracts between stages so each module can be
developed and tested independently:

    scene_graph  ->  abstraction : labeled points  P_s = {p, l, a, o, s}
    abstraction  ->  llm         : sparse ``VoxelGrid`` (+ world2voxel)
    llm          ->  planning    : ``ValueMapSet`` over the grid
    planning     ->  sim         : ``PlanResult`` (waypoints + metrics)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# Voxel indices are integer (i, j, k) triples in grid space.
Index3 = tuple[int, int, int]


class Priority(str, enum.Enum):
    """Instruction priority modes (manuscript Sec. III-D.1).

    Fusion weights (lambda_A, lambda_H, lambda_C) per mode live in
    ``configs/default.yaml`` and are resolved by ``bodynav.llm.value_maps``.
    """

    SAFETY_FIRST = "safety_first"
    EFFICIENCY_FIRST = "efficiency_first"
    AREA_PREFERENCE = "area_preference"


@dataclass
class Voxel:
    """One occupied cell of the sparse adaptive grid (Sec. III-B.3).

    Attributes:
        index: (i, j, k) grid coordinate at the *base* resolution lattice.
        center: world-frame center (meters).
        resolution: edge length r(v) in meters, in [r_min, r0].
        occupancy: 1 occupied / 0 free (free voxels are usually implicit).
        semantic_label: priority-weighted majority label (hazards weighted
            highest so mixed-category voxels stay risk-aware).
        spot_id: scene-graph spot node owning the majority of points, if any.
        synthetic: True if the voxel's surface was inferred by
            semantics-guided completion rather than observed (Sec. III-C).
        confidence: completeness score c(o) inherited from the parent spot
            when synthetic, else 1.0.
    """

    index: Index3
    center: np.ndarray
    resolution: float
    occupancy: int = 1
    semantic_label: str = "unknown"
    spot_id: str | None = None
    synthetic: bool = False
    confidence: float = 1.0


@dataclass
class VoxelGrid:
    """Sparse multi-resolution voxel grid with world<->voxel transforms.

    The grid is indexed on a uniform base lattice of pitch ``base_pitch``
    (the finest resolution); coarser voxels simply carry a larger
    ``resolution`` value. This keeps ``world2voxel`` a single affine map,
    which is the contract the LLM toolset relies on (Sec. III-D).
    """

    origin: np.ndarray  # world coords of lattice corner (3,)
    base_pitch: float  # meters; finest lattice pitch (r_min)
    voxels: dict[Index3, Voxel] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def world2voxel(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) world points to (N, 3) integer lattice indices."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.floor((pts - self.origin) / self.base_pitch).astype(int)

    def voxel2world(self, indices: np.ndarray) -> np.ndarray:
        """Map (N, 3) lattice indices to (N, 3) world-frame cell centers."""
        idx = np.atleast_2d(np.asarray(indices, dtype=float))
        return self.origin + (idx + 0.5) * self.base_pitch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.voxels)

    def __contains__(self, index: Index3) -> bool:
        return tuple(index) in self.voxels

    def __iter__(self) -> Iterator[Voxel]:
        return iter(self.voxels.values())

    def get(self, index: Index3) -> Voxel | None:
        return self.voxels.get(tuple(index))

    def add(self, voxel: Voxel) -> None:
        self.voxels[tuple(voxel.index)] = voxel

    def occupied_indices(self) -> np.ndarray:
        """(N, 3) int array of occupied voxel indices."""
        return np.array(
            [v.index for v in self.voxels.values() if v.occupancy], dtype=int
        ).reshape(-1, 3)


@dataclass
class GridArrays:
    """Columnar view of a :class:`VoxelGrid`.

    The grid's dict-of-``Voxel`` form is the right interface for building
    and inspecting a map; it is the wrong one for the stages that consume
    it, which want every centre or every label as one array. This carries
    the same content, and :func:`as_arrays` converts either way.
    """

    origin: np.ndarray
    base_pitch: float
    index: np.ndarray  # (M, 3) int
    center: np.ndarray  # (M, 3) float
    resolution: np.ndarray  # (M,) float
    occupancy: np.ndarray  # (M,)
    semantic_label: np.ndarray  # (M,) object
    spot_id: np.ndarray  # (M,) object
    synthetic: np.ndarray  # (M,) bool
    confidence: np.ndarray  # (M,) float

    def __len__(self) -> int:
        return len(self.index)

    # the same affine map VoxelGrid exposes, so either form can be handed
    # to the toolset and to the stages that only need the transform
    def world2voxel(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return np.floor((pts - self.origin) / self.base_pitch).astype(int)

    def voxel2world(self, indices: np.ndarray) -> np.ndarray:
        idx = np.atleast_2d(np.asarray(indices, dtype=float))
        return self.origin + (idx + 0.5) * self.base_pitch


def as_arrays(grid) -> GridArrays:
    """Accept a ``VoxelGrid`` or a ``GridArrays`` and return columns."""
    if isinstance(grid, GridArrays):
        return grid
    voxels = list(grid)
    if not voxels:
        empty = np.empty((0, 3))
        return GridArrays(
            origin=np.asarray(grid.origin, dtype=float), base_pitch=grid.base_pitch,
            index=empty.astype(int), center=empty, resolution=np.empty(0),
            occupancy=np.empty(0, dtype=int),
            semantic_label=np.empty(0, dtype=object),
            spot_id=np.empty(0, dtype=object),
            synthetic=np.empty(0, dtype=bool), confidence=np.empty(0),
        )
    return GridArrays(
        origin=np.asarray(grid.origin, dtype=float),
        base_pitch=float(grid.base_pitch),
        index=np.array([v.index for v in voxels], dtype=int).reshape(-1, 3),
        center=np.array([v.center for v in voxels], dtype=float).reshape(-1, 3),
        resolution=np.array([v.resolution for v in voxels], dtype=float),
        occupancy=np.array([v.occupancy for v in voxels]),
        semantic_label=np.array([v.semantic_label for v in voxels], dtype=object),
        spot_id=np.array([v.spot_id for v in voxels], dtype=object),
        synthetic=np.array([v.synthetic for v in voxels], dtype=bool),
        confidence=np.array([v.confidence for v in voxels], dtype=float),
    )


@dataclass
class TaskSpec:
    """Machine-readable parse T = {S, G, H, C, pi} of an instruction
    (Sec. III-D.1), before grounding.

    Entity fields hold scene-graph references (names/queries), resolved to
    voxel masks by ``bodynav.llm.grounding``.
    """

    start: str | None
    goal: str
    hazards: list[str] = field(default_factory=list)
    constraints: dict[str, float | str] = field(default_factory=dict)
    priority: Priority = Priority.SAFETY_FIRST
    #: explicit instruction-provided margins, e.g. {"welding_zone": 5.0}
    safety_distances: dict[str, float] = field(default_factory=dict)
    raw_instruction: str = ""
    #: open-vocabulary grounding only: the scene category the model says
    #: the goal / each hazard phrase refers to (None / {} under lexical)
    goal_category: str | None = None
    hazard_categories: dict[str, str] = field(default_factory=dict)


@dataclass
class ValueMapSet:
    """Component and fused value maps over the voxel grid (Sec. III-D.2).

    Arrays are keyed positionally against ``indices`` (M, 3); values are
    float. The fused field is normalized to [-1, 1].
    """

    indices: np.ndarray  # (M, 3) voxel indices the maps are defined on
    affordance: np.ndarray  # V_A, goal attraction         (M,)
    hazard: np.ndarray  # V_H, hazard repulsion (<= 0)     (M,)
    constraint: np.ndarray  # V_C, feasibility penalties   (M,)
    fused: np.ndarray  # V*  in [-1, 1]                    (M,)
    #: per-voxel provenance for explanations ("which hazard caused this")
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        """Structural checks applied before the planner may consume the
        maps (Sec. III-D): shape agreement, finiteness, non-empty support."""
        n = len(self.indices)
        for name in ("affordance", "hazard", "constraint", "fused"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if n == 0:
            raise ValueError("value maps have empty support")


@dataclass
class PlanResult:
    """Output of the full pipeline for one instruction."""

    success: bool
    #: accepted post-processed trajectory, world frame (K, 3); empty if the
    #: instruction was correctly rejected as infeasible
    path: np.ndarray
    task: TaskSpec | None = None
    value_maps: ValueMapSet | None = None
    #: metrics computed on the accepted trajectory (Sec. IV): path length,
    #: normalized length, planning time, min/mean hazard clearance, ...
    metrics: dict[str, float] = field(default_factory=dict)
    #: agent trace (prompt -> parse -> tool calls -> code -> maps) for
    #: inspection and release artifacts
    trace: list[dict] = field(default_factory=list)
    failure_reason: str | None = None
