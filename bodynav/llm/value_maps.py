"""Value-map composition and fusion (Sec. III-D.2, Eqs. 6-9).

Component fields over the grid:

  V_A(v) = max_{i in M_g} exp(-||p - p_i||^2 / 2 sigma_g^2)        (Eq. 6)
  V_H(v) = -sum_{i in M_h} w_h(s_i) exp(-d(p, p_i)^2 / 2 sigma_h^2) (Eq. 7)
  V_C(v) = -alpha_s [grad_z - theta_safe]_+
           -alpha_w [w_min - w(v)]_+                                (Eq. 8)

Fusion (Eq. 9): V = lambda_A V_A + lambda_H V_H + lambda_C V_C with
priority-dependent weights, then V* = Normalize[Smooth(V)] in [-1, 1].
Per-voxel scene-graph metadata is retained for explanations (which hazard
contributed most to a detour).

These reference implementations are also the *validation target* for
agent-generated code: the agent's output is checked against structural
constraints, not against these functions, but they define the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodynav.core.types import Priority, ValueMapSet, VoxelGrid, as_arrays
from bodynav.llm.grounding import GroundedTask

#: Priority-dependent fusion weights (lambda_A, lambda_H, lambda_C) —
#: lightly tuned to preserve the intended dominance order (Sec. III-D.2).
FUSION_WEIGHTS: dict[Priority, tuple[float, float, float]] = {
    Priority.SAFETY_FIRST: (0.3, 0.6, 0.1),
    Priority.EFFICIENCY_FIRST: (0.6, 0.2, 0.2),
    Priority.AREA_PREFERENCE: (0.4, 0.3, 0.3),
}


@dataclass
class ValueMapParams:
    sigma_g: float = 5.0  # goal-attraction spread (m)
    sigma_h_default: tuple[float, float] = (3.0, 5.0)  # class-table range (m)
    theta_safe: float = 0.15  # slope threshold
    alpha_s: float = 2.0  # slope penalty gain
    alpha_w: float = 1.0  # corridor-width penalty gain


def affordance_map(
    centers: np.ndarray, goal_points: np.ndarray, p: ValueMapParams
) -> np.ndarray:
    """Eq. 6: V_A(v) = max_i exp(-||p - p_i||^2 / 2 sigma_g^2)."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    goal_points = np.asarray(goal_points, dtype=float).reshape(-1, 3)
    if not len(goal_points):
        return np.zeros(len(centers))
    from scipy.spatial import cKDTree

    d, _ = cKDTree(goal_points).query(centers, k=1)
    return np.exp(-(d ** 2) / (2.0 * p.sigma_g ** 2))


def hazard_map(
    centers: np.ndarray,
    hazards: list[tuple[np.ndarray, float, float]],
    p: ValueMapParams,
) -> np.ndarray:
    """Eq. 7: V_H(v) = -sum_i w_h(s_i) exp(-d^2 / 2 sigma_h^2), <= 0.

    ``hazards`` are (position, weight, sigma) triples -- the class weight
    and spread the scene graph resolved for each hazard node.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    out = np.zeros(len(centers))
    for pos, weight, sigma in hazards:
        d2 = np.sum((centers - np.asarray(pos, dtype=float)) ** 2, axis=1)
        out -= float(weight) * np.exp(-d2 / (2.0 * float(sigma) ** 2))
    return out


def constraint_map(
    slope: np.ndarray, width: np.ndarray, w_min: float, p: ValueMapParams
) -> np.ndarray:
    """Eq. 8: slope and corridor-width penalties, both one-sided."""
    slope = np.asarray(slope, dtype=float)
    width = np.asarray(width, dtype=float)
    return -(p.alpha_s * np.maximum(slope - p.theta_safe, 0.0)
             + p.alpha_w * np.maximum(w_min - width, 0.0))


def fuse(
    affordance: np.ndarray,
    hazard: np.ndarray,
    constraint: np.ndarray,
    priority: Priority,
    *,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Eq. 9 + smoothing + normalization to [-1, 1].

    Smoothing is over the *value* vector, not over space: the maps are
    defined on a sparse, multi-resolution voxel set, so a spatial kernel
    would need a lattice the maps do not have. A light 1-D pass over the
    fused values suppresses single-voxel spikes without moving the field.
    """
    lam_a, lam_h, lam_c = FUSION_WEIGHTS[priority]
    v = lam_a * np.asarray(affordance, dtype=float) \
        + lam_h * np.asarray(hazard, dtype=float) \
        + lam_c * np.asarray(constraint, dtype=float)
    if smooth_sigma > 0 and len(v) > 3:
        from scipy.ndimage import gaussian_filter1d

        v = gaussian_filter1d(v, sigma=smooth_sigma, mode="nearest")
    scale = float(np.max(np.abs(v)))
    return v / scale if scale > 0 else v


def build_value_maps(
    grid: VoxelGrid,
    *,
    goal_points: np.ndarray,
    hazards: list[tuple[np.ndarray, float, float]],
    slope: np.ndarray,
    width: np.ndarray,
    priority: Priority = Priority.SAFETY_FIRST,
    params: ValueMapParams | None = None,
    w_min: float = 0.8,
    smooth_sigma: float = 1.0,
) -> ValueMapSet:
    """Assemble a :class:`ValueMapSet` over a grid from resolved inputs.

    This is the reference semantics of Sec. III-D.2 with the parameters
    supplied directly. Stage 3's agent does not replace this function --
    it *produces its arguments* (which goal, which hazards, what spread,
    which priority) as schema-validated code, and the fusion below is what
    executes. Keeping the split here is what lets stage 4 run and be
    measured before stage 3 exists.
    """
    p = params or ValueMapParams()
    arrays = as_arrays(grid)
    centers, indices = arrays.center, arrays.index

    v_a = affordance_map(centers, goal_points, p)
    v_h = hazard_map(centers, hazards, p)
    v_c = constraint_map(slope, width, w_min, p)
    fused = fuse(v_a, v_h, v_c, priority, smooth_sigma=smooth_sigma)

    maps = ValueMapSet(
        indices=indices,
        affordance=v_a,
        hazard=v_h,
        constraint=v_c,
        fused=fused,
        metadata={"centers": centers, "priority": np.array([priority.value])},
    )
    maps.validate()
    return maps


def compose(grounded: GroundedTask, grid: VoxelGrid, p: ValueMapParams) -> ValueMapSet:
    """Reference (non-agent) composition path, used by the Pure-Planner
    baseline and for validating agent output shapes."""
    raise NotImplementedError
