"""Collision geometry for the simulator, from BodyNav's own voxels (Sec. IV).

A point cloud does not collide: `UsdGeomPoints` carries no collision
geometry, so a robot cannot navigate against the scan as imported. The
obvious fix is to reconstruct a surface mesh, and the better one is to
build the colliders from the **voxels BodyNav planned on**. Then a contact
in simulation is a genuine planner failure rather than an artefact of a
meshing pass the planner never saw -- which is the cleanest statement of
SCR available, and it costs a box merge instead of a day of Poisson
reconstruction.

One occupied cell per box is unusable (31,664 static prims on this floor),
so cells are merged greedily into maximal boxes first. The merge is exact:
the union of the boxes is exactly the set of occupied cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ColliderBoxes:
    """Axis-aligned boxes covering an occupancy grid, in world metres."""

    lower: np.ndarray  # (B, 3)
    upper: np.ndarray  # (B, 3)
    meta: dict

    def __len__(self) -> int:
        return len(self.lower)

    @property
    def volume(self) -> float:
        return float(np.prod(self.upper - self.lower, axis=1).sum())


def merge_boxes(occupied: np.ndarray, origin: np.ndarray, pitch: float,
                *, max_boxes: int | None = None) -> ColliderBoxes:
    """Greedy exact box decomposition of an occupancy grid.

    Runs along x, then extends in y, then in z, taking the largest box
    each time -- the standard voxel-mesher greedy pass. Exactness is the
    property that matters here: physics must see neither more nor less
    than the map does, or the executed run stops measuring the plan.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 3:
        raise ValueError(f"expected a 3-D grid, got {occupied.shape}")
    origin = np.asarray(origin, dtype=float)
    remaining = occupied.copy()
    dims = np.array(occupied.shape)

    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []

    # Walk the occupied cells once, in scan order, skipping any the
    # previous boxes already claimed. Re-scanning the grid for the next
    # unclaimed cell would be quadratic -- 2.3 million cells per box.
    order = np.argwhere(occupied)
    for cell in order:
        i, j, k = (int(v) for v in cell)
        if not remaining[i, j, k]:
            continue

        # extend along x while the run stays occupied
        i_end = i
        while i_end + 1 < dims[0] and remaining[i_end + 1, j, k]:
            i_end += 1

        # extend along y while the whole x-run is occupied
        j_end = j
        while j_end + 1 < dims[1] and remaining[i:i_end + 1, j_end + 1, k].all():
            j_end += 1

        # extend along z while the whole xy-slab is occupied
        k_end = k
        while (k_end + 1 < dims[2]
               and remaining[i:i_end + 1, j:j_end + 1, k_end + 1].all()):
            k_end += 1

        remaining[i:i_end + 1, j:j_end + 1, k:k_end + 1] = False
        lowers.append(origin + np.array([i, j, k]) * pitch)
        uppers.append(origin + (np.array([i_end, j_end, k_end]) + 1) * pitch)

        if max_boxes is not None and len(lowers) >= max_boxes:
            break

    lower = np.asarray(lowers).reshape(-1, 3)
    upper = np.asarray(uppers).reshape(-1, 3)
    return ColliderBoxes(
        lower=lower, upper=upper,
        meta={
            "cells": int(occupied.sum()),
            "boxes": len(lower),
            "pitch": float(pitch),
            "reduction": round(1.0 - len(lower) / max(int(occupied.sum()), 1), 4),
            "complete": not bool(remaining.any()),
        },
    )


def boxes_to_mesh(boxes: ColliderBoxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulated box soup: (points, face_vertex_counts, face_indices).

    Every box contributes its six faces. Interior faces between adjacent
    boxes are left in place: they are never contacted, and removing them
    costs an adjacency pass that buys nothing for a static collider.
    """
    n = len(boxes)
    if n == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=int)

    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    size = boxes.upper - boxes.lower
    points = (boxes.lower[:, None, :] + corners[None, :, :] * size[:, None, :])
    points = points.reshape(-1, 3)

    quads = np.array([
        [0, 3, 2, 1],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # -y
        [1, 2, 6, 5],  # +x
        [2, 3, 7, 6],  # +y
        [3, 0, 4, 7],  # -x
    ], dtype=int)
    offsets = (np.arange(n) * 8)[:, None, None]
    faces = (quads[None, :, :] + offsets).reshape(-1, 4)
    counts = np.full(len(faces), 4, dtype=int)
    return points, counts, faces.reshape(-1)
