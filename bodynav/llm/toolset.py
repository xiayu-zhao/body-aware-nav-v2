"""Approved toolset for the constrained agent (Sec. III-D).

The agent may *only* call the functions registered here: scene-graph
queries, voxel/mask builders, distance transforms, and Gaussian kernels.
Generated code is executed in a namespace containing exactly this toolset
plus numpy — nothing else — so language flexibility never buys arbitrary
code execution.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from bodynav.core.types import VoxelGrid
from bodynav.scene_graph.graph import BodyNavSceneGraph


class ApprovedToolset:
    """Callable registry handed to the code-execution sandbox."""

    def __init__(self, graph: BodyNavSceneGraph, grid: VoxelGrid):
        self.graph = graph
        self.grid = grid
        self._registry: dict[str, Callable] = {
            # scene-graph queries
            "find_nodes": graph.find,
            "node_attributes": graph.attributes,
            "adjacent": graph.adjacent,
            "resolve_reference": self.resolve_reference,
            # geometry / masks
            "world2voxel": grid.world2voxel,
            "voxel2world": grid.voxel2world,
            "entity_mask": self.entity_mask,
            "distance_transform": self.distance_transform,
            "gaussian_kernel": self.gaussian_kernel,
        }

    # -- mask / field builders -----------------------------------------
    def entity_mask(self, node_ids) -> np.ndarray:
        """Binary mask over the grid support for scene-graph nodes."""
        from bodynav.llm.grounding import entity_to_mask

        if isinstance(node_ids, str):
            node_ids = [node_ids]
        return entity_to_mask(list(node_ids), self.graph, self.grid)

    def distance_transform(self, mask: np.ndarray) -> np.ndarray:
        """Per-voxel Euclidean distance (m) to the nearest True voxel.

        Computed on the voxel *centres* with a KD-tree rather than on a
        dense lattice: the grid is sparse and multi-resolution, so there is
        no single lattice a dense transform could run on.
        """
        from scipy.spatial import cKDTree

        mask = np.asarray(mask, dtype=bool)
        centers = self._centers()
        if mask.shape != (len(centers),):
            raise ValueError(
                f"mask has shape {mask.shape}, expected ({len(centers)},)")
        if not mask.any():
            return np.full(len(centers), np.inf)
        d, _ = cKDTree(centers[mask]).query(centers, k=1)
        return d

    def gaussian_kernel(self, mask: np.ndarray, sigma: float) -> np.ndarray:
        """exp(-d^2 / 2 sigma^2) falloff from a mask (Eqs. 6-7 building block)."""
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        d = self.distance_transform(mask)
        with np.errstate(over="ignore"):
            out = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return np.where(np.isfinite(d), out, 0.0)

    def _centers(self) -> np.ndarray:
        from bodynav.core.types import as_arrays

        if not hasattr(self, "_center_cache"):
            self._center_cache = as_arrays(self.grid).center
        return self._center_cache

    def support(self) -> int:
        """Number of voxels the maps must cover."""
        return len(self._centers())

    def resolve_reference(self, phrase: str) -> list[str]:
        """Candidate nodes for a natural-language reference."""
        from bodynav.llm.grounding import resolve_reference

        return resolve_reference(phrase, self.graph)[0]

    # -- sandbox --------------------------------------------------------
    #: Builtins the generated program may use. Bound explicitly by name
    #: rather than by leaving ``__builtins__`` in place, so the set is a
    #: decision rather than an inheritance.
    SAFE_BUILTINS = {
        "len": len, "abs": abs, "float": float, "int": int, "bool": bool,
        "min": min, "max": max, "sum": sum, "round": round,
        "range": range, "enumerate": enumerate, "zip": zip, "sorted": sorted,
        "list": list, "tuple": tuple,
    }

    def namespace(self) -> dict:
        """Execution namespace for generated code: toolset + numpy only."""
        ns = dict(self._registry)
        ns["np"] = np
        ns.update(self.SAFE_BUILTINS)
        return ns
