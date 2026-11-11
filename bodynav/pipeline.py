"""End-to-end BodyNav pipeline orchestrator (manuscript Sec. III, Fig. 1).

Four stages run identically for UAV-scanned, UGV-scanned, or pre-existing
point clouds:

    1. hierarchical scene-graph generation        (bodynav.scene_graph)
    2. robot/task-aware abstraction + completion  (bodynav.abstraction)
    3. constrained LLM value-map composition      (bodynav.llm)
    4. feasibility-gated value-guided A*          (bodynav.planning)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bodynav.core.types import PlanResult
from bodynav.robots.profiles import RobotProfile


class BodyNavPipeline:
    """Wires the four BodyNav stages together behind one call.

    Heavy sub-systems (SAM, the LLM backend) are injected lazily so the
    pipeline can run partially (e.g. planning on a cached scene graph and
    grid) without pulling in every dependency.
    """

    def __init__(self, config: dict, robot: RobotProfile):
        self.config = config
        self.robot = robot
        self._scene_graph = None  # bodynav.scene_graph.graph.BodyNavSceneGraph
        self._grid = None  # bodynav.core.types.VoxelGrid

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path, robot: RobotProfile) -> "BodyNavPipeline":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config=config, robot=robot)

    # ------------------------------------------------------------------
    def build_scene_graph(self, cloud, frames=None):
        """Stage 1: hierarchical scene graph G = (N, E, Phi, Psi).

        Args:
            cloud: (N, 3+) point cloud or path to one (.ply).
            frames: posed RGB-D frames for SAM segmentation; optional when
                loading a cached/simulation-annotated graph.
        """
        raise NotImplementedError("Stage 1 wiring lands in the next step.")

    def build_abstraction(self, instruction: str | None = None):
        """Stage 2: robot slice -> task filter -> adaptive voxelization,
        then semantics-guided completion + emphasis (Sec. III-B/III-C)."""
        raise NotImplementedError("Stage 2 wiring lands in the next step.")

    def compose_value_maps(self, instruction: str):
        """Stage 3: constrained LLM agent -> validated ValueMapSet."""
        raise NotImplementedError("Stage 3 wiring lands in the next step.")

    def plan(self, value_maps) -> PlanResult:
        """Stage 4: feasibility-gated value-guided A* + post-processing."""
        raise NotImplementedError("Stage 4 wiring lands in the next step.")

    # ------------------------------------------------------------------
    def run(self, cloud, instruction: str, frames=None) -> PlanResult:
        """Full pipeline for one instruction. Target public API (see README)."""
        self.build_scene_graph(cloud, frames)
        self.build_abstraction(instruction)
        maps = self.compose_value_maps(instruction)
        return self.plan(maps)
