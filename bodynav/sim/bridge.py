"""Abstract simulator bridge + offline fallback.

The pipeline itself is simulator-agnostic; a bridge supplies scans and
consumes paths. ``IsaacSimBridge`` (bodynav.sim.isaac_bridge) is the
concrete backend matching the manuscript's setup (Isaac Sim via Isaac
ROS on ROS 2, sequential per trial: pre-execution scan -> BodyNav
representation -> planning on a static scene).
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np

from bodynav.core.types import PlanResult


class SimBridge(abc.ABC):
    """Minimal contract between BodyNav and any simulation/deployment target."""

    @abc.abstractmethod
    def acquire_scan(self) -> tuple[np.ndarray, list]:
        """Return (point cloud (N, 3+), posed RGB-D frames) for the
        pre-execution scan."""

    @abc.abstractmethod
    def robot_pose(self) -> np.ndarray:
        """Current robot pose, world frame (4, 4)."""

    @abc.abstractmethod
    def execute_path(self, result: PlanResult) -> bool:
        """Hand the accepted trajectory to the platform controller.
        Returns execution success. BodyNav evaluates navigation-level
        feasibility; gait/flight control is the platform's concern."""


class OfflineBridge(SimBridge):
    """File-based fallback: loads a saved scan, writes paths to disk.
    Lets the full pipeline run without ROS 2 / Isaac Sim installed
    (e.g., on the real-world lab-floor dataset)."""

    def __init__(self, cloud_path: str, frames_dir: str | None = None,
                 output_dir: str = "outputs", *,
                 stride: int = 1, max_points: int | None = None,
                 up_axis: str = "z"):
        self.cloud_path = cloud_path
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        #: subsampling applied at load time — the as-built lab-floor scan
        #: is multi-gigabyte, so experiments are swept over point budgets
        self.stride = stride
        self.max_points = max_points
        #: vertical axis of the source file; the lab floor scan is Y-up
        self.up_axis = up_axis
        self.colors: np.ndarray | None = None

    def acquire_scan(self):
        """Load the saved cloud. Posed RGB-D frames are returned when a
        ``frames_dir`` was given; the lab-floor dataset has none, so the
        list is empty and downstream stages fall back to geometry."""
        from bodynav.io.pointcloud import load_cloud

        xyz, colors = load_cloud(
            self.cloud_path, stride=self.stride, max_points=self.max_points,
            up_axis=self.up_axis,
        )
        self.colors = colors
        frames: list = []
        if self.frames_dir:
            frames = sorted(Path(self.frames_dir).glob("*.npz"))
        return xyz, frames

    def robot_pose(self) -> np.ndarray:
        return np.eye(4)

    def execute_path(self, result: PlanResult) -> bool:
        """Serialize waypoints + metrics instead of driving a platform."""
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "path.csv", np.asarray(result.path).reshape(-1, 3),
                   delimiter=",", header="x,y,z", comments="")
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {"success": result.success,
                 "failure_reason": result.failure_reason,
                 "metrics": result.metrics},
                f, indent=2,
            )
        return bool(result.success)
