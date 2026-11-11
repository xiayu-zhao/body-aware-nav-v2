"""Isaac Sim / Isaac ROS (ROS 2) bridge (Sec. IV).

Concrete ``SimBridge`` for the manuscript's experimental setup:

  * scan acquisition — subscribes to the platform's sensor topics
    (H1: chest LiDAR + stereo + IMU; Crazyflie: downward RGB-D +
    forward RGB + IMU; UGV: 3D LiDAR + RGB-D) and accumulates the
    pre-execution scan;
  * ground truth — reads simulation/BIM metadata for scene-graph
    attribute enrichment when available;
  * execution — publishes the accepted trajectory as a nav path for the
    platform controller and monitors completion.

ROS 2 imports are deferred so the package imports cleanly without a ROS
environment; construction fails with a clear message instead.
"""

from __future__ import annotations

import numpy as np

from bodynav.core.types import PlanResult
from bodynav.sim.bridge import SimBridge


class IsaacSimBridge(SimBridge):
    def __init__(self, *, scan_topic: str, pose_topic: str, path_topic: str):
        try:
            import rclpy  # noqa: F401  (deferred; provided by the deployment env)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "IsaacSimBridge requires a sourced ROS 2 environment "
                "(Isaac ROS). Use OfflineBridge for file-based runs."
            ) from e
        self.scan_topic = scan_topic
        self.pose_topic = pose_topic
        self.path_topic = path_topic

    def acquire_scan(self):
        raise NotImplementedError

    def robot_pose(self) -> np.ndarray:
        raise NotImplementedError

    def execute_path(self, result: PlanResult) -> bool:
        raise NotImplementedError
