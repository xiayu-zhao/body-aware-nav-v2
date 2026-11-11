"""BodyNav: Body-Aware Language-guided Spatial Abstraction for robot path planning.

Perception-to-planning pipeline coupling hierarchical 3D scene graphs,
robot/task-aware adaptive voxelization, constrained LLM value-map
composition, and feasibility-gated value-guided A* planning.
"""

from bodynav.pipeline import BodyNavPipeline

__version__ = "0.1.0"
__all__ = ["BodyNavPipeline", "__version__"]
