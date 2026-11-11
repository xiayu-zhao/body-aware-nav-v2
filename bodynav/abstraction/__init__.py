"""Stage 2 — robot- and task-aware abstraction (Sec. III-B) plus
semantics-guided scan completion and emphasis (Sec. III-C)."""

from bodynav.abstraction.robot_filter import (
    height_slice,
    local_ground_height,
    workspace_kernel_mask,
)
from bodynav.abstraction.voxelizer import (
    AbstractionStats,
    AdaptiveVoxelizer,
    VoxelizerParams,
)

__all__ = [
    "AbstractionStats",
    "AdaptiveVoxelizer",
    "VoxelizerParams",
    "height_slice",
    "local_ground_height",
    "workspace_kernel_mask",
]
