"""Evaluation metrics (Sec. IV "Metrics").

(i)   Semantic grounding — per-level mIoU; grounding accuracy for
      single-entity / hierarchical / relational references.
(ii)  Abstraction — point reduction; mean geometric deviation (one-sided
      distance from occupied-voxel centers to a Poisson reconstruction of
      the raw cloud, isolating abstraction error from planning error).
(iii) Value-map fidelity — voxel-wise Pearson correlation against
      expert-defined maps, per component and priority.
(iv)  Planning — PSR, normalized length L(P)/L(P*), planning time, SCR
      (zero violations), min/mean hazard clearance, violations by type.
(v)   Completion (real-world, Sec. VI) — completeness before/after,
      geometric deviation of completed surfaces vs. a reference re-scan,
      fine-voxel share at emphasized spots.
"""

from __future__ import annotations

import numpy as np

from bodynav.core.types import PlanResult, ValueMapSet


def grounding_miou(pred_labels, gt_labels, level: str) -> float:
    raise NotImplementedError


def geometric_deviation(grid, raw_cloud: np.ndarray) -> float:
    """Mean one-sided distance, occupied voxel centers -> Poisson mesh."""
    raise NotImplementedError


def value_map_correlation(maps: ValueMapSet, expert: ValueMapSet) -> dict[str, float]:
    """Pearson rho per component + fused."""
    raise NotImplementedError


def planning_metrics(result: PlanResult, optimal_length: float, hazards) -> dict[str, float]:
    """PSR contribution, normalized length, min/mean clearance, violations."""
    raise NotImplementedError


def completion_metrics(report, reference_scan=None) -> dict[str, float]:
    """Real-world completion metrics from a CompletionReport."""
    raise NotImplementedError
