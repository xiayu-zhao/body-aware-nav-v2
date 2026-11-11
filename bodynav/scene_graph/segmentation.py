"""Multi-view segmentation via 3D-centric SAM prompting (Sec. III-A.1).

Prompts SAM *in 3D*: K furthest-point-sampled prompts are projected into
every posed frame (Eq. 1), depth-tested against D_t to reject occluded
projections, aggregated by retention probability pi_k = m_k / n_k >= delta,
back-projected, and consolidated across views with the symmetric overlap
score Ovl(S_i, S_j) = |S_i ∩ S_j| / min(|S_i|, |S_j|). Segments with
Ovl >= tau_seg merge into one spot-level entity.

Expert refinement hooks trigger only for hazard-class or low-confidence
(pi_k < 0.4) segments; a correction propagates to all views through the
3D-centric aggregation.

Attribution: the cross-view consolidation and open-vocabulary pipeline
follow HOV-SG (https://github.com/hovsg/HOV-SG, MIT); parts of stage 1
may be adapted from it — see docs/third_party.md for the policy and the
deliberate differences (3D-centric prompting, site-semantic hierarchy,
continuous hazard attributes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PosedFrame:
    """One posed RGB-D observation (I_t, D_t, T_t) with intrinsics K_c."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float meters
    pose: np.ndarray  # (4, 4) world-from-camera
    intrinsics: np.ndarray  # (3, 3)


def sample_prompts(cloud: np.ndarray, k: int) -> np.ndarray:
    """Furthest-point-sample K 3D prompt locations from the cloud."""
    raise NotImplementedError


def project_prompts(
    prompts: np.ndarray, frame: PosedFrame, depth_tolerance: float = 0.05
) -> np.ndarray:
    """Project prompts into a frame (Eq. 1) with a depth-based occlusion
    test; returns pixel coords with NaN rows for occluded/out-of-view."""
    raise NotImplementedError


def segment_cloud(
    cloud: np.ndarray,
    frames: list[PosedFrame],
    *,
    k_prompts: int,
    retention_delta: float,
    overlap_tau: float,
) -> dict[int, np.ndarray]:
    """Full 3D-centric segmentation: returns {spot_id: point index array}.

    Requires the ``segmentation`` extra (torch + segment-anything).
    """
    raise NotImplementedError
