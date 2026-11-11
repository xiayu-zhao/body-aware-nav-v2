"""Node attribute schema Phi(v) = (geom, s, a) with
a = (a_aff, a_mat, a_haz) (Sec. III-A.3).

Hazard nodes carry *continuous* safety parameters — a clearance radius
(bounding volume inflated by a policy-derived margin) and a class risk
weight — consumed directly by task filtering (Sec. III-B.2), value maps
(Sec. III-D.2), and planning gates (Sec. III-E.1).

Attribute values derive from simulation/BIM metadata when available, else
from open-vocabulary comparison of aggregated semantic evidence against
curated site prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Hazard class risk weights w_h (Sec. III-D.2); extend per site policy.
DEFAULT_HAZARD_WEIGHTS: dict[str, float] = {
    "welding_zone": 1.0,
    "crane_zone": 0.6,
    "overhead_work": 0.4,
}

#: Categories eligible for semantics-guided completion (Sec. III-C),
#: grouped by the completion rule that applies.
COMPLETION_CATEGORIES: dict[str, tuple[str, ...]] = {
    "planar_extension": ("wall", "floor", "ceiling", "door", "safety_sign", "window"),
    "axis_symmetric_extrusion": ("pipe", "cone", "post", "column", "duct"),
    "prior_shape_fitting": ("desk", "chair", "ladder", "stairs", "barrel", "shelf"),
}


@dataclass
class Affordances:
    navigable: bool = False
    loadable: bool = False
    hazardous: bool = False


@dataclass
class HazardParams:
    """Continuous safety parameters for hazard-class nodes."""

    clearance_radius: float  # meters, bounding volume + policy margin
    risk_weight: float  # w_h in [0, 1]
    hazard_class: str = "generic"


@dataclass
class NodeAttributes:
    """Phi(v) for one scene-graph node."""

    centroid: np.ndarray  # (3,)
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)
    semantic: str = "unknown"
    affordances: Affordances = field(default_factory=Affordances)
    material: str = "unknown"
    hazard: HazardParams | None = None
    #: completeness score c(o) in [0, 1] (Sec. III-C); 1.0 = fully observed
    completeness: float = 1.0
    #: True once semantics-guided completion has synthesized geometry
    completed: bool = False


def enrich_from_open_vocabulary(semantic_evidence, site_prompts) -> NodeAttributes:
    """Open-vocabulary attribute assignment fallback (no BIM metadata)."""
    raise NotImplementedError
