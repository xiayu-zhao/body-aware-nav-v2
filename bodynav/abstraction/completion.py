"""Semantics-guided scan completion and emphasis (Sec. III-C).

Real scans are incomplete where navigation needs geometry most: occlusion,
oblique viewpoints, and reflective/low-texture surfaces leave half-captured
doors and furniture, illegible safety signage, and truncated pipes,
ladders, and stair edges.

Per spot o with category prior M_{s(o)} (canonical extent + symmetry
class), the completeness score (Eq. 5)

    c(o) = |surface(o_observed)| / |surface(o_expected | prior)|  in [0, 1]

Nodes with c(o) < tau_c in navigation-critical categories are completed by
category-conditioned rules (see COMPLETION_CATEGORIES in
bodynav.scene_graph.attributes):

  * planar extension           — walls, floors, ceilings, doors, signage
  * axis-symmetric extrusion   — pipes, cones, posts
  * prior-shape fitting        — furniture, ladders, stairs

Completed surfaces are flagged synthetic and carry c(o) as a confidence
attribute so the planner can distinguish observed from inferred geometry.
Completed and safety-relevant spots then emit the emphasis cue phi_E
consumed by the adaptive voxelizer, concentrating fine voxel density on
objects that matter for navigation: precision improves where instructions
and safety demand it, and the scene graph's contextual integrity is
restored without re-scanning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints


@dataclass
class CompletionParams:
    tau_c: float = 0.7  # completeness threshold
    emphasis_strength: float = 1.0  # peak phi_E at completed/safety spots
    #: max fraction of a spot's surface that may be synthesized; guards
    #: against hallucinating geometry from tiny fragments
    max_synthetic_ratio: float = 0.6


@dataclass
class CompletionReport:
    """Per-scene audit for evaluation (real-world Table, Sec. VI)."""

    audited: int = 0
    incomplete: int = 0
    completed_by_category: dict[str, int] | None = None
    mean_completeness_before: float = 0.0
    mean_completeness_after: float = 0.0


#: Category priors: canonical extent (m) and the symmetry the completion
#: rule exploits. Values are ordinary building dimensions, not tuned
#: constants -- a door leaf is ~0.9 x 2.0 m wherever you are.
CATEGORY_PRIORS: dict[str, dict] = {
    "wall":            {"extent": (4.0, 0.2, 2.5), "rule": "planar_extension"},
    "ceiling":         {"extent": (4.0, 4.0, 0.1), "rule": "planar_extension"},
    "floor":           {"extent": (4.0, 4.0, 0.1), "rule": "planar_extension"},
    "door":            {"extent": (0.9, 0.06, 2.0), "rule": "planar_extension"},
    "safety_sign":     {"extent": (0.4, 0.02, 0.6), "rule": "planar_extension"},
    "pipe":            {"extent": (3.0, 0.15, 0.15), "rule": "axis_symmetric_extrusion"},
    "duct":            {"extent": (3.0, 0.5, 0.4), "rule": "axis_symmetric_extrusion"},
    "column":          {"extent": (0.5, 0.5, 2.5), "rule": "axis_symmetric_extrusion"},
    "cone":            {"extent": (0.36, 0.36, 0.5), "rule": "axis_symmetric_extrusion"},
    "furniture":       {"extent": (1.4, 0.8, 0.75), "rule": "prior_shape_fitting"},
    "equipment":       {"extent": (1.0, 0.8, 1.6), "rule": "prior_shape_fitting"},
    "stored_material": {"extent": (1.2, 1.0, 1.0), "rule": "prior_shape_fitting"},
}

#: Categories where missing geometry changes a navigation decision. A
#: half-observed wall or an unmarked cone matters; incomplete clutter in a
#: corner does not, and completing it would only add synthetic obstacles.
NAVIGATION_CRITICAL: frozenset[str] = frozenset({
    "door", "safety_sign", "cone", "pipe", "duct", "column", "wall",
    "stored_material", "equipment",
})


def _surface_area(points: np.ndarray, cell: float = 0.05) -> float:
    """Observed surface area, as the area of the occupied cells of the
    object's dominant projection.

    Counting points would measure scan density, not surface: the same
    door scanned twice as densely is not twice as complete.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 3:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    drop = int(np.argmin(extent))          # project along the thin axis
    plane = np.delete(points, drop, axis=1)
    cells = np.unique(np.floor(plane / cell).astype(int), axis=0)
    return float(len(cells)) * cell * cell


def _expected_area(category: str, cell: float = 0.05) -> float:
    prior = CATEGORY_PRIORS.get(category)
    if prior is None:
        return 0.0
    dims = sorted(prior["extent"], reverse=True)
    return float(dims[0] * dims[1])        # the two largest: the face we see


def completeness_score(spot_points: np.ndarray, category: str) -> float:
    """c(o): observed surface area vs. prior-aligned expected surface (Eq. 5)."""
    expected = _expected_area(category)
    if expected <= 0:
        return 1.0                          # no prior, nothing to compare
    return float(np.clip(_surface_area(spot_points) / expected, 0.0, 1.0))


def _plane_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and orthonormal axes (major, minor, normal) by PCA."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt


def complete_spot(
    spot_points: np.ndarray, category: str, params: CompletionParams
) -> np.ndarray:
    """Synthesize completion points for one spot via its category rule;
    returned points are flagged synthetic downstream.

    Each rule extends the observed geometry to the category prior *in the
    object's own frame*, and only outward from what was seen: nothing is
    invented where the scan saw empty space, and the synthetic fraction is
    capped so a fragment cannot become a whole object.
    """
    points = np.asarray(spot_points, dtype=float).reshape(-1, 3)
    prior = CATEGORY_PRIORS.get(category)
    if prior is None or len(points) < 8:
        return np.empty((0, 3))

    rule = prior["rule"]
    extent = np.asarray(prior["extent"], dtype=float)
    centre, axes = _plane_frame(points)
    local = (points - centre) @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    span = hi - lo
    rng = np.random.default_rng(0)
    target = np.sort(extent)[::-1]

    if rule == "planar_extension":
        # extend within the observed plane to the prior's face
        want_u, want_v = target[0], target[1]
        grow_u = max(want_u - span[0], 0.0) / 2.0
        grow_v = max(want_v - span[1], 0.0) / 2.0
        if grow_u < 0.02 and grow_v < 0.02:
            return np.empty((0, 3))
        n = min(int(len(points) * params.max_synthetic_ratio), 4000)
        u = rng.uniform(lo[0] - grow_u, hi[0] + grow_u, n)
        v = rng.uniform(lo[1] - grow_v, hi[1] + grow_v, n)
        w = rng.normal(0.0, max(span[2], 1e-3) / 4.0, n)
        keep = (u < lo[0]) | (u > hi[0]) | (v < lo[1]) | (v > hi[1])
        new_local = np.column_stack([u, v, w])[keep]
    elif rule == "axis_symmetric_extrusion":
        # extrude the observed cross-section along the object's own axis
        want = target[0]
        grow = max(want - span[0], 0.0) / 2.0
        if grow < 0.02:
            return np.empty((0, 3))
        n = min(int(len(points) * params.max_synthetic_ratio), 4000)
        idx = rng.integers(0, len(local), n)
        section = local[idx].copy()
        offset = rng.choice([-1.0, 1.0], n) * rng.uniform(0.0, grow, n)
        section[:, 0] = np.where(offset < 0, lo[0], hi[0]) + offset
        new_local = section
    else:  # prior_shape_fitting: fill the prior box's unobserved faces
        want = target
        grow = np.maximum(want[:3] - span, 0.0) / 2.0
        if float(grow.max()) < 0.02:
            return np.empty((0, 3))
        n = min(int(len(points) * params.max_synthetic_ratio), 4000)
        new_local = np.column_stack([
            rng.uniform(lo[k] - grow[k], hi[k] + grow[k], n) for k in range(3)
        ])
        inside = np.all((new_local >= lo) & (new_local <= hi), axis=1)
        new_local = new_local[~inside]

    if not len(new_local):
        return np.empty((0, 3))
    return new_local @ axes + centre


def emphasis_cue(
    category: str, completeness: float, params: CompletionParams,
    *, completed: bool,
) -> float:
    """phi_E for one spot (Sec. III-C).

    Emphasis is not a reward for having been completed: it marks geometry
    a planner should resolve finely -- synthesized surfaces, whose shape
    is inferred, and safety-relevant categories, whose shape matters --
    and it scales with how little was actually seen.
    """
    if completed:
        return float(np.clip(params.emphasis_strength * (1.0 - completeness), 0.0, 1.0))
    if category in ("cone", "safety_sign"):
        return float(np.clip(params.emphasis_strength * 0.5, 0.0, 1.0))
    return 0.0


def audit_and_complete(
    graph: BodyNavSceneGraph, points: LabeledPoints, params: CompletionParams,
    *, member_index: dict | None = None,
) -> tuple[LabeledPoints, np.ndarray, CompletionReport]:
    """Full pass: audit every spot, complete those below tau_c, and return
    (augmented points, per-point emphasis cue phi_E, report).

    phi_E is nonzero on completed spots and on safety-relevant categories
    (signage, hazards) regardless of completeness.
    """
    xyz = np.asarray(points.xyz, dtype=float)
    semantic = np.asarray(points.semantic, dtype=object)
    instance = np.asarray(points.instance_id, dtype=int)

    report = CompletionReport(completed_by_category={})
    before: list[float] = []
    after: list[float] = []
    new_points: list[np.ndarray] = []
    new_labels: list[str] = []
    new_instances: list[int] = []
    new_emphasis: list[float] = []
    emphasis = np.zeros(len(xyz), dtype=float)

    for spot_id in np.unique(instance):
        if spot_id < 0:
            continue
        sel = np.flatnonzero(instance == spot_id)
        category = str(semantic[sel[0]])
        if category not in CATEGORY_PRIORS:
            continue
        report.audited += 1
        c = completeness_score(xyz[sel], category)
        before.append(c)

        completed = False
        if c < params.tau_c and category in NAVIGATION_CRITICAL:
            report.incomplete += 1
            synthetic = complete_spot(xyz[sel], category, params)
            if len(synthetic):
                completed = True
                report.completed_by_category[category] = (
                    report.completed_by_category.get(category, 0) + 1)
                new_points.append(synthetic)
                new_labels += [category] * len(synthetic)
                new_instances += [int(spot_id)] * len(synthetic)
                c_after = completeness_score(
                    np.vstack([xyz[sel], synthetic]), category)
            else:
                c_after = c
        else:
            c_after = c
        after.append(c_after)

        phi = emphasis_cue(category, c, params, completed=completed)
        emphasis[sel] = phi
        if completed:
            new_emphasis += [phi] * len(new_points[-1])

    report.mean_completeness_before = float(np.mean(before)) if before else 0.0
    report.mean_completeness_after = float(np.mean(after)) if after else 0.0

    if not new_points:
        return points, emphasis, report

    synth = np.vstack(new_points)
    n_new = len(synth)
    augmented = LabeledPoints(
        xyz=np.vstack([xyz, synth]),
        floor_id=np.concatenate([points.floor_id, np.zeros(n_new, dtype=int)]),
        area_id=np.concatenate([points.area_id, np.full(n_new, -1, dtype=int)]),
        instance_id=np.concatenate([instance, np.asarray(new_instances, dtype=int)]),
        semantic=np.concatenate([semantic, np.asarray(new_labels, dtype=object)]),
        colors=None,
        extras={**points.extras,
                "synthetic": np.concatenate([np.zeros(len(xyz), dtype=bool),
                                             np.ones(n_new, dtype=bool)])},
    )
    return (augmented,
            np.concatenate([emphasis, np.asarray(new_emphasis, dtype=float)]),
            report)


def holdout_deviation(
    spot_points: np.ndarray, category: str, params: CompletionParams,
    *, fraction: float = 0.4, seed: int = 0,
) -> float | None:
    """Accuracy of completion, measured without a re-scan.

    A slice of the object's observed surface is withheld, the rest is
    completed by the category rule, and the synthesized points are
    compared against the withheld returns. This is the check that
    synthetic geometry is not fabricating obstacles: without it, Sec.
    III-C reports geometry nobody has verified. Returns the mean distance
    in metres, or None when the spot is too small to split.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(spot_points, dtype=float).reshape(-1, 3)
    if len(points) < 40:
        return None

    rng = np.random.default_rng(seed)
    centre, axes = _plane_frame(points)
    local = (points - centre) @ axes.T
    # withhold a contiguous end of the object, which is what occlusion
    # actually removes -- a random subset leaves the extent intact and
    # makes the completion look better than it is
    order = np.argsort(local[:, 0])
    cut = int(len(points) * (1.0 - fraction))
    observed, withheld = points[order[:cut]], points[order[cut:]]
    if len(observed) < 8 or len(withheld) < 8:
        return None

    synthetic = complete_spot(observed, category, params)
    if not len(synthetic):
        return None
    d, _ = cKDTree(synthetic).query(withheld, k=1)
    return float(np.mean(d))
