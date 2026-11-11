"""Stage 2b: completeness scoring, category-conditioned completion, phi_E.

Completion synthesizes geometry nobody observed, so the tests are mostly
about restraint: it must not fire above tau_c, must not exceed the
synthetic cap, must not invent an object from a fragment, and what it
does synthesize must land near the surface it is standing in for.
"""
from __future__ import annotations

import numpy as np
import pytest

from bodynav.abstraction.completion import (
    CATEGORY_PRIORS,
    NAVIGATION_CRITICAL,
    CompletionParams,
    audit_and_complete,
    complete_spot,
    completeness_score,
    emphasis_cue,
    holdout_deviation,
)
from bodynav.scene_graph.attributes import Affordances, NodeAttributes
from bodynav.scene_graph.graph import BodyNavSceneGraph
from bodynav.scene_graph.hierarchy import LabeledPoints

PARAMS = CompletionParams()


def _door(fraction: float = 1.0, seed: int = 0) -> np.ndarray:
    """A door leaf, of which only ``fraction`` along its width was seen."""
    rng = np.random.default_rng(seed)
    n = 4000
    return np.column_stack([
        rng.uniform(0.0, 0.9 * fraction, n),
        rng.normal(0.0, 0.01, n),
        rng.uniform(0.0, 2.0, n),
    ])


def _pipe(length: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 3000
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        rng.uniform(0.0, length, n),
        0.075 * np.cos(theta),
        0.075 * np.sin(theta) + 2.4,
    ])


# -- completeness -------------------------------------------------------
def test_a_whole_door_scores_complete():
    assert completeness_score(_door(1.0), "door") == pytest.approx(1.0, abs=0.15)


def test_a_third_of_a_door_scores_about_a_third():
    c = completeness_score(_door(0.33), "door")
    assert 0.2 < c < 0.5


def test_score_is_about_surface_not_density():
    """Scanning the same door twice as densely does not complete it."""
    sparse = _door(0.5, seed=1)
    dense = np.vstack([sparse, sparse + np.array([0.001, 0.0, 0.001])])
    assert completeness_score(dense, "door") == pytest.approx(
        completeness_score(sparse, "door"), abs=0.05)


def test_unknown_category_is_not_judged():
    assert completeness_score(_door(0.1), "not_a_category") == 1.0


# -- completion rules ---------------------------------------------------
def test_planar_extension_widens_a_partial_door():
    observed = _door(0.4)
    made = complete_spot(observed, "door", PARAMS)
    assert len(made)
    # the synthesized points extend the observed face, and stay in its plane
    assert made[:, 0].max() > observed[:, 0].max()
    assert np.abs(made[:, 1]).max() < 0.15
    assert completeness_score(np.vstack([observed, made]), "door") > \
        completeness_score(observed, "door")


def test_axis_symmetric_extrusion_lengthens_a_truncated_pipe():
    observed = _pipe(length=1.0)
    made = complete_spot(observed, "pipe", PARAMS)
    assert len(made)
    assert made[:, 0].max() > observed[:, 0].max() or made[:, 0].min() < observed[:, 0].min()
    # the cross-section is preserved: the extrusion does not fatten the pipe
    radial = np.linalg.norm(made[:, 1:] - np.array([0.0, 2.4]), axis=1)
    assert radial.max() < 0.2


def test_completion_respects_the_synthetic_cap():
    observed = _door(0.3)
    strict = CompletionParams(max_synthetic_ratio=0.1)
    made = complete_spot(observed, "door", strict)
    assert len(made) <= 0.1 * len(observed) + 1


def test_a_complete_object_gains_nothing():
    assert len(complete_spot(_pipe(length=3.0), "pipe", PARAMS)) == 0


def test_a_fragment_is_not_expanded_into_an_object():
    """Fewer than a handful of returns is not evidence of a door."""
    assert len(complete_spot(_door(0.4)[:5], "door", PARAMS)) == 0


# -- emphasis -----------------------------------------------------------
def test_emphasis_scales_with_how_little_was_seen():
    high = emphasis_cue("door", 0.2, PARAMS, completed=True)
    low = emphasis_cue("door", 0.65, PARAMS, completed=True)
    assert 0.0 < low < high <= 1.0


def test_safety_categories_are_emphasised_even_when_complete():
    assert emphasis_cue("cone", 1.0, PARAMS, completed=False) > 0
    assert emphasis_cue("furniture", 1.0, PARAMS, completed=False) == 0


# -- audit --------------------------------------------------------------
@pytest.fixture
def scene():
    door, pipe = _door(0.35), _pipe(length=0.8)
    xyz = np.vstack([door, pipe])
    instance = np.concatenate([np.zeros(len(door), int), np.ones(len(pipe), int)])
    semantic = np.array(["door"] * len(door) + ["pipe"] * len(pipe), dtype=object)
    points = LabeledPoints(xyz=xyz, floor_id=np.zeros(len(xyz), int),
                           area_id=np.zeros(len(xyz), int), instance_id=instance,
                           semantic=semantic)
    attrs = {"floor_0": NodeAttributes(centroid=np.zeros(3), aabb_min=np.zeros(3),
                                       aabb_max=np.ones(3), semantic="floor",
                                       affordances=Affordances())}
    return BodyNavSceneGraph.build(points, attrs), points


def test_audit_completes_both_and_reports(scene):
    graph, points = scene
    augmented, emphasis, report = audit_and_complete(graph, points, PARAMS)
    assert report.audited == 2 and report.incomplete == 2
    assert set(report.completed_by_category) == {"door", "pipe"}
    assert len(augmented.xyz) > len(points.xyz)
    assert report.mean_completeness_after > report.mean_completeness_before


def test_emphasis_is_aligned_with_the_augmented_cloud(scene):
    graph, points = scene
    augmented, emphasis, _ = audit_and_complete(graph, points, PARAMS)
    assert emphasis.shape == (len(augmented.xyz),)
    assert np.all(emphasis >= 0) and np.all(emphasis <= 1)
    # the synthetic points carry the cue too, or the voxelizer cannot see it
    assert emphasis[len(points.xyz):].max() > 0


def test_synthetic_points_are_flagged(scene):
    graph, points = scene
    augmented, _, _ = audit_and_complete(graph, points, PARAMS)
    flag = augmented.extras["synthetic"]
    assert flag.shape == (len(augmented.xyz),)
    assert not flag[:len(points.xyz)].any() and flag[len(points.xyz):].all()


def test_nothing_is_completed_above_tau_c(scene):
    graph, points = scene
    lenient = CompletionParams(tau_c=0.0)
    augmented, _, report = audit_and_complete(graph, points, lenient)
    assert report.incomplete == 0
    assert len(augmented.xyz) == len(points.xyz)


def test_non_critical_categories_are_left_alone():
    """Incomplete clutter in a corner is not worth synthesizing."""
    assert "clutter" not in NAVIGATION_CRITICAL
    assert "clutter" not in CATEGORY_PRIORS


# -- holdout accuracy ---------------------------------------------------
def test_holdout_recovers_a_withheld_end():
    """A pipe whose end is hidden should be rebuilt close to the truth."""
    dev = holdout_deviation(_pipe(length=2.0), "pipe", PARAMS)
    assert dev is not None
    assert dev < 0.5


def test_holdout_declines_when_there_is_too_little_to_split():
    assert holdout_deviation(_door(0.4)[:20], "door", PARAMS) is None
