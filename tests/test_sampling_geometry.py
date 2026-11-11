"""The sampler and the judge must measure the same hazard geometry.

``route_choice`` decides whether a task is a fair test: it certifies that
the direct route violates the stated margin and that some route honours
it. ``hazard_compliance`` then scores whether the route actually did.
If those two measure different geometry the benchmark is incoherent --
tasks get certified as honourable against an easier condition than the
one they are graded on, and at large margins every method scores zero for
a reason that has nothing to do with any method.

That is exactly what happened: the sampler measured distance to hazard
*centroids* while the judge measured distance to their *bounding boxes*,
and a box is never further away than its own centre. These tests pin the
agreement rather than the implementation.
"""
from __future__ import annotations

import numpy as np

from bodynav.eval.sampling import _as_boxes


def test_as_boxes_accepts_a_box_unchanged():
    lo, hi = np.array([1.0, 2.0]), np.array([3.0, 5.0])
    (out_lo, out_hi), = _as_boxes([(lo, hi)])
    assert np.allclose(out_lo, lo)
    assert np.allclose(out_hi, hi)


def test_as_boxes_treats_a_point_as_a_zero_size_box():
    """Points remain usable, but as a degenerate box rather than silently."""
    (lo, hi), = _as_boxes([np.array([4.0, 7.0])])
    assert np.allclose(lo, [4.0, 7.0])
    assert np.allclose(hi, [4.0, 7.0])


def test_a_box_is_never_further_than_its_centre():
    """The property that made the two measures disagree.

    Distance from any query point to a box is bounded above by the
    distance to its centroid, so a feasibility test run on centroids
    admits margins the box-based judge will reject. The gap grows with
    the box, which is why the mismatch was invisible on small clutter and
    total on a 4 m stack of stored material.
    """
    rng = np.random.default_rng(0)
    lo = np.array([-1.0, -2.0])
    hi = np.array([2.0, 3.0])
    centre = 0.5 * (lo + hi)
    for q in rng.uniform(-10, 10, size=(200, 2)):
        to_box = np.linalg.norm(np.maximum(np.maximum(lo - q, q - hi), 0.0))
        to_centre = np.linalg.norm(q - centre)
        assert to_box <= to_centre + 1e-9


def test_box_footprint_is_rasterised_not_just_its_centre_cell():
    """A hazard occupies every cell it covers, not one.

    The seed grid `route_choice` builds is what its distance transform
    measures from, so marking only the centre cell of a 3 x 5 m stack
    would put the whole feasibility test 2.5 m off.
    """
    pitch = 0.5
    origin = np.array([0.0, 0.0])
    seeds = np.ones((20, 20), dtype=bool)
    for lo_xy, hi_xy in _as_boxes([(np.array([1.0, 1.0]), np.array([4.0, 6.0]))]):
        i0, j0 = np.floor((lo_xy - origin) / pitch).astype(int)
        i1, j1 = np.ceil((hi_xy - origin) / pitch).astype(int)
        seeds[i0:i1 + 1, j0:j1 + 1] = False
    marked = ~seeds
    assert marked.sum() > 1
    # every cell whose centre lies inside the box must be marked
    for i in range(marked.shape[0]):
        for j in range(marked.shape[1]):
            cx, cy = (i + 0.5) * pitch, (j + 0.5) * pitch
            if 1.0 <= cx <= 4.0 and 1.0 <= cy <= 6.0:
                assert marked[i, j], (i, j)
