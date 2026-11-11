"""Smoke tests: the scaffold imports, data models behave, profiles load.

These must pass on a bare ``pip install -e .`` (no torch, no ROS 2, no
LLM key). Run with: pytest tests/ -q
"""

from pathlib import Path

import numpy as np
import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "robots"


def test_imports():
    import bodynav  # noqa: F401
    from bodynav.core import TaskSpec, ValueMapSet, VoxelGrid  # noqa: F401
    from bodynav.llm.validation import MaskValidationError  # noqa: F401
    from bodynav.planning import ValueGuidedAStar  # noqa: F401
    from bodynav.sim import OfflineBridge, SimBridge  # noqa: F401


def test_voxel_grid_roundtrip():
    from bodynav.core import VoxelGrid

    grid = VoxelGrid(origin=np.zeros(3), base_pitch=0.025)
    pts = np.array([[0.10, 0.20, 0.30], [1.00, 2.00, 3.00]])
    idx = grid.world2voxel(pts)
    back = grid.voxel2world(idx)
    # centers must lie within half a pitch of the original points
    assert np.all(np.abs(back - pts) <= grid.base_pitch / 2 + 1e-9)


@pytest.mark.parametrize("name", ["unitree_h1", "crazyflie21", "husky_ur5"])
def test_robot_profiles_load(name):
    from bodynav.robots import load_profile

    p = load_profile(name, config_dir=CONFIG_DIR)
    assert p.name == name
    lo, hi = p.height_band
    assert lo < hi
    if p.kind == "humanoid":
        assert p.foothold_band is not None
    if p.kind == "aerial":
        assert p.speed_limits is not None


def test_h1_matches_manuscript():
    from bodynav.robots import load_profile

    h1 = load_profile("unitree_h1", config_dir=CONFIG_DIR)
    # The manuscript says 2.0 m, which allowed 0.2 m above the 1.8 m
    # stance for a raised arm or carried load. That allowance is retired
    # (2026-09-09): the arm is never raised in this study, so the envelope
    # is the stance plus 0.05 m of gait margin. root.tex must be updated
    # to match before release -- this assertion is the reminder.
    assert h1.height_band == (0.0, 1.85)
    assert h1.required_vertical_clearance == 1.85
    assert h1.r_reach == 0.8 and h1.r_safety == 0.5
    assert h1.max_slope_deg == 20.0
    assert h1.min_lateral_clearance == 0.6
    assert h1.foothold_band == 0.3


def test_value_map_validation_rejects_bad_shapes():
    from bodynav.core import ValueMapSet

    n = 4
    ok = ValueMapSet(
        indices=np.zeros((n, 3), dtype=int),
        affordance=np.zeros(n),
        hazard=np.zeros(n),
        constraint=np.zeros(n),
        fused=np.zeros(n),
    )
    ok.validate()

    bad = ValueMapSet(
        indices=np.zeros((n, 3), dtype=int),
        affordance=np.zeros(n + 1),  # wrong shape
        hazard=np.zeros(n),
        constraint=np.zeros(n),
        fused=np.zeros(n),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_fusion_weights_match_manuscript():
    from bodynav.core.types import Priority
    from bodynav.llm.value_maps import FUSION_WEIGHTS

    assert FUSION_WEIGHTS[Priority.SAFETY_FIRST] == (0.3, 0.6, 0.1)
    assert FUSION_WEIGHTS[Priority.EFFICIENCY_FIRST] == (0.6, 0.2, 0.2)
    assert FUSION_WEIGHTS[Priority.AREA_PREFERENCE] == (0.4, 0.3, 0.3)
