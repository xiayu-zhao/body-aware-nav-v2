"""Stage 4: obstacles, the feasibility gate, value-guided A*, smoothing.

The fixture is a corridor whose geometry is known by construction: 20 m
long, 3 m wide, with a pillar in the middle that leaves 0.9 m either side
and a low soffit across one end. Every assertion is about a property the
manuscript claims -- the gate runs before the cost, the value map biases
the route, the shortcut keeps feasibility -- not about a tuned constant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bodynav.core.types import GridArrays, Priority, VoxelGrid, Voxel, as_arrays
from bodynav.llm.value_maps import ValueMapParams, build_value_maps
from bodynav.planning.astar import PlannerParams, ValueGuidedAStar
from bodynav.planning.obstacles import compute_clearance_fields
from bodynav.planning.postprocess import _segment_ok, revalidate, shortcut, smooth_bspline
from bodynav.robots import load_profile

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "robots"
PITCH = 0.1


def _grid_from_points(xyz, labels) -> VoxelGrid:
    grid = VoxelGrid(origin=np.zeros(3), base_pitch=0.05)
    for k, (p, lab) in enumerate(zip(xyz, labels)):
        grid.add(Voxel(index=(k, 0, 0), center=np.asarray(p, dtype=float),
                       resolution=0.05, semantic_label=lab))
    return grid


@pytest.fixture
def corridor_grid():
    """20 x 4 m corridor, walls at y = 0 and y = 4, pillar at x = 10.

    The pillar spans y = 1.6..2.4, leaving 1.6 m either side -- comfortably
    more than the Husky's 0.85 m, so a failure to find a path is a real
    failure and not a fixture that is too tight to squeeze through.
    """
    rng = np.random.default_rng(0)
    pts, labels = [], []

    # a regular floor lattice: sampled randomly, a 0.1 m planning cell can
    # miss its floor point entirely and the standing layer goes ragged --
    # fixture noise that looks exactly like a planner bug
    fx, fy = np.meshgrid(np.arange(0, 20, 0.05), np.arange(0, 4, 0.05))
    fx, fy = fx.ravel(), fy.ravel()
    pts.append(np.column_stack([fx, fy, np.zeros(len(fx))]))
    labels += ["floor"] * len(fx)

    for y in (0.0, 4.0):
        n = 4000
        pts.append(np.column_stack([rng.uniform(0, 20, n), np.full(n, y),
                                    rng.uniform(0.05, 2.5, n)]))
        labels += ["wall"] * n

    n = 1500  # pillar 0.8 m wide centred in the corridor
    pts.append(np.column_stack([rng.uniform(9.6, 10.4, n), rng.uniform(1.6, 2.4, n),
                                rng.uniform(0.05, 2.5, n)]))
    labels += ["column"] * n

    return _grid_from_points(np.vstack(pts), labels)


def _fields(grid, robot):
    return compute_clearance_fields(grid, [], robot, pitch=PITCH, floor_z=0.0)


def test_lateral_clearance_is_measured_in_plan(corridor_grid):
    """Not to the floor underfoot -- the mistake that rejects every cell."""
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    ij = f.index(np.array([[5.0, 2.0, 0.2]]))[0]
    # mid-corridor, 2 m from each wall
    assert f.lateral_grid[ij[0], ij[1], ij[2]] == pytest.approx(2.0, abs=0.2)


def test_gate_rejects_cells_narrower_than_the_body(corridor_grid):
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    planner = ValueGuidedAStar(PlannerParams(), robot)
    _, passable = planner._ground_layer(f)

    def at(x, y):
        ij = f.index(np.array([[x, y, 0.2]]))[0]
        return bool(passable[ij[0], ij[1]])

    assert at(5.0, 2.0)          # mid-corridor
    assert not at(5.0, 0.05)     # against the wall
    assert not at(10.0, 2.0)     # inside the pillar


def test_astar_routes_around_the_pillar(corridor_grid):
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    planner = ValueGuidedAStar(PlannerParams(), robot)
    lattice = _lattice(f, planner)
    maps = build_value_maps(lattice, goal_points=np.array([[18.0, 2.0, 0.0]]),
                            hazards=[], slope=np.zeros(len(lattice)),
                            width=np.full(len(lattice), 4.0),
                            priority=Priority.SAFETY_FIRST, params=ValueMapParams())
    path = planner.search(corridor_grid, maps, f,
                          np.array([2.0, 2.0, 0.1]), np.array([18.0, 2.0, 0.1]))
    assert path is not None
    assert path[0][0] < 3.0 and path[-1][0] > 17.0
    # it must pass the pillar on one side, never through it
    near = path[np.abs(path[:, 0] - 10.0) < 0.5]
    assert len(near)
    assert np.all((near[:, 1] < 1.6) | (near[:, 1] > 2.4))
    # and it is longer than the straight line because of the detour
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    assert length > 16.0


def test_no_path_when_the_corridor_is_blocked(corridor_grid):
    """A wall across the corridor is the correct answer 'infeasible'."""
    rng = np.random.default_rng(2)
    arrays = as_arrays(corridor_grid)
    n = 3000
    block = np.column_stack([rng.uniform(9.8, 10.2, n), rng.uniform(0, 4, n),
                             rng.uniform(0.05, 2.5, n)])
    grid = VoxelGrid(origin=np.zeros(3), base_pitch=0.05)
    for k, (p, lab) in enumerate(zip(np.vstack([arrays.center, block]),
                                     list(arrays.semantic_label) + ["wall"] * n)):
        grid.add(Voxel(index=(k, 0, 0), center=p, resolution=0.05, semantic_label=lab))

    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(grid, robot)
    planner = ValueGuidedAStar(PlannerParams(), robot)
    path = planner.search(grid, None, f, np.array([2.0, 2.0, 0.1]),
                          np.array([18.0, 2.0, 0.1]))
    assert path is None
    # Either phrasing is the same finding. With the goal snapped inside the
    # start's component, a wall across the corridor is reported at the snap
    # -- nothing on the far side is reachable -- rather than after an
    # exhaustive search that was always going to come back empty.
    assert any(m in planner.stats["reason"]
               for m in ("no feasible path", "no feasible cell"))


def _lattice(fields, planner) -> GridArrays:
    """The free cells of the search lattice, as a grid the value maps can
    be evaluated on -- what the driver does, and what makes V* non-zero
    where the robot actually travels."""
    _, passable = planner._ground_layer(fields)
    pi, pj = np.nonzero(passable)
    pk = np.clip(fields.ground_k + 1, 0, fields.occupied.shape[2] - 1)[pi, pj]
    ijk = np.column_stack([pi, pj, pk])
    centers = fields.world(ijk)
    n = len(centers)
    return GridArrays(
        origin=fields.origin, base_pitch=fields.pitch, index=ijk, center=centers,
        resolution=np.full(n, fields.pitch), occupancy=np.ones(n, dtype=int),
        semantic_label=np.full(n, "free", dtype=object),
        spot_id=np.full(n, None, dtype=object),
        synthetic=np.zeros(n, dtype=bool), confidence=np.ones(n),
    )


def test_value_map_biases_the_route(corridor_grid):
    """A hazard on one side pushes the path to the other."""
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    planner = ValueGuidedAStar(PlannerParams(eta=0.3), robot)
    lattice = _lattice(f, planner)

    def route(hazards):
        maps = build_value_maps(
            lattice, goal_points=np.array([[18.0, 2.0, 0.0]]), hazards=hazards,
            slope=np.zeros(len(lattice)), width=np.full(len(lattice), 4.0),
            priority=Priority.SAFETY_FIRST, params=ValueMapParams(),
        )
        return planner.search(corridor_grid, maps, f, np.array([2.0, 2.0, 0.1]),
                              np.array([18.0, 2.0, 0.1]))

    low = route([(np.array([10.0, 0.4, 0.5]), 1.0, 2.0)])   # hazard near y = 0
    high = route([(np.array([10.0, 3.6, 0.5]), 1.0, 2.0)])  # hazard near y = 4
    mid_low = low[np.abs(low[:, 0] - 10.0) < 1.0][:, 1].mean()
    mid_high = high[np.abs(high[:, 0] - 10.0) < 1.0][:, 1].mean()
    assert mid_low > mid_high, "the path should lean away from the hazard"


def test_shortcut_reduces_waypoints_and_keeps_feasibility(corridor_grid):
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    planner = ValueGuidedAStar(PlannerParams(), robot)
    raw = planner.search(corridor_grid, None, f, np.array([2.0, 2.0, 0.1]),
                         np.array([18.0, 2.0, 0.1]))
    short = shortcut(raw, corridor_grid, f, gate=planner.feasible)
    assert len(short) < len(raw)
    # Feasibility is defined over the cells the segment *traverses* (a DDA
    # walk), which is what the gate's clearance test is sized for.
    for a, b in zip(short[:-1], short[1:]):
        assert _segment_ok(a, b, f, planner.feasible)
    # and the endpoints are unchanged
    assert np.allclose(short[0], raw[0]) and np.allclose(short[-1], raw[-1])


def test_shortcut_without_a_gate_is_a_no_op():
    path = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert np.array_equal(shortcut(path, None, None), path)


def test_smoothing_respects_the_curvature_bound():
    path = np.column_stack([np.linspace(0, 10, 12), np.zeros(12), np.zeros(12)])
    smooth = smooth_bspline(path, max_curvature=1.0)
    assert len(smooth) >= len(path)
    assert np.allclose(smooth[:, 1], 0.0, atol=1e-6)

    zig = np.column_stack([np.arange(8), np.tile([0.0, 1.5], 4), np.zeros(8)])
    # a hairpin exceeds any tight bound, so the piecewise-linear path stands
    assert np.array_equal(smooth_bspline(zig, max_curvature=0.01), zig)


def test_revalidate_restores_the_feasible_path_when_smoothing_strays(corridor_grid):
    robot = load_profile("husky_ur5", config_dir=CONFIG_DIR)
    f = _fields(corridor_grid, robot)
    planner = ValueGuidedAStar(PlannerParams(), robot)
    original = np.array([[2.0, 2.0, 0.1], [10.0, 0.8, 0.1], [18.0, 2.0, 0.1]])
    through_wall = np.array([[2.0, 2.0, 0.1], [10.0, -1.0, 0.1], [18.0, 2.0, 0.1]])
    kept = revalidate(through_wall, original, corridor_grid, f, robot,
                      gate=planner.feasible)
    assert np.array_equal(kept, original)
