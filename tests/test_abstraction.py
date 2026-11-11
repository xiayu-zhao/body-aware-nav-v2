"""Stage 2 tests: PLY I/O, the Eq. 2 slice, and Eq. 4 voxelization.

The scene fixture is a synthetic corridor with a known ground plane, a
known slope, and a known overhead run of pipe, so the expected slice and
resolution assignment can be asserted exactly rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodynav.abstraction import (
    AdaptiveVoxelizer,
    VoxelizerParams,
    height_slice,
    local_ground_height,
    workspace_kernel_mask,
)
from bodynav.io.ply import PlyError, read_ply
from bodynav.io.pointcloud import cloud_info, load_cloud, voxel_downsample
from bodynav.robots import load_profile
from bodynav.scene_graph.hierarchy import LabeledPoints

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "robots"


@pytest.fixture
def husky():
    """The Husky+UR5 — both the survey platform that captured the lab-floor
    scan and the wheeled platform BodyNav plans for, so it replaces the old
    stand-in `ugv` profile everywhere."""
    return load_profile("husky_ur5", config_dir=CONFIG_DIR)


@pytest.fixture
def corridor():
    """20 x 4 m corridor: flat floor, two walls, an overhead pipe at 2.4 m."""
    rng = np.random.default_rng(0)
    floor = np.column_stack([
        rng.uniform(0, 20, 4000), rng.uniform(0, 4, 4000), np.zeros(4000),
    ])
    wall_a = np.column_stack([
        rng.uniform(0, 20, 2000), np.zeros(2000), rng.uniform(0, 3, 2000),
    ])
    wall_b = np.column_stack([
        rng.uniform(0, 20, 2000), np.full(2000, 4.0), rng.uniform(0, 3, 2000),
    ])
    pipe = np.column_stack([
        rng.uniform(0, 20, 1000), np.full(1000, 2.0), np.full(1000, 2.4),
    ])
    xyz = np.vstack([floor, wall_a, wall_b, pipe])
    semantic = np.array(
        ["floor"] * 4000 + ["wall"] * 4000 + ["pipe"] * 1000, dtype=object
    )
    return LabeledPoints.from_xyz(xyz, semantic=semantic)


# -- PLY I/O ------------------------------------------------------------
def _write_ascii_ply(path, xyz, colors=None):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}",
             "property float x", "property float y", "property float z"]
    if colors is not None:
        lines += ["property uchar red", "property uchar green",
                  "property uchar blue"]
    lines.append("end_header")
    for i, p in enumerate(xyz):
        row = f"{p[0]} {p[1]} {p[2]}"
        if colors is not None:
            c = colors[i]
            row += f" {c[0]} {c[1]} {c[2]}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_binary_ply(path, xyz, colors=None, big_endian=False):
    order = ">" if big_endian else "<"
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    header = ["ply",
              f"format binary_{'big' if big_endian else 'little'}_endian 1.0",
              f"element vertex {len(xyz)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    header.append("end_header")
    dtype = np.dtype([(n, order + t) for n, t in fields])
    rec = np.zeros(len(xyz), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if colors is not None:
        rec["red"], rec["green"], rec["blue"] = (
            colors[:, 0], colors[:, 1], colors[:, 2])
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rec.tobytes())


@pytest.mark.parametrize("big_endian", [False, True])
def test_binary_ply_roundtrip(tmp_path, big_endian):
    xyz = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-1.5, 0.25, 9.0]])
    colors = np.array([[255, 0, 0], [0, 128, 255], [10, 20, 30]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    _write_binary_ply(path, xyz, colors, big_endian=big_endian)

    got_xyz, got_colors = load_cloud(path)
    assert np.allclose(got_xyz, xyz)
    assert got_colors is not None
    assert np.allclose(got_colors, colors / 255.0)


def test_ascii_ply_roundtrip(tmp_path):
    xyz = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    path = tmp_path / "cloud.ply"
    _write_ascii_ply(path, xyz)
    got_xyz, got_colors = load_cloud(path)
    assert np.allclose(got_xyz, xyz)
    assert got_colors is None


def test_ply_subsampling(tmp_path):
    xyz = np.arange(300, dtype=float).reshape(100, 3)
    path = tmp_path / "cloud.ply"
    _write_binary_ply(path, xyz)

    assert cloud_info(path)["n_points"] == 100
    assert len(load_cloud(path, stride=4)[0]) == 25
    # max_points caps the count without the caller computing a stride
    assert len(load_cloud(path, max_points=10)[0]) <= 10
    assert np.allclose(load_cloud(path, stride=4)[0], xyz[::4])


def test_ply_rejects_non_ply(tmp_path):
    path = tmp_path / "bogus.ply"
    path.write_bytes(b"CCB2\x00\x00\x00\x00not a ply at all")
    with pytest.raises(PlyError):
        read_ply(path)


def test_up_axis_rotation_is_right_handed():
    from bodynav.io.pointcloud import to_z_up

    pts = np.array([[1.0, 2.0, 3.0]])
    # Y-up: the old vertical (y=2) must become the new z
    rotated = to_z_up(pts, "y")
    assert rotated[0, 2] == pytest.approx(2.0)
    assert to_z_up(pts, "x")[0, 2] == pytest.approx(1.0)
    assert np.allclose(to_z_up(pts, "z"), pts)

    for axis in ("x", "y"):
        r = to_z_up(np.eye(3), axis)
        assert np.linalg.det(r) == pytest.approx(1.0)  # no mirroring
        assert np.allclose(np.linalg.norm(r, axis=1), 1.0)

    with pytest.raises(ValueError):
        to_z_up(pts, "w")


def test_detect_up_axis_finds_the_floor():
    from bodynav.io.pointcloud import detect_up_axis

    rng = np.random.default_rng(3)
    n = 20000
    # a floor slab: spread wide in x/z, concentrated in y
    xyz = np.column_stack([
        rng.uniform(0, 30, n),
        rng.normal(0, 0.01, n),
        rng.uniform(0, 30, n),
    ])
    assert detect_up_axis(xyz) == "y"
    assert detect_up_axis(xyz[:, [0, 2, 1]]) == "z"


def test_load_cloud_applies_up_axis(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "cloud.ply"
    _write_binary_ply(path, xyz)
    got, _ = load_cloud(path, up_axis="y")
    assert np.allclose(got[:, 2], xyz[:, 1])


def test_voxel_downsample_reduces_and_stays_in_bounds():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 1, size=(5000, 3))
    out, _ = voxel_downsample(xyz, 0.1)
    assert len(out) < len(xyz)
    assert out.min() >= xyz.min() - 1e-9 and out.max() <= xyz.max() + 1e-9


# -- Eq. 2: robot slice -------------------------------------------------
def test_local_ground_tracks_a_slope():
    """A 10% ramp must be tracked, not flattened to a single global min."""
    x = np.linspace(0, 20, 5000)
    xyz = np.column_stack([x, np.zeros_like(x), 0.1 * x])
    z_gnd = local_ground_height(xyz, cell=0.5)
    # ground follows the ramp to within one cell's rise
    assert np.max(np.abs(z_gnd - 0.1 * x)) < 0.1 * 0.5 + 1e-6


def test_local_ground_ignores_below_floor_outliers():
    rng = np.random.default_rng(2)
    xyz = np.column_stack([
        rng.uniform(0, 5, 2000), rng.uniform(0, 5, 2000), np.zeros(2000),
    ])
    xyz[0, 2] = -3.0  # reflective-surface artifact
    strict = local_ground_height(xyz, cell=0.5, quantile=0.0)
    robust = local_ground_height(xyz, cell=0.5, quantile=0.02)
    assert strict.min() == pytest.approx(-3.0)
    # the artifact is discarded and the true floor is recovered exactly
    assert robust.min() == pytest.approx(0.0)
    assert np.all(robust == 0.0)


def test_height_slice_keeps_band_and_thins_overhead(corridor, husky):
    sliced = height_slice(corridor, husky, overhead_keep_ratio=0.05, seed=0)
    z_rel = sliced.extras["z_rel"]
    h_min, h_max = husky.height_band

    in_band = sliced.extras["in_band"]
    # everything kept is either inside the band or overhead, never below
    assert np.all(z_rel[in_band] <= h_max + 1e-9)
    assert np.all(z_rel >= h_min - 1e-9)

    # the 1000-point pipe at 2.4 m sits above the 1.2 m band: ~5% survives
    kept_pipe = np.sum(sliced.semantic.astype(str) == "pipe")
    assert 10 < kept_pipe < 120

    # per-point arrays stay aligned after selection
    assert len(sliced.xyz) == len(sliced.semantic) == len(z_rel)
    assert len(sliced.xyz) < len(corridor.xyz)


def test_height_slice_reduction_in_manuscript_range(corridor, husky):
    """Sec. III-B.1: reduction from the slice alone. The manuscript's
    60-80% is being restated as 42-62% -- the fraction scales with the
    ratio of the operational band to the storey height -- so this bound
    stays deliberately loose."""
    sliced = height_slice(corridor, husky, overhead_keep_ratio=0.05)
    reduction = 1 - len(sliced.xyz) / len(corridor.xyz)
    assert 0.3 < reduction < 0.9


def test_workspace_kernel(husky):
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # H1's kernel is r_reach + r_safety = 0.8 + 0.5 m
    h1 = load_profile("unitree_h1", config_dir=CONFIG_DIR)
    mask = workspace_kernel_mask(xyz, np.array([0.0, 0.0]), h1)
    assert mask.tolist() == [True, False]

    # union over a trajectory of poses
    mask = workspace_kernel_mask(
        xyz, np.array([[0.0, 0.0], [10.0, 0.0]]), h1
    )
    assert mask.tolist() == [True, True]

    # Eq. 3's radius is r_reach + r_safety, so a manipulator reaches past
    # what the safety envelope alone covers. Both H1 (0.8 + 0.5) and the
    # Husky+UR5 (0.85 + 0.4) have one; the Crazyflie (r_reach = 0) does not.
    for robot in (h1, husky):
        r = robot.r_reach + robot.r_safety
        probe = np.array([[robot.r_safety + 0.05, 0.0, 0.0], [r + 0.05, 0.0, 0.0]])
        assert workspace_kernel_mask(
            probe, np.array([0.0, 0.0]), robot
        ).tolist() == [True, False]

    # The Crazyflie carries no manipulator (r_reach = 0), so its kernel
    # collapses to the safety envelope: a point the two manipulator
    # platforms cover sits outside it.
    uav = load_profile("crazyflie21", config_dir=CONFIG_DIR)
    probe = np.array([[uav.r_safety + 0.05, 0.0, 0.0]])
    assert workspace_kernel_mask(probe, np.array([0.0, 0.0]), uav).tolist() == [False]
    assert workspace_kernel_mask(probe, np.array([0.0, 0.0]), h1).tolist() == [True]


# -- Eq. 4: adaptive voxelization ---------------------------------------
def test_resolution_ladder_is_dyadic_and_capped():
    v = AdaptiveVoxelizer(VoxelizerParams(), load_profile("husky_ur5", config_dir=CONFIG_DIR))
    levels = v.levels()
    assert levels[0] == pytest.approx(0.025)
    assert np.all(levels <= 0.5)
    assert levels[-1] == pytest.approx(0.4)
    assert np.allclose(levels[1:] / levels[:-1], 2.0)


def test_phi_robot_peaks_near_ground(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    phi = v.phi_robot(sliced)
    z_rel = sliced.extras["z_rel"]

    assert np.all((phi >= 0) & (phi <= 1))
    assert phi[z_rel < 0.2].max() == pytest.approx(1.0)
    # above the 1.2 m band phi decays
    above = z_rel > husky.height_band[1] + 0.6
    if above.any():
        assert phi[above].max() < 1e-9


def test_phi_task_is_one_inside_mask_and_decays_outside(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(task_falloff=1.0), husky)
    mask = corridor.xyz[:, 0] < 1.0  # first metre of the corridor
    phi = v.phi_task(corridor, mask)

    assert np.all(phi[mask] == 1.0)
    assert np.all((phi >= 0) & (phi <= 1))
    far = corridor.xyz[:, 0] > 5.0
    assert np.all(phi[far] == 0.0)
    # a band just outside the mask is partially refined
    near = (corridor.xyz[:, 0] >= 1.0) & (corridor.xyz[:, 0] < 1.5)
    assert np.any(phi[near] > 0)


def test_phi_task_rejects_wrong_shape(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    with pytest.raises(ValueError):
        v.phi_task(corridor, np.zeros(7, dtype=bool))


def test_voxelize_produces_multiple_resolutions(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    mask = sliced.xyz[:, 0] < 2.0
    grid = v.voxelize(sliced, mask)

    assert len(grid) > 0
    resolutions = {vox.resolution for vox in grid}
    assert len(resolutions) > 1, "adaptive grid collapsed to one resolution"
    assert min(resolutions) >= 0.025 - 1e-9
    assert max(resolutions) <= 0.4 + 1e-9

    # voxels inside the task mask are finer than distant background
    inside = [vox.resolution for vox in grid if vox.center[0] < 2.0]
    outside = [vox.resolution for vox in grid if vox.center[0] > 10.0]
    assert np.mean(inside) < np.mean(outside)


def test_voxelize_indices_are_on_the_base_lattice(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    grid = v.voxelize(sliced, np.zeros(len(sliced.xyz), dtype=bool))

    for vox in grid:
        scale = int(round(vox.resolution / grid.base_pitch))
        # a voxel's stored index is its own cell's corner on the base lattice
        assert all(i % scale == 0 for i in vox.index)
        corner = grid.origin + np.array(vox.index) * grid.base_pitch
        assert np.allclose(vox.center, corner + vox.resolution / 2.0)


def test_every_point_lands_in_a_voxel(corridor, husky):
    """No geometry may be dropped: each surviving point must fall inside
    exactly one voxel of the sparse grid."""
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    # a task mask forces several levels to coexist, which is where the
    # coarse/fine boundary handling can lose points
    mask = sliced.xyz[:, 0] < 3.0
    grid = v.voxelize(sliced, mask)

    levels = v.levels()
    hits = np.zeros(len(sliced.xyz), dtype=int)
    for pitch in levels:
        scale = int(round(pitch / grid.base_pitch))
        cells = np.floor((sliced.xyz - grid.origin) / pitch).astype(int) * scale
        for i, idx in enumerate(map(tuple, cells)):
            vox = grid.get(idx)
            if vox is not None and abs(vox.resolution - pitch) < 1e-9:
                hits[i] += 1

    assert np.all(hits >= 1), f"{np.sum(hits == 0)} points fell outside every voxel"
    assert np.all(hits <= 1), f"{np.sum(hits > 1)} points landed in overlapping voxels"


def test_voxelize_reduces_point_count(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    grid = v.voxelize(sliced, np.zeros(len(sliced.xyz), dtype=bool))

    stats = v.last_stats
    assert stats.n_voxels == len(grid)
    assert stats.n_points == len(sliced.xyz)
    assert stats.reduction > 0.0
    assert sum(stats.voxels_by_resolution.values()) == len(grid)


def test_emphasis_refines_where_applied(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    zeros = np.zeros(len(sliced.xyz), dtype=bool)

    baseline = v.voxelize(sliced, zeros)
    base_mean = v.last_stats.mean_resolution

    emphasis = (sliced.xyz[:, 0] < 5.0).astype(float)
    v.voxelize(sliced, zeros, emphasis=emphasis)
    assert v.last_stats.mean_resolution < base_mean
    assert len(baseline) > 0


def test_emphasis_rejects_wrong_shape(corridor, husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    with pytest.raises(ValueError):
        v.voxelize(sliced, np.zeros(len(sliced.xyz), dtype=bool),
                   emphasis=np.zeros(3))


def test_hazard_labels_win_the_vote():
    labels = np.array(["floor"] * 5 + ["welding_zone"] * 2, dtype=object)
    weights = {"welding_zone": 4.0}
    # 5 floor points at weight 1 lose to 2 hazard points at weight 4
    assert AdaptiveVoxelizer.vote_label(labels, weights) == "welding_zone"
    assert AdaptiveVoxelizer.vote_label(labels, {}) == "floor"
    assert AdaptiveVoxelizer.vote_label(np.array([], dtype=object), {}) == "unknown"


def test_empty_cloud_is_handled(husky):
    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    empty = LabeledPoints.from_xyz(np.zeros((0, 3)))
    grid = v.voxelize(empty, np.zeros(0, dtype=bool))
    assert len(grid) == 0
    assert v.last_stats.reduction == 0.0


def test_grid_roundtrips_through_disk(tmp_path, corridor, husky):
    from bodynav.io import load_grid, save_grid

    v = AdaptiveVoxelizer(VoxelizerParams(), husky)
    sliced = height_slice(corridor, husky)
    grid = v.voxelize(sliced, sliced.xyz[:, 0] < 2.0)

    path = tmp_path / "grid.npz"
    save_grid(grid, path)
    back = load_grid(path)

    assert len(back) == len(grid)
    assert np.allclose(back.origin, grid.origin)
    assert back.base_pitch == pytest.approx(grid.base_pitch)
    for idx, vox in grid.voxels.items():
        other = back.get(idx)
        assert other is not None
        assert other.resolution == pytest.approx(vox.resolution)
        assert other.semantic_label == vox.semantic_label


# -- PLY writing --------------------------------------------------------
def test_write_ply_roundtrips_positions_and_colors(tmp_path):
    from bodynav.io.ply import read_header, write_ply

    rng = np.random.default_rng(3)
    xyz = rng.uniform(-40, 40, (500, 3))
    rgb = rng.integers(0, 256, (500, 3), dtype=np.uint8)
    path = write_ply(tmp_path / "c.ply", xyz, rgb, comments=["stage=x", "prim=y"])

    header = read_header(path)
    assert header.fmt == "binary_little_endian"
    back, fields = read_ply(path)
    # float32 storage, so compare at single precision
    assert np.allclose(back, xyz, atol=1e-4)
    assert np.array_equal(
        np.stack([fields["red"], fields["green"], fields["blue"]], axis=1), rgb
    )


def test_write_ply_accepts_float_colors_and_rejects_mismatch(tmp_path):
    from bodynav.io.ply import write_ply

    xyz = np.zeros((4, 3))
    path = write_ply(tmp_path / "f.ply", xyz, np.ones((4, 3)))
    _, fields = read_ply(path)
    assert np.array_equal(fields["red"], np.full(4, 255, dtype=np.uint8))

    with pytest.raises(PlyError):
        write_ply(tmp_path / "bad.ply", xyz, np.ones((3, 3)))
