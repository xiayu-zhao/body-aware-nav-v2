"""Point-cloud loading, subsampling, and multi-session registration.

Supports the two acquisition modes in the manuscript: simulation scans
exported from Isaac Sim and real multi-session UGV scans (e.g. the
lab-floor dataset: several .ply sessions registered into one colored
cloud).

.ply is read by the built-in pure-numpy reader (``bodynav.io.ply``) so no
heavyweight geometry stack is required; other formats fall back to
open3d when it happens to be installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bodynav.io.ply import extract_colors, read_header, read_ply


#: Rotations taking a source frame's up-axis onto +Z, right-handed.
#: Handheld/ARKit-derived scanners (Scaniverse, Polycam) export Y-up; every
#: BodyNav stage assumes Z-up because Eq. 2 slices on height.
_UP_AXIS_ROTATIONS: dict[str, np.ndarray] = {
    "z": np.eye(3),
    # +90 deg about X: (x, y, z) -> (x, -z, y)
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    # -90 deg about Y: (x, y, z) -> (-z, y, x)
    "x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
}


def to_z_up(xyz: np.ndarray, up_axis: str) -> np.ndarray:
    """Rotate a cloud so ``up_axis`` becomes +Z, preserving handedness."""
    key = up_axis.lower()
    if key not in _UP_AXIS_ROTATIONS:
        raise ValueError(f"up_axis must be one of x, y, z; got {up_axis!r}")
    if key == "z":
        return xyz
    return np.asarray(xyz, dtype=np.float64) @ _UP_AXIS_ROTATIONS[key].T


def detect_up_axis(xyz: np.ndarray, *, bin_size: float = 0.01) -> str:
    """Guess the vertical axis from point concentration.

    Floors and ceilings pack a large share of an indoor scan into a few
    height bins, while the two horizontal axes spread out. The axis whose
    1 cm histogram is most concentrated is the vertical one. Used only to
    warn about a mismatched ``up_axis``, never to override the caller.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    scores = []
    for ax in range(3):
        v = xyz[:, ax]
        span = v.max() - v.min()
        if span <= 0:
            scores.append(np.inf)
            continue
        counts, _ = np.histogram(v, bins=max(int(span / bin_size), 1))
        share = np.sort(counts / counts.sum())[::-1]
        # number of bins needed to cover half the points; smaller = flatter
        scores.append(int(np.searchsorted(np.cumsum(share), 0.5)) + 1)
    return "xyz"[int(np.argmin(scores))]


def cloud_info(path: str | Path) -> dict:
    """Cheap metadata probe: point count and available vertex fields.

    Reads only the header, so it is instant even on multi-gigabyte scans
    and is the right way to size a subsample before loading.
    """
    path = Path(path)
    header = read_header(path)
    el = header.element("vertex")
    if el is None:
        raise ValueError(f"{path}: no 'vertex' element")
    return {
        "path": str(path),
        "format": header.fmt,
        "n_points": el.count,
        "fields": [p.name for p in el.properties],
        "file_bytes": path.stat().st_size,
    }


def load_cloud(
    path: str | Path,
    *,
    stride: int = 1,
    max_points: int | None = None,
    up_axis: str = "z",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a .ply/.pcd cloud; returns (xyz (N, 3), colors (N, 3) | None).

    Args:
        stride: keep every ``stride``-th point.
        max_points: cap the returned point count by raising ``stride``.
            Useful for iterating on the multi-gigabyte lab-floor scan.
        up_axis: vertical axis of the *source* file. Anything other than
            ``"z"`` is rotated onto Z-up, since Eq. 2 slices on height.

    Colors are float in [0, 1] when present.
    """
    path = Path(path)
    if path.suffix.lower() == ".ply":
        xyz, fields = read_ply(path, stride=stride, max_points=max_points)
        return to_z_up(xyz, up_axis), extract_colors(fields)

    import open3d as o3d  # optional; only for non-PLY inputs

    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    if stride > 1 or max_points:
        n = len(xyz)
        step = stride
        if max_points and n > max_points:
            step = max(step, -(-n // max_points))
        xyz = xyz[::step]
        colors = colors[::step] if colors is not None else None
    return to_z_up(xyz, up_axis), colors


def voxel_downsample(
    xyz: np.ndarray, pitch: float, *, colors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Grid-average downsample: one representative point per ``pitch`` cell.

    Pure numpy (bincount over linearized cell ids); used to bring raw scan
    density down to a workable level before the abstraction stage.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size == 0:
        return xyz.reshape(0, 3), (None if colors is None else colors[:0])
    if pitch <= 0:
        raise ValueError(f"pitch must be > 0, got {pitch}")

    idx = np.floor((xyz - xyz.min(axis=0)) / pitch).astype(np.int64)
    dims = idx.max(axis=0) + 1
    keys = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)

    out = np.stack(
        [np.bincount(inv, weights=xyz[:, d], minlength=len(uniq)) / counts
         for d in range(3)],
        axis=1,
    )
    out_colors = None
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        out_colors = np.stack(
            [np.bincount(inv, weights=colors[:, c], minlength=len(uniq)) / counts
             for c in range(colors.shape[1])],
            axis=1,
        )
    return out, out_colors


def register_sessions(paths: list[str | Path], *, voxel_down: float = 0.02) -> np.ndarray:
    """Register multiple scan sessions into one cloud (coarse global +
    ICP refinement); used for the real-world multi-session UGV scans.

    Not implemented: the lab-floor dataset (Sec. VI) is registered in
    CloudCompare and exported as a single merged cloud, so BodyNav consumes
    it via ``load_cloud``. This entry point exists for scans that arrive
    unregistered.
    """
    raise NotImplementedError(
        "sessions are pre-registered in CloudCompare; load the merged .ply"
    )


def save_grid(grid, path: str | Path) -> None:
    """Serialize a VoxelGrid (npz) for caching between pipeline stages."""
    path = Path(path)
    voxels = list(grid.voxels.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin=np.asarray(grid.origin, dtype=np.float64),
        base_pitch=np.float64(grid.base_pitch),
        index=np.array([v.index for v in voxels], dtype=np.int64).reshape(-1, 3),
        center=np.array([v.center for v in voxels], dtype=np.float64).reshape(-1, 3),
        resolution=np.array([v.resolution for v in voxels], dtype=np.float64),
        occupancy=np.array([v.occupancy for v in voxels], dtype=np.int8),
        semantic_label=np.array([v.semantic_label for v in voxels], dtype=object),
        spot_id=np.array([v.spot_id for v in voxels], dtype=object),
        synthetic=np.array([v.synthetic for v in voxels], dtype=bool),
        confidence=np.array([v.confidence for v in voxels], dtype=np.float64),
    )


def load_grid_arrays(path: str | Path):
    """Column view of a saved grid: the arrays, with no Voxel objects.

    Everything downstream of the abstraction stage wants columns -- all
    centres, all resolutions -- not half a million dataclasses. Building
    those costs seconds and a gigabyte, and every consumer immediately
    stacks them back into arrays.
    """
    from bodynav.core.types import GridArrays

    with np.load(Path(path), allow_pickle=True) as data:
        return GridArrays(
            origin=np.asarray(data["origin"], dtype=float),
            base_pitch=float(data["base_pitch"]),
            index=np.asarray(data["index"], dtype=int).reshape(-1, 3),
            center=np.asarray(data["center"], dtype=float).reshape(-1, 3),
            resolution=np.asarray(data["resolution"], dtype=float),
            occupancy=np.asarray(data["occupancy"]),
            semantic_label=np.asarray(data["semantic_label"], dtype=object),
            spot_id=np.asarray(data["spot_id"], dtype=object),
            synthetic=np.asarray(data["synthetic"], dtype=bool),
            confidence=np.asarray(data["confidence"], dtype=float),
        )


def load_grid(path: str | Path):
    """Inverse of :func:`save_grid`.

    Note the arrays are pulled out of the archive *once*. Indexing an npz
    inside a loop re-reads and re-decompresses the whole array on every
    access: with eight fields and half a million voxels that is four
    million decompressions, which ran to 15 GB of churn before it was
    caught. Prefer :func:`load_grid_arrays` unless you need the objects.
    """
    from bodynav.core.types import Voxel, VoxelGrid

    a = load_grid_arrays(path)
    grid = VoxelGrid(origin=a.origin, base_pitch=a.base_pitch)
    for i in range(len(a)):
        spot = a.spot_id[i]
        grid.add(
            Voxel(
                index=(int(a.index[i, 0]), int(a.index[i, 1]), int(a.index[i, 2])),
                center=a.center[i],
                resolution=float(a.resolution[i]),
                occupancy=int(a.occupancy[i]),
                semantic_label=str(a.semantic_label[i]),
                spot_id=None if spot is None else str(spot),
                synthetic=bool(a.synthetic[i]),
                confidence=float(a.confidence[i]),
            )
        )
    return grid
