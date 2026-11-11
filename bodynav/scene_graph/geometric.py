"""Geometric scene understanding for as-built scans (Sec. III-A, regime [A]).

The manuscript's stage 1 prompts SAM in 3D across posed RGB-D frames
(:mod:`bodynav.scene_graph.segmentation`). **The as-built lab-floor scan has
no posed frames** -- it is a registered LiDAR cloud with colour and
nothing else -- so multi-view prompting is not merely expensive here, it
is undefined. This module is the path that regime actually uses:
structure by geometry, instances by density clustering, and labels from a
**closed vocabulary** of rules over cluster geometry and colour.

The paper must say which path produced which numbers. The honest framing
is that BodyNav's hierarchy and graph are shared, while the segmentation
front-end is swapped for the evidence the scan can support.

Vocabulary
----------
Structure ``floor``, ``ceiling``, ``wall`` come from planar/vertical
extraction. Instances take one of ``pipe``, ``duct``, ``column``,
``cone``, ``equipment``, ``furniture``, ``stored_material``, ``clutter``.
Nothing outside this list is ever emitted, so a downstream category error
is a rule bug, not a hallucinated class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The closed label vocabulary. Anything else is a bug.
VOCABULARY: tuple[str, ...] = (
    "floor", "ceiling", "wall",
    "pipe", "duct", "column", "cone",
    "equipment", "furniture", "stored_material", "clutter",
)


@dataclass
class StructureSplit:
    """Per-point structural classification of a single storey."""

    floor: np.ndarray
    ceiling: np.ndarray
    wall: np.ndarray
    above_ceiling: np.ndarray
    objects: np.ndarray
    meta: dict = field(default_factory=dict)

    def labels(self) -> np.ndarray:
        """(N,) label array with ``unknown`` where nothing matched."""
        out = np.full(len(self.floor), "unknown", dtype=object)
        out[self.above_ceiling] = "duct"     # services above the ceiling
        out[self.wall] = "wall"
        out[self.ceiling] = "ceiling"
        out[self.floor] = "floor"
        return out


def split_structure(
    xyz: np.ndarray,
    floor: tuple[float, float],
    *,
    slab_tol: float = 0.08,
    ceiling_tol: float = 0.10,
    cell: float = 0.20,
    wall_span_frac: float = 0.5,
    wall_min_points: int = 12,
) -> StructureSplit:
    """Split a storey into floor / ceiling / wall / above-ceiling / objects.

    Walls are found by vertical extent rather than by plane fitting: in a
    bird's-eye cell, a wall's returns span most of the storey, while a
    desk or a pallet occupies a slice near the floor. This costs one
    histogram instead of a RANSAC sweep over four million points, and --
    unlike plane fitting -- it does not break on the curved and cluttered
    walls that as-built interiors actually have.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    floor_z, ceiling_z = floor
    storey = ceiling_z - floor_z
    z = xyz[:, 2]

    is_floor = np.abs(z - floor_z) <= slab_tol
    is_ceiling = np.abs(z - ceiling_z) <= ceiling_tol
    is_above = z > ceiling_z + ceiling_tol
    rest = ~(is_floor | is_ceiling | is_above) & (z > floor_z)

    lo = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - lo) / cell).astype(int)
    dims = ij.max(axis=0) + 1
    keys = ij[:, 0] * dims[1] + ij[:, 1]

    n_cells = int(dims[0] * dims[1])
    r_keys = keys[rest]
    r_z = z[rest]
    counts = np.bincount(r_keys, minlength=n_cells)
    zmax = np.full(n_cells, -np.inf)
    zmin = np.full(n_cells, np.inf)
    np.maximum.at(zmax, r_keys, r_z)
    np.minimum.at(zmin, r_keys, r_z)
    span = np.where(counts > 0, zmax - zmin, 0.0)

    wall_cell = (span >= wall_span_frac * storey) & (counts >= wall_min_points)
    is_wall = rest & wall_cell[keys]
    is_object = rest & ~is_wall

    meta = {
        "floor_z": round(float(floor_z), 4),
        "ceiling_z": round(float(ceiling_z), 4),
        "storey_m": round(float(storey), 4),
        "cell": cell,
        "wall_cells": int(wall_cell.sum()),
        "fractions": {
            k: round(float(v.mean()), 4)
            for k, v in (("floor", is_floor), ("ceiling", is_ceiling),
                         ("wall", is_wall), ("above_ceiling", is_above),
                         ("objects", is_object))
        },
    }
    return StructureSplit(is_floor, is_ceiling, is_wall, is_above, is_object, meta)


def cluster_objects(
    xyz: np.ndarray,
    *,
    pitch: float = 0.05,
    eps: float = 0.18,
    min_samples: int = 6,
    min_points: int = 40,
) -> np.ndarray:
    """Euclidean instance clustering; returns (N,) cluster ids, -1 = noise.

    DBSCAN over four million points is not tractable, and it is not
    necessary: the cloud is clustered at 5 cm downsampled resolution and
    labels are pushed back to full resolution by nearest neighbour. The
    push-back is capped at ``pitch`` so a full-resolution point never
    inherits a label from a cluster it does not touch.
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    from bodynav.io.pointcloud import voxel_downsample

    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros(0, dtype=int)

    seeds, _ = voxel_downsample(xyz, pitch)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(seeds)

    tree = cKDTree(seeds)
    dist, idx = tree.query(xyz, k=1, distance_upper_bound=pitch * 1.5)
    out = np.where(np.isfinite(dist), labels[np.minimum(idx, len(seeds) - 1)], -1)

    # drop clusters too small to be an instance, renumber densely
    keep = {}
    for cid, count in zip(*np.unique(out[out >= 0], return_counts=True)):
        if count >= min_points:
            keep[int(cid)] = len(keep)
    return np.array([keep.get(int(c), -1) for c in out], dtype=int)


def cluster_features(pts: np.ndarray, colors: np.ndarray | None, floor_z: float) -> dict:
    """Geometric and colour descriptors used by :func:`label_cluster`."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    dims = hi - lo
    footprint = sorted(dims[:2])
    feats = {
        "n": int(len(pts)),
        "base_h": float(lo[2] - floor_z),
        "top_h": float(hi[2] - floor_z),
        "height": float(dims[2]),
        "width": float(footprint[1]),
        "depth": float(footprint[0]),
        "elongation": float(footprint[1] / max(footprint[0], 1e-3)),
        "verticality": float(dims[2] / max(footprint[1], 1e-3)),
        "volume": float(np.prod(np.maximum(dims, 1e-3))),
    }
    if colors is not None and len(colors):
        rgb = colors.mean(axis=0)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        feats["rgb"] = [round(float(v), 3) for v in rgb]
        feats["orangeness"] = float(rgb[0] - 0.5 * (rgb[1] + rgb[2]))
    return feats


def label_cluster(f: dict, *, storey: float) -> str:
    """Closed-vocabulary label for one instance.

    Ordered most specific first. The thresholds are site parameters, not
    tuned constants: they encode what the categories *are* (a cone is
    knee-high, orange and on the floor; a pipe is long, thin and
    overhead), which is what makes them transferable to another building.
    """
    overhead = f["base_h"] > 0.6 * storey

    if overhead and f["elongation"] >= 3.0 and f["width"] <= 0.5:
        return "pipe"
    if overhead:
        return "duct"
    if f["height"] >= 0.8 * storey and f["verticality"] >= 2.0 and f["width"] <= 1.2:
        return "column"
    if (f["base_h"] < 0.25 and f["height"] <= 0.9 and f["width"] <= 0.7
            and f.get("orangeness", 0.0) > 0.08):
        return "cone"
    if f["base_h"] < 0.3 and f["height"] <= 1.2 and f["width"] <= 2.5:
        return "furniture"
    if f["height"] > 1.2 and f["width"] <= 1.5:
        return "equipment"
    if f["volume"] >= 1.0:
        return "stored_material"
    return "clutter"


def clearance_map(
    xyz: np.ndarray,
    floor: tuple[float, float],
    *,
    cell: float = 0.25,
    ignore_below: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Per-cell overhead clearance: height of the lowest overhead return.

    Cells with nothing overhead report the ceiling height. This is the map
    that separates the embodiments: the same corridor is open to a 1.25 m
    Husky and closed to a 2.0 m humanoid wherever services drop between
    the two.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    floor_z, ceiling_z = floor
    lo = xyz[:, :2].min(axis=0)
    hi = xyz[:, :2].max(axis=0)
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)

    over = xyz[xyz[:, 2] > floor_z + ignore_below]
    ij = np.clip(np.floor((over[:, :2] - lo) / cell).astype(int), 0, dims - 1)
    keys = ij[:, 0] * dims[1] + ij[:, 1]

    clear = np.full(int(dims[0] * dims[1]), np.inf)
    np.minimum.at(clear, keys, over[:, 2] - floor_z)
    clear = clear.reshape(tuple(dims))
    clear[~np.isfinite(clear)] = ceiling_z - floor_z
    meta = {"cell": cell, "origin": lo.tolist(), "dims": dims.tolist(),
            "ignore_below": ignore_below}
    return clear, meta


def body_occupancy(
    xyz: np.ndarray,
    floor: tuple[float, float],
    meta: dict,
    *,
    band: tuple[float, float] = (0.15, 1.25),
) -> np.ndarray:
    """Per-cell return count inside a body-height band, on ``meta``'s grid."""
    xyz = np.asarray(xyz, dtype=np.float64)
    floor_z = floor[0]
    cell, dims = meta["cell"], np.asarray(meta["dims"], dtype=int)
    origin = np.asarray(meta["origin"], dtype=float)

    z_rel = xyz[:, 2] - floor_z
    sel = (z_rel > band[0]) & (z_rel < band[1])
    ij = np.clip(np.floor((xyz[sel, :2] - origin) / cell).astype(int), 0, dims - 1)
    occ = np.zeros(tuple(dims), dtype=np.int32)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)
    return occ


def low_clearance_regions(
    clear: np.ndarray,
    meta: dict,
    *,
    tall_needs: float,
    short_needs: float,
    min_area_m2: float = 0.5,
    free_below: np.ndarray | None = None,
) -> list[dict]:
    """Connected regions passable by the short robot but not the tall one.

    This is the multi-embodiment contrast made measurable: every region
    returned is a place where one map of the building says 'go' and the
    other says 'stop', from one scan and one pipeline.

    ``free_below`` -- cells the short robot can actually stand in -- is
    what separates a low soffit from the top of a filing cabinet. Without
    it, every tall object in the building reports as a low passage.
    """
    from scipy import ndimage

    cell = meta["cell"]
    origin = np.asarray(meta["origin"], dtype=float)
    blocks_tall = (clear < tall_needs) & (clear >= short_needs)
    if free_below is not None:
        blocks_tall &= free_below
    labels, n = ndimage.label(blocks_tall)

    out = []
    for lab in range(1, n + 1):
        mask = labels == lab
        area = float(mask.sum() * cell * cell)
        if area < min_area_m2:
            continue
        ii, jj = np.nonzero(mask)
        out.append({
            "area_m2": round(area, 2),
            "min_clearance_m": round(float(clear[mask].min()), 3),
            "mean_clearance_m": round(float(clear[mask].mean()), 3),
            "centroid_xy": [round(float(origin[0] + (ii.mean() + 0.5) * cell), 2),
                            round(float(origin[1] + (jj.mean() + 0.5) * cell), 2)],
            "bbox_xy": [
                [round(float(origin[0] + ii.min() * cell), 2),
                 round(float(origin[1] + jj.min() * cell), 2)],
                [round(float(origin[0] + (ii.max() + 1) * cell), 2),
                 round(float(origin[1] + (jj.max() + 1) * cell), 2)],
            ],
        })
    out.sort(key=lambda r: -r["area_m2"])
    return out


def find_portals(
    xyz: np.ndarray,
    floor: tuple[float, float],
    free: np.ndarray,
    seen: np.ndarray,
    occlusion: np.ndarray,
    origin: np.ndarray,
    cell: float,
    *,
    min_area_m2: float = 20.0,
    max_width_m: float = 1.5,
    max_thickness_m: float = 0.12,
    min_height_m: float = 1.5,
    max_portals: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    """Open doorways that were shut when the building was scanned.

    An as-built scan records the building on one afternoon, including
    which doors happened to be closed. A closed fire door is a 5-10 cm
    panel that severs a corridor loop into halves, and a map built from
    that scan says two halves of one floor are unreachable from each
    other -- which is false about the building, and would silently make
    half the evaluation's tasks infeasible.

    A portal is identified by **thickness and height**, which is what
    separates a door leaf from everything else: this floor's walls are
    10-25 cm of masonry and its door leaves 6 cm, while the furniture and
    rails that block a side room reach only part of the way up. A barrier
    is opened only if it is thinner than ``max_thickness_m``, taller than
    ``min_height_m``, narrower than ``max_width_m``, and is the minimal
    crossing between two free regions that are each substantial. So the
    search can never tunnel through a wall, and never drives through a
    desk.

    Returns the updated free mask and one record per portal, so the paper
    can report exactly how many were opened and where.
    """
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    free = free.copy()
    dims = np.asarray(free.shape)
    floor_z = floor[0]
    z_rel = xyz[:, 2] - floor_z
    body = (z_rel > 0.15) & (z_rel < 2.0)
    body_pts = xyz[body]
    portals: list[dict] = []

    def cell_of(pts):
        return np.clip(np.floor((pts[:, :2] - origin) / cell).astype(int), 0, dims - 1)

    body_ij = cell_of(body_pts)
    body_key = body_ij[:, 0] * dims[1] + body_ij[:, 1]

    # Per-cell barrier thickness: the minor horizontal extent of the
    # body-band points inside the cell. A door leaf is a few centimetres
    # across its minor axis; a masonry wall is a quarter of a metre. The
    # search needs this in the *cost*, not only in the accept test --
    # at 25 cm resolution a doorway cell and a wall cell are otherwise
    # indistinguishable, so the cheapest crossing is chosen arbitrarily
    # and lands in the wall as often as in the door.
    n_cells = int(dims[0] * dims[1])
    counts = np.bincount(body_key, minlength=n_cells).astype(float)
    nz = counts > 0
    thickness_map = np.full(n_cells, np.inf)
    stds = []
    for axis in (0, 1):
        v = body_pts[:, axis]
        mean = np.bincount(body_key, weights=v, minlength=n_cells)
        sq = np.bincount(body_key, weights=v * v, minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = sq / np.maximum(counts, 1) - (mean / np.maximum(counts, 1)) ** 2
        stds.append(np.sqrt(np.maximum(var, 0.0)))
    # a uniform slab of width w has std w/sqrt(12)
    thickness_map[nz] = np.sqrt(12.0) * np.minimum(stds[0], stds[1])[nz]
    thin = np.clip(thickness_map, 0.0, 1.0).reshape(tuple(dims))

    # cost of crossing a cell: free is nearly free, blocked-but-scanned is
    # expensive in proportion to how solid it looks, unscanned is barred.
    n = int(dims[0] * dims[1])
    for _ in range(max_portals):
        lab, _ = ndimage.label(free)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = np.flatnonzero(sizes * cell * cell >= min_area_m2)
        if len(keep) < 2:
            break
        order = keep[np.argsort(sizes[keep])[::-1]]
        a, b = order[0], order[1]

        cost = np.where(seen, 1.0 + 100.0 * occlusion + 400.0 * thin, 1e6)
        cost[free] = 1e-3
        flat = cost.ravel()

        rows, cols, vals = [], [], []
        idx = np.arange(n).reshape(dims)
        for di, dj in ((1, 0), (0, 1)):
            u = idx[: dims[0] - di, : dims[1] - dj].ravel()
            v = idx[di:, dj:].ravel()
            w = 0.5 * (flat[u] + flat[v])
            rows += [u, v]
            cols += [v, u]
            vals += [w, w]
        graph = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()

        src = np.flatnonzero((lab == a).ravel())
        # min_only returns (distances, predecessors, sources)
        dist, pred, _ = dijkstra(
            graph, indices=src, return_predecessors=True, min_only=True
        )
        tgt = np.flatnonzero((lab == b).ravel())
        end = int(tgt[np.argmin(dist[tgt])])
        if not np.isfinite(dist[end]):
            break

        path, node = [], end
        while node >= 0:
            path.append(node)
            node = pred[node]
        barrier = [p for p in path if not free.ravel()[p]]
        if not barrier or len(barrier) * cell > max_width_m:
            break

        # measure the blocking element: thickness across the crossing
        sel = np.isin(body_key, barrier)
        pts = body_pts[sel]
        if len(pts) < 20:
            break
        ii, jj = np.unravel_index(barrier, tuple(dims))
        span = np.array([ii.max() - ii.min(), jj.max() - jj.min()]) * cell
        # Thickness is measured across the whole crossing, not per cell: a
        # 24 cm wall straddling a cell boundary presents 12 cm in each of
        # two cells and would pass a per-cell test. Percentiles, so a
        # stray return does not inflate it.
        extent = np.percentile(pts[:, :2], 97.5, axis=0) - np.percentile(pts[:, :2], 2.5, axis=0)
        thickness = float(extent.min())
        width = float(max(span) + cell)
        height = float(pts[:, 2].max() - pts[:, 2].min())

        if thickness > max_thickness_m or height < min_height_m:
            # masonry, or something short like a rail, a desk or a stack of
            # material. Either way it is not a door, so the building stays
            # as it was scanned.
            break

        # Open the whole leaf, not just the cells the route happened to
        # cross. A door fills its aperture, so freeing one cell of it
        # leaves a 25 cm gap in a 90 cm doorway -- passable to the search,
        # impassable to the robot, and the map then lies in the most
        # expensive way available.
        thin_blocked = (thin <= max_thickness_m) & ~free & seen
        comp, _ = ndimage.label(thin_blocked, structure=np.ones((3, 3)))
        leaf_ids = {int(comp.ravel()[b]) for b in barrier} - {0}
        leaf = np.isin(comp, list(leaf_ids)) if leaf_ids else np.zeros_like(free)
        # bound it: a portal is a doorway, not a whole partition wall
        li, lj = np.nonzero(leaf)
        if len(li):
            centre = np.array([ii.mean(), jj.mean()])
            within = np.hypot(li - centre[0], lj - centre[1]) * cell <= max_width_m
            leaf = np.zeros_like(free)
            leaf[li[within], lj[within]] = True
        opened = leaf | np.isin(np.arange(free.size).reshape(free.shape),
                                np.array(barrier))
        free |= opened
        aperture = float(opened.sum()) * cell * cell
        oi, oj = np.nonzero(opened)
        portals.append({
            "centre_xy": [round(float(origin[0] + (oi.mean() + 0.5) * cell), 2),
                          round(float(origin[1] + (oj.mean() + 0.5) * cell), 2)],
            "width_m": round(float(max(np.ptp(oi), np.ptp(oj)) + 1) * cell, 2),
            "crossing_width_m": round(width, 2),
            "aperture_m2": round(aperture, 2),
            "thickness_m": round(thickness, 3),
            "height_m": round(height, 2),
            "cells": len(barrier),
            "joins_m2": [round(float(sizes[a] * cell * cell), 1),
                         round(float(sizes[b] * cell * cell), 1)],
        })
    return free, portals
