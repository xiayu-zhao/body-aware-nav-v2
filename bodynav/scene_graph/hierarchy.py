"""Hierarchical organization: floors -> areas -> spots (Sec. III-A.2).

Floors: local maxima of the 1 cm height histogram H(z) that reach >= 90%
of the global maximum are DBSCAN-clustered (eps = 0.2 m) into
floor--ceiling pairs (handles irregular ceilings).

Areas: per-floor bird's-eye density projection -> adaptive thresholding,
morphological boundary extraction, watershed segmentation. The taxonomy
follows site semantics (work zones, staging, corridors, restricted zones)
rather than generic room categories.

Spots: consolidated instances from segmentation, each retaining centroid,
axis-aligned bounding volume, and category.

Output: labeled points P_s = {p_k, l_k, a_k, o_k, s_k} (coordinates,
floor, area, instance, semantic label).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LabeledPoints:
    """P_s: points with per-point floor / area / instance / semantic ids."""

    xyz: np.ndarray  # (N, 3)
    floor_id: np.ndarray  # (N,) int
    area_id: np.ndarray  # (N,) int
    instance_id: np.ndarray  # (N,) int
    semantic: np.ndarray  # (N,) object (str labels)
    colors: np.ndarray | None = None  # (N, 3) optional
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xyz)

    def select(self, sel: np.ndarray) -> "LabeledPoints":
        """Subset every per-point array by a boolean mask or index array.

        Entries of ``extras`` that are per-point arrays are subset too;
        anything else (scalars, metadata) is carried through unchanged, so
        stages can stash both kinds without special-casing.
        """
        sel = np.asarray(sel)
        n = len(self.xyz)
        extras = {
            k: (v[sel] if isinstance(v, np.ndarray) and len(v) == n else v)
            for k, v in self.extras.items()
        }
        return LabeledPoints(
            xyz=self.xyz[sel],
            floor_id=self.floor_id[sel],
            area_id=self.area_id[sel],
            instance_id=self.instance_id[sel],
            semantic=self.semantic[sel],
            colors=None if self.colors is None else self.colors[sel],
            extras=extras,
        )

    @classmethod
    def from_xyz(
        cls, xyz: np.ndarray, *, colors: np.ndarray | None = None,
        semantic: np.ndarray | str = "unknown",
    ) -> "LabeledPoints":
        """Unlabeled cloud wrapper, for running stage 2 on a raw scan
        before the scene graph exists (the bootstrap path used to bring up
        the real-world dataset)."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = len(xyz)
        if isinstance(semantic, str):
            semantic = np.full(n, semantic, dtype=object)
        return cls(
            xyz=xyz,
            floor_id=np.zeros(n, dtype=int),
            area_id=np.zeros(n, dtype=int),
            instance_id=np.full(n, -1, dtype=int),
            semantic=np.asarray(semantic, dtype=object),
            colors=colors,
        )


def detect_floors(
    xyz: np.ndarray,
    *,
    bin_size: float = 0.01,
    peak_ratio: float = 0.9,
    dbscan_eps: float = 0.2,
    min_storey: float = 1.5,
) -> list[tuple[float, float]]:
    """Return (floor_z, ceiling_z) pairs from the height histogram.

    Horizontal surfaces are histogram peaks: a slab and a ceiling each
    concentrate a large share of returns in a couple of centimetres, while
    walls spread theirs uniformly over the storey. Peaks within
    ``dbscan_eps`` are one surface (a ceiling is rarely flat to 1 cm), and
    a floor/ceiling pair must be at least ``min_storey`` apart so a
    mezzanine lip or a row of desks is not mistaken for a storey.

    ``peak_ratio`` is relative to the *largest* peak, so a scan whose
    ceiling returns are sparser than its floor returns (which is the usual
    case -- the scanner looks down more than up) still yields the pair;
    the ceiling is then taken as the tallest surface above the floor
    rather than as a peak in its own right.
    """
    z = np.asarray(xyz, dtype=np.float64)[:, 2]
    if len(z) == 0:
        return []

    edges = np.arange(z.min(), z.max() + bin_size, bin_size)
    hist, _ = np.histogram(z, bins=edges)
    centers = edges[:-1] + bin_size / 2

    strong = np.flatnonzero(hist >= peak_ratio * hist.max())
    if not len(strong):
        return []

    # cluster adjacent strong bins into surfaces (1-D DBSCAN by gap)
    surfaces: list[np.ndarray] = []
    run = [strong[0]]
    for i in strong[1:]:
        if (centers[i] - centers[run[-1]]) <= dbscan_eps:
            run.append(i)
        else:
            surfaces.append(np.array(run))
            run = [i]
    surfaces.append(np.array(run))

    # each surface's height is its density-weighted centre
    heights = [
        float(np.average(centers[s], weights=hist[s])) for s in surfaces
    ]
    heights.sort()

    pairs: list[tuple[float, float]] = []
    for floor_z in heights:
        if pairs and floor_z - pairs[-1][0] < min_storey:
            continue
        above = z[z > floor_z + min_storey]
        if not len(above):
            continue
        # ceiling: densest surface above, at 5 cm resolution
        c_edges = np.arange(above.min(), above.max() + 0.05, 0.05)
        c_hist, _ = np.histogram(above, bins=c_edges)
        if not c_hist.size or c_hist.max() == 0:
            continue
        ceiling_z = float(c_edges[int(np.argmax(c_hist))] + 0.025)
        pairs.append((floor_z, ceiling_z))
    return pairs


def segment_areas(
    xyz: np.ndarray,
    floor: tuple[float, float],
    *,
    cell: float = 0.25,
    min_area_m2: float = 4.0,
    seed_spacing: float = 3.5,
    link_radius: float = 1.0,
    occlusion_tau: float = 0.3,
    speckle_m2: float = 0.5,
    bridge_portals: bool = True,
    portal_min_area_m2: float = 20.0,
    return_free: bool = False,
) -> tuple[np.ndarray, dict]:
    """Watershed functional zones for one floor; returns per-point area ids.

    Free space is the complement of the vertical structure in a bird's-eye
    occupancy map. Watershed on its distance transform splits that free
    space where it pinches -- doorways, corridor mouths -- which is what
    makes the result *functional* zones (rooms, corridor segments) rather
    than arbitrary tiles.

    ``seed_spacing`` is the one parameter that decides what "a zone" means,
    and it is a claim about the building, not a tuning knob: seeds closer
    together than the width of the spaces being segmented will chop a
    single corridor into pieces. On the lab floor, 1.5 m yields 38 zones
    with corridors fragmented; 3.5 m -- roughly a corridor width -- yields
    19, one per corridor run and per room.
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    xyz = np.asarray(xyz, dtype=np.float64)
    floor_z, ceiling_z = floor
    lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)

    # Occupancy is an *occlusion* test, not a return count. A solid object
    # standing on the floor hides the floor beneath it, so it shows many
    # body-band returns and few floor returns; open floor shows the
    # reverse. A raw count threshold cannot tell the two apart -- three
    # stray returns from a reflective surface marked a cell blocked, which
    # shrank 2 m corridors to 0.7 m ribbons and cut them wherever noise
    # happened to land.
    band = (xyz[:, 2] > floor_z + 0.15) & (xyz[:, 2] < min(ceiling_z - 0.1, floor_z + 2.0))
    ij = np.clip(np.floor((xyz[band, :2] - lo) / cell).astype(int), 0, dims - 1)
    occ = np.zeros(tuple(dims), dtype=np.int32)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)

    on_floor = np.abs(xyz[:, 2] - floor_z) < 0.10
    ij_f = np.clip(np.floor((xyz[on_floor, :2] - lo) / cell).astype(int), 0, dims - 1)
    floor_returns = np.zeros(tuple(dims), dtype=np.int32)
    np.add.at(floor_returns, (ij_f[:, 0], ij_f[:, 1]), 1)
    occlusion = occ / np.maximum(occ + floor_returns, 1)

    # scanned extent: cells with any return at all (the building's inside)
    ij_all = np.clip(np.floor((xyz[:, :2] - lo) / cell).astype(int), 0, dims - 1)
    seen = np.zeros(tuple(dims), dtype=bool)
    seen[ij_all[:, 0], ij_all[:, 1]] = True
    seen = ndimage.binary_closing(seen, np.ones((5, 5)))

    free = seen & (occlusion <= occlusion_tau)

    # Despeckle by dropping small components rather than by morphological
    # opening. Opening erodes every boundary by a cell, which severs a
    # corridor wherever the free ribbon narrows to one -- it cost 16% of
    # the free space here and broke the loop in two.
    spec_lab, _ = ndimage.label(free)
    spec_sizes = np.bincount(spec_lab.ravel())
    spec_sizes[0] = 0
    tiny = np.flatnonzero(spec_sizes * cell * cell < speckle_m2)
    free[np.isin(spec_lab, tiny)] = False

    # Doors shut on the afternoon of the scan cut the floor into pieces
    # that are connected in the building. Opening them is a decision about
    # the building, so it is recorded rather than buried.
    portals: list[dict] = []
    if bridge_portals:
        from bodynav.scene_graph.geometric import find_portals

        free, portals = find_portals(
            xyz, floor, free, seen, occlusion, lo, cell,
            min_area_m2=portal_min_area_m2,
        )

    dist = ndimage.distance_transform_edt(free)
    coords = peak_local_max(
        dist, min_distance=max(int(round(seed_spacing / cell)), 1),
        labels=free, exclude_border=False,
    )
    seeds = np.zeros(tuple(dims), dtype=int)
    for k, (i, j) in enumerate(coords, start=1):
        seeds[i, j] = k
    labels = watershed(-dist, seeds, mask=free)

    # drop specks, then renumber densely from 0
    keep, next_id = {}, 0
    for lab, count in zip(*np.unique(labels[labels > 0], return_counts=True)):
        if count * cell * cell >= min_area_m2:
            keep[int(lab)] = next_id
            next_id += 1
    remap = np.full(labels.max() + 1, -1, dtype=int)
    for lab, new in keep.items():
        remap[lab] = new
    grid = remap[labels]

    per_point = grid[ij_all[:, 0], ij_all[:, 1]]

    # Adjacency between areas is *free-space contact*, not bounding-box
    # proximity. These corridors run diagonally, so their axis-aligned
    # boxes overlap across the courtyard they enclose; an AABB test then
    # declares opposite sides of the building adjacent and any route
    # planned on that graph walks through the middle of it.
    #
    # Contact is tested within ``link_radius`` rather than between touching
    # cells: zones are separated by the door frames and corridor pinches
    # the watershed cut them at, so a strict 4-neighbour test leaves the
    # graph disconnected -- 14 edges over 19 zones on the lab floor, with
    # no route from one side of the loop to the other. The radius stays
    # local, so it bridges a doorway and never a courtyard.
    reach = max(int(round(link_radius / cell)), 1)
    touching: set[tuple[int, int]] = set()
    for da in range(reach + 1):
        for db in range(-reach, reach + 1):
            if (da, db) == (0, 0) or da * da + db * db > reach * reach:
                continue
            a = grid[: dims[0] - da, max(0, -db): dims[1] - max(0, db)]
            b = grid[da:, max(0, db): dims[1] - max(0, -db)]
            pairs = np.unique(np.stack([a.ravel(), b.ravel()], axis=1), axis=0)
            for u, v in pairs:
                if u >= 0 and v >= 0 and u != v:
                    touching.add((int(min(u, v)), int(max(u, v))))

    meta = {
        "cell": cell,
        "seed_spacing": seed_spacing,
        "origin": lo.tolist(),
        "dims": dims.tolist(),
        "n_areas": next_id,
        "link_radius": link_radius,
        "occlusion_tau": occlusion_tau,
        "portals": portals,
        "free_cells": int(free.sum()),
        "free_area_m2": round(float(free.sum() * cell * cell), 1),
        "adjacency": sorted(touching),
    }
    if return_free:
        return per_point, meta, free
    return per_point, meta


def organize(
    xyz: np.ndarray,
    instances: dict[int, np.ndarray],
    semantics: dict[int, str],
    *,
    floor_id: np.ndarray | None = None,
    area_id: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> LabeledPoints:
    """Assemble the full hierarchy from segmentation output (cross-view
    majority voting for instance membership happens upstream)."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    inst = np.full(n, -1, dtype=int)
    sem = np.full(n, "unknown", dtype=object)
    for iid, idx in instances.items():
        inst[idx] = iid
        sem[idx] = semantics.get(iid, "unknown")
    return LabeledPoints(
        xyz=xyz,
        floor_id=np.zeros(n, dtype=int) if floor_id is None else np.asarray(floor_id),
        area_id=np.full(n, -1, dtype=int) if area_id is None else np.asarray(area_id),
        instance_id=inst,
        semantic=sem,
        colors=colors,
    )
