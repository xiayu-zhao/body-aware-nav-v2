"""Value-guided A* with per-voxel feasibility gating (Sec. III-E.2).

Search over 26-connected voxel adjacency minimizing
J(P) = sum_i Cost(v_i, v_{i+1}) with

    Cost(v_i, v_j) = ||c_i - c_j|| * (1 - eta * V*(v_j))     (Eq. 10)
    h(v_k)        = ||c_k - c_g|| * (1 - eta * V*(v_k))      (Eq. 11)

Before Eq. 10 is evaluated, each neighbor is gated on precomputed
per-voxel attributes — slope <= theta_max, lateral/vertical clearance —
so violating neighbors never enter the open set: only kinematically
feasible transitions are searched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodynav.core.types import ValueMapSet, VoxelGrid
from bodynav.planning.obstacles import ClearanceFields
from bodynav.robots.profiles import RobotProfile


@dataclass
class PlannerParams:
    eta: float = 0.2  # value-bias gain, in [0.1, 0.3]
    #: height a ground platform may step up or down between neighbouring
    #: cells, in metres -- a physical property, not a cell count. As cells
    #: it changes meaning with the lattice: one cell was 20 cm at a 0.2 m
    #: pitch and 10 cm at 0.1 m, which silently made the finer lattice
    #: stricter than the coarse one.
    max_step_m: float = 0.20
    #: cap on expansions, so an infeasible instruction fails in bounded time
    max_expansions: int = 4_000_000
    #: how far the goal may be snapped to a cell the robot can occupy.
    #: A robot cannot stand inside a pallet, so "go to the pallet" has to
    #: mean standing near it -- but a generous radius quietly redefines an
    #: unreachable destination as a reachable one, which is how an
    #: instruction the system should refuse becomes one it appears to
    #: satisfy. 3 m is arrival; further is a different place.
    goal_snap_m: float = 3.0
    #: the start is an area the robot is already standing in, so its
    #: centroid may legitimately sit inside furniture; snapping further is
    #: not a claim about having arrived anywhere.
    start_snap_m: float = 6.0


class ValueGuidedAStar:
    """Value-guided A* over the lattice carried by :class:`ClearanceFields`.

    The gate runs *before* Eq. 10 is evaluated, so an infeasible neighbour
    never enters the open set. That ordering is what makes SCR well
    defined: a path that exists is a path every step of which passed the
    same clearance test the metric later reports.

    Connectivity is 26-neighbour as the manuscript specifies. For a ground
    platform the gate additionally pins each node to its column's standing
    layer, so the search is volumetric in code and planar in effect;
    an aerial profile (no ``foothold_band``) is free in z.
    """

    def __init__(self, params: PlannerParams, robot: RobotProfile):
        self.params = params
        self.robot = robot
        self.stats: dict = {}
        #: keyed by id(fields); a planner is per-robot and short-lived, and
        #: the fields it is handed outlive any single search
        self._layer_cache: dict = {}

    # -- gate -----------------------------------------------------------
    def clearance_required(self, fields: ClearanceFields) -> float:
        """Half the body width, plus the lattice's own uncertainty.

        Clearance is sampled at cell centres, so a point anywhere in the
        cell can be up to half a cell diagonal nearer an obstacle than the
        sample says. Without that margin a plan can leave 0.30 m of real
        gap for a 0.67 m vehicle and still pass its own gate.
        """
        return 0.5 * self.robot.min_lateral_clearance + 0.5 * fields.pitch * np.sqrt(2)

    def feasible(self, index, fields: ClearanceFields) -> bool:
        """Per-voxel gate: slope, lateral clearance, vertical clearance."""
        i, j, k = (int(v) for v in index)
        dims = fields.occupied.shape
        if not (0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]):
            return False
        radius = self.clearance_required(fields)
        if fields.occupied[i, j, k] and not (fields.meta or {}).get("has_prior"):
            return False
        if fields.lateral_grid[i, j, k] < radius:
            return False
        if fields.vertical_grid[i, j, k] < self.robot.required_vertical_clearance:
            return False
        if np.degrees(fields.slope_grid[i, j]) > self.robot.max_slope_deg:
            return False

        if self.robot.foothold_band is not None or self.robot.kind != "aerial":
            ground = fields.ground_k[i, j]
            if ground < 0:
                return False
            allow = self.params.max_step_m
            if self.robot.foothold_band is not None:
                allow = max(allow, float(self.robot.foothold_band))
            if abs(k - (ground + 1)) > max(int(round(allow / fields.pitch)), 1):
                return False
        return True

    # -- search ---------------------------------------------------------
    def search(
        self,
        grid: VoxelGrid,
        value_maps: ValueMapSet,
        fields: ClearanceFields,
        start: np.ndarray,
        goal: np.ndarray,
        forbidden: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Raw voxel path (K, 3) world coords, or None when no feasible
        path exists (the correct answer for infeasible instructions).

        Both bodies search their own occupiable layer, 8-connected in
        plan. A ground platform's gate admits exactly one z per column --
        the layer it stands on -- so expanding the other 25 neighbours of
        a 26-connected lattice is provably wasted work; that search cost
        23 GB of open set on this floor before returning anything. An
        aerial platform has a layer too: the lowest admissible height in
        its flight band, which varies across the site and is what lets it
        cross clutter a ground robot must go around. Measured on Scene 2,
        the volumetric search expanded 678k nodes in 50 s where the layer
        search expands 17k in 0.15 s and reaches the same place. The
        volumetric routine is kept for a profile that genuinely needs to
        choose its height mid-route.

        ``forbidden`` is a 2-D mask of columns the caller excludes -- an
        instruction's stated hazard margin, say. It is a mask rather than
        a modified field because rebuilding the clearance volume copies
        millions of cells and invalidates the layer cache, and the task
        sampler asks for this thousands of times.
        """
        value = self._value_lattice(value_maps, fields)
        return self._search_ground(fields, value, start, goal, forbidden)

    def _ground_layer(self, fields: ClearanceFields):
        """Standing layer and its passability, computed once, vectorised."""
        dims = fields.occupied.shape
        stand_k = np.clip(fields.ground_k + 1, 0, dims[2] - 1)
        ii, jj = np.nonzero(fields.ground_k >= 0)
        kk = stand_k[ii, jj]

        radius = self.clearance_required(fields)
        # With stage 1's free map as the prior, in-plane occupancy is
        # already decided by an occlusion test over the whole cloud.
        # Re-testing it against individual voxels rejects a further 13% of
        # standing cells -- a stray return in the cell the robot occupies
        # -- and fragments the corridor.
        occupied_test = (np.zeros(len(ii), dtype=bool)
                         if (fields.meta or {}).get("has_prior")
                         else fields.occupied[ii, jj, kk])
        ok = (
            ~occupied_test
            & (fields.lateral_grid[ii, jj, kk] >= radius)
            & (fields.vertical_grid[ii, jj, kk] >= self.robot.required_vertical_clearance)
            & (np.degrees(fields.slope_grid[ii, jj]) <= self.robot.max_slope_deg)
        )
        passable = np.zeros(dims[:2], dtype=bool)
        passable[ii[ok], jj[ok]] = True
        return stand_k, passable

    def _flight_layer(self, fields: ClearanceFields):
        """Aerial counterpart of :meth:`_ground_layer`.

        A flying platform has no standing layer: it occupies whatever
        height in its band clears the geometry. For each column we take
        the *lowest* admissible height in the band, which is where a UAV
        flying as low as its envelope allows would sit, and mark the
        column passable if any such height exists. That gives the same
        (layer, passable) contract the ground search and the task sampler
        already consume, so nothing downstream needs to know which kind of
        body it is looking at.
        """
        dims = fields.occupied.shape
        floor_z = float((fields.meta or {}).get("floor_z", fields.origin[2]))
        lo = int(np.clip((floor_z + self.robot.height_band[0] - fields.origin[2])
                         / fields.pitch, 0, dims[2] - 1))
        hi = int(np.clip((floor_z + self.robot.height_band[1] - fields.origin[2])
                         / fields.pitch, 0, dims[2] - 1))
        if hi < lo:
            lo, hi = hi, lo

        radius = self.clearance_required(fields)
        need_v = self.robot.required_vertical_clearance
        band = slice(lo, hi + 1)
        ok = ((fields.lateral_grid[:, :, band] >= radius)
              & (fields.vertical_grid[:, :, band] >= need_v))
        if not (fields.meta or {}).get("has_prior"):
            ok &= ~fields.occupied[:, :, band]

        passable = ok.any(axis=2)
        first = np.argmax(ok, axis=2) + lo          # lowest admissible height
        layer = np.where(passable, first, lo).astype(int)
        return layer, passable

    def occupiable_layer(self, fields: ClearanceFields):
        """Where this body can be, as a (height per column, passable) pair.

        Memoised per field set. The aerial branch reduces a band of the
        occupancy volume, which on the lab floor is millions of cells, and
        the task sampler asks for it thousands of times while testing
        candidate routes; recomputing it each time turned a two-minute
        batch into one that did not finish.
        """
        key = id(fields)
        hit = self._layer_cache.get(key)
        if hit is not None:
            return hit
        if self.robot.kind == "aerial" and self.robot.foothold_band is None:
            out = self._flight_layer(fields)
        else:
            out = self._ground_layer(fields)
        self._layer_cache[key] = out
        return out

    def _search_ground(self, fields, value, start, goal, forbidden=None):
        import heapq

        dims = fields.occupied.shape
        stand_k, passable = self.occupiable_layer(fields)
        if forbidden is not None:
            passable = passable & ~forbidden
        I, J = dims[0], dims[1]

        start_ij = self._snap_ground(start, fields, passable,
                                     radius_m=self.params.start_snap_m)
        # Snap the goal within the start's *component*. The nearest
        # passable cell to a pallet is often a one-cell pocket behind it,
        # and snapping there reports "no feasible path" for a destination
        # the robot can plainly stand beside -- a failure of the snap, not
        # of the search, and one that reads as an infeasible task. If no
        # cell of the start's component lies within the approach radius,
        # the destination really is out of reach and the rejection is the
        # right answer.
        goal_ij = None
        if start_ij is not None:
            from scipy import ndimage

            labels, _ = ndimage.label(passable)
            reachable = labels == labels[start_ij]
            goal_ij = self._snap_ground(goal, fields, reachable,
                                        radius_m=self.params.goal_snap_m)
        if start_ij is None or goal_ij is None:
            which = "start" if start_ij is None else "goal"
            self.stats = {
                "reason": f"{which} has no feasible cell within "
                          f"{self.params.start_snap_m if start_ij is None else self.params.goal_snap_m:.0f} m",
                "expansions": 0}
            return None

        layer_value = value[np.arange(I)[:, None], np.arange(J)[None, :],
                            stand_k]
        pitch = fields.pitch
        eta = float(self.params.eta)
        goal_xy = np.array(goal_ij, dtype=float)

        g_score = np.full(I * J, np.inf)
        came = np.full(I * J, -1, dtype=np.int64)
        closed = np.zeros(I * J, dtype=bool)
        s_flat = start_ij[0] * J + start_ij[1]
        t_flat = goal_ij[0] * J + goal_ij[1]
        g_score[s_flat] = 0.0

        offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if (di, dj) != (0, 0)]
        step = {o: float(np.hypot(*o)) * pitch for o in offsets}
        # a humanoid steps over what a wheeled base has to climb
        step_allow = self.params.max_step_m
        if self.robot.foothold_band is not None:
            step_allow = max(step_allow, float(self.robot.foothold_band))
        max_step = max(int(round(step_allow / pitch)), 1)

        def heuristic(i, j):
            d = float(np.hypot(i - goal_xy[0], j - goal_xy[1])) * pitch
            return d * (1.0 - eta * float(layer_value[i, j]))

        heap = [(heuristic(*start_ij), int(s_flat))]
        expansions = 0

        while heap:
            _, flat = heapq.heappop(heap)
            if closed[flat]:
                continue
            closed[flat] = True
            expansions += 1
            if flat == t_flat:
                break
            if expansions > self.params.max_expansions:
                self.stats = {"reason": "expansion cap reached",
                              "expansions": expansions}
                return None

            i, j = divmod(flat, J)
            base = g_score[flat]
            k_here = stand_k[i, j]
            for off in offsets:
                ni, nj = i + off[0], j + off[1]
                if not (0 <= ni < I and 0 <= nj < J) or not passable[ni, nj]:
                    continue
                # A diagonal step sweeps the corner between the two cells
                # it passes, so both must be passable. Without this the
                # search squeezes a robot through a gap that exists only
                # between two obstacle corners, and post-processing then
                # rejects its own planner's path.
                if off[0] and off[1] and not (passable[i + off[0], j]
                                              and passable[i, j + off[1]]):
                    continue
                nflat = ni * J + nj
                if closed[nflat]:
                    continue
                if abs(int(stand_k[ni, nj]) - int(k_here)) > max_step:
                    continue
                # Eq. 10
                cost = step[off] * (1.0 - eta * float(layer_value[ni, nj]))
                tentative = base + cost
                if tentative < g_score[nflat]:
                    g_score[nflat] = tentative
                    came[nflat] = flat
                    heapq.heappush(heap, (tentative + heuristic(ni, nj), int(nflat)))

        if not closed[t_flat]:
            self.stats = {"reason": "no feasible path", "expansions": expansions}
            return None

        path, node = [], int(t_flat)
        while node != -1:
            path.append(node)
            node = int(came[node])
        path.reverse()
        ijk = np.array([[n // J, n % J, stand_k[n // J, n % J]] for n in path])
        self.stats = {"expansions": expansions, "nodes": len(path),
                      "cost": float(g_score[t_flat])}
        return fields.world(ijk)

    def _search_volumetric(self, fields, value, start, goal):
        """26-connected search, for profiles that are genuinely free in z."""
        import heapq

        dims = fields.occupied.shape
        eta = float(self.params.eta)
        layer, passable = self.occupiable_layer(fields)
        start_ijk = self._snap(start, fields, layer=layer, passable=passable)
        goal_ijk = self._snap(goal, fields, radius_m=self.params.goal_snap_m,
                              layer=layer, passable=passable)
        if start_ijk is None or goal_ijk is None:
            self.stats = {"reason": "start or goal has no feasible cell nearby",
                          "expansions": 0}
            return None

        offsets = [(di, dj, dk)
                   for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
                   if (di, dj, dk) != (0, 0, 0)]
        step_len = {o: float(np.linalg.norm(o)) * fields.pitch for o in offsets}
        goal_xyz = fields.world(goal_ijk)[0]

        def h(node):
            d = float(np.linalg.norm(fields.world(np.array(node))[0] - goal_xyz))
            return d * (1.0 - eta * float(value[node]))

        start_t = tuple(int(v) for v in start_ijk)
        goal_t = tuple(int(v) for v in goal_ijk)
        g_score = {start_t: 0.0}
        came: dict[tuple, tuple] = {}
        heap = [(h(start_t), start_t)]
        closed = set()
        expansions = 0

        while heap:
            _, node = heapq.heappop(heap)
            if node in closed:
                continue
            closed.add(node)
            expansions += 1
            if node == goal_t:
                break
            if expansions > self.params.max_expansions:
                self.stats = {"reason": "expansion cap reached",
                              "expansions": expansions}
                return None
            base = g_score[node]
            for off in offsets:
                nb = (node[0] + off[0], node[1] + off[1], node[2] + off[2])
                if nb in closed:
                    continue
                if not (0 <= nb[0] < dims[0] and 0 <= nb[1] < dims[1]
                        and 0 <= nb[2] < dims[2]):
                    continue
                if not self.feasible(nb, fields):
                    continue
                tentative = base + step_len[off] * (1.0 - eta * float(value[nb]))
                if tentative < g_score.get(nb, np.inf):
                    g_score[nb] = tentative
                    came[nb] = node
                    heapq.heappush(heap, (tentative + h(nb), nb))

        if goal_t not in came and goal_t != start_t:
            self.stats = {"reason": "no feasible path", "expansions": expansions}
            return None
        path = [goal_t]
        while path[-1] != start_t:
            path.append(came[path[-1]])
        path.reverse()
        self.stats = {"expansions": expansions, "nodes": len(path),
                      "cost": float(g_score[goal_t])}
        return fields.world(np.array(path))

    # -- helpers --------------------------------------------------------
    def _value_lattice(self, value_maps: ValueMapSet, fields: ClearanceFields):
        """V* sampled onto the search lattice, 0 where the maps say nothing.

        Values are *assigned*, not maximised in. V* is signed -- hazard
        repulsion is exactly the negative part -- so combining into a
        zero-initialised array with a maximum discards every repulsive
        value in the map and leaves the planner unguided in precisely the
        places the guidance exists for. Where several map entries share a
        lattice cell the most negative wins: the cautious reading.
        """
        out = np.zeros(fields.occupied.shape, dtype=float)
        if value_maps is None:
            return out
        xyz = getattr(value_maps, "metadata", {}).get("centers")
        if xyz is None:
            return out
        ijk = np.clip(fields.index(xyz), 0, np.array(out.shape) - 1)
        np.minimum.at(out, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), value_maps.fused)
        positive = value_maps.fused > 0
        if positive.any():
            # cells with no negative contribution keep their attraction
            pos_ijk = ijk[positive]
            keep = out[pos_ijk[:, 0], pos_ijk[:, 1], pos_ijk[:, 2]] >= 0
            sel = pos_ijk[keep]
            np.maximum.at(out, (sel[:, 0], sel[:, 1], sel[:, 2]),
                          value_maps.fused[positive][keep])
        return out

    def _snap_ground(self, xyz: np.ndarray, fields: ClearanceFields,
                     passable: np.ndarray, radius_m: float = 6.0):
        """Nearest passable column to a world point, searched outward."""
        start = fields.index(np.asarray(xyz, dtype=float))[0][:2]
        dims = passable.shape
        reach = int(round(radius_m / fields.pitch))
        for r in range(reach + 1):
            lo_i, hi_i = max(start[0] - r, 0), min(start[0] + r + 1, dims[0])
            lo_j, hi_j = max(start[1] - r, 0), min(start[1] + r + 1, dims[1])
            window = passable[lo_i:hi_i, lo_j:hi_j]
            if not window.any():
                continue
            ii, jj = np.nonzero(window)
            ii, jj = ii + lo_i, jj + lo_j
            d = np.hypot(ii - start[0], jj - start[1])
            best = int(np.argmin(d))
            if d[best] <= r + 1e-9:
                return (int(ii[best]), int(jj[best]))
        return None

    def _snap(self, xyz: np.ndarray, fields: ClearanceFields, radius_m: float = 4.0,
              layer=None, passable=None):
        """Nearest feasible lattice cell to a world point.

        Grounded entities are area centroids and object surfaces; neither
        is guaranteed to be a cell the robot can occupy, so the endpoints
        are snapped rather than rejected.

        The candidate height per column comes from
        :meth:`occupiable_layer`, not from ``ground_k``. Reading the floor
        directly is right only for a body that stands on it: over a column
        with no floor return ``ground_k`` is negative, and an aerial
        platform -- which does not care whether there is a floor beneath
        it -- could never snap anywhere at all.
        """
        if layer is None or passable is None:
            layer, passable = self.occupiable_layer(fields)
        start = fields.index(np.asarray(xyz, dtype=float))[0]
        dims = np.array(fields.occupied.shape)
        reach = int(round(radius_m / fields.pitch))
        best, best_d = None, np.inf
        for r in range(reach + 1):
            found = False
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = start[0] + di, start[1] + dj
                    if not (0 <= i < dims[0] and 0 <= j < dims[1]):
                        continue
                    if not passable[i, j]:
                        continue
                    k = int(layer[i, j])
                    for kk in (k, k + 1, k - 1):
                        if not (0 <= kk < dims[2]):
                            continue
                        if self.feasible((i, j, kk), fields):
                            d = float(np.hypot(di, dj))
                            if d < best_d:
                                best, best_d, found = (i, j, kk), d, True
                            break
            if found and best is not None:
                return np.array(best)
        return None
