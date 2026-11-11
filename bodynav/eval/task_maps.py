"""Per-instruction abstraction: the map Eq. 4 actually describes.

Eq. 4 makes voxel resolution a function of the grounded instruction --
phi_T is 1 inside the task mask and decays outside it. Building one map
per platform and running every instruction against it therefore does not
evaluate the method; it evaluates a map specialised for whichever task
happened to be built into it, which on this floor was a single fixed
anchor pair. The effect is not small: rebuilding for the task moved the
H1 from finding no route on three sampled tasks to finding two, on grids
a quarter the size.

Grids are cached on disk, keyed by platform and task, because the
evaluation runs in chunks and rebuilding 3 M points per chunk would cost
more than the planning it feeds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from bodynav.abstraction import AdaptiveVoxelizer, VoxelizerParams, height_slice
from bodynav.abstraction.robot_filter import workspace_kernel_mask
from bodynav.abstraction.task_filter import build_corridor, priority_merge, task_mask
from bodynav.core.types import as_arrays
from bodynav.io import load_grid_arrays, save_grid
from bodynav.scene_graph.hierarchy import LabeledPoints


def build_params(config: dict) -> VoxelizerParams:
    v = config["abstraction"]["voxel"]
    return VoxelizerParams(r0=v["r0_m"], r_min=v["r_min_m"], alpha=v["alpha"],
                           beta=v["beta"], gamma=v["gamma"],
                           foothold_buffer=v["foothold_buffer_m"])


def load_points(cloud: Path, labels: Path) -> LabeledPoints:
    """The labelled cloud stage 2 consumes, as stage 2 assembles it."""
    from bodynav.io import load_cloud

    xyz, colors = load_cloud(str(cloud))
    lab = np.load(labels, allow_pickle=True)
    semantic = np.asarray(lab["vocabulary"][lab["semantic_code"]], dtype=object)
    return LabeledPoints(
        xyz=xyz, floor_id=lab["floor_id"].astype(int),
        area_id=lab["area_id"].astype(int),
        instance_id=lab["instance_id"].astype(int),
        semantic=semantic, colors=colors)


class TaskMapBuilder:
    """Adaptive grids conditioned on the task, cached on disk and in RAM."""

    def __init__(self, points: LabeledPoints, graph, params: VoxelizerParams,
                 *, cache_dir: Path, critical_spacing: float = 0.02,
                 background_spacing: float = 0.05):
        self.points = points
        self.graph = graph
        self.params = params
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.critical_spacing = critical_spacing
        self.background_spacing = background_spacing
        self._sliced: dict[str, LabeledPoints] = {}
        self._grids: dict[str, object] = {}
        self._hazards = graph.find(hazardous=True)

    def sliced(self, robot):
        """Eq. 2's band, which depends on the platform and not the task."""
        if robot.name not in self._sliced:
            self._sliced[robot.name] = height_slice(self.points, robot)
        return self._sliced[robot.name]

    def _key(self, robot, start_id: str, goal_id: str, margin: float) -> str:
        p = self.params
        stamp = f"{p.r0}:{p.r_min}:{p.alpha}:{p.beta}:{p.gamma}:" \
                f"{self.critical_spacing}:{self.background_spacing}:{margin:g}"
        digest = hashlib.sha1(
            f"{robot.name}|{start_id}|{goal_id}|{stamp}".encode()).hexdigest()[:12]
        return f"grid_{robot.name}_{digest}"

    def grid_for(self, robot, start_id: str, goal_id: str, *, verbose=False,
                 hazard_margin_m: float = 0.0):
        """The adaptive grid for this platform on this task.

        ``hazard_margin_m`` is the largest margin the instruction stated,
        and it widens the corridor. Without it the fine region follows the
        route the robot would take if nobody had asked for anything, and a
        stated margin then pushes it into background the map deliberately
        coarsened: the gate passed a path the 2.5 cm reference measured at
        0.224 m for a body needing 0.425 m. The margin bounds how far the
        instruction can displace the route, so it bounds how much wider
        the accurate region has to be.
        """
        key = self._key(robot, start_id, goal_id, hazard_margin_m)
        if key in self._grids:
            return self._grids[key]
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            grid = load_grid_arrays(path)
            self._grids[key] = grid
            return grid

        sliced = self.sliced(robot)
        # Eq. 3's workspace kernel along this task's corridor, then V_t
        corridor = build_corridor(self.graph, start_id, goal_id, [],
                                  base_width=2.0 + 2.0 * float(hazard_margin_m))
        kernel = workspace_kernel_mask(sliced.xyz, corridor.centerline[:, :2],
                                       robot, extra_radius=hazard_margin_m)
        v_t = task_mask(sliced, corridor, self._hazards, self.graph) & kernel
        merged = priority_merge(sliced, v_t,
                                critical_spacing=self.critical_spacing,
                                background_spacing=self.background_spacing)
        grid = AdaptiveVoxelizer(self.params, robot).voxelize(
            merged, merged.extras["v_t"])
        save_grid(grid, path)
        arrays = as_arrays(grid)
        self._grids[key] = arrays
        if verbose:
            print(f"    built {robot.name} map for {start_id}->{goal_id}: "
                  f"{len(arrays.index):,} voxels -> {path.name}")
        return arrays
