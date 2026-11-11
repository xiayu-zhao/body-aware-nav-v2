# BodyNav architecture notes

## Data flow and stage contracts

```
                    ┌────────────────────────────────────────────┐
 point cloud +      │ 1. bodynav.scene_graph                        │
 posed RGB-D  ────▶ │ segmentation → hierarchy → graph + attrs   │──▶ BodyNavSceneGraph
                    └────────────────────────────────────────────┘        │
                                                                          │ grounded
                    ┌────────────────────────────────────────────┐        │ entities
 LabeledPoints ───▶ │ 2. bodynav.abstraction                        │◀───────┤ (V_t mask)
                    │ robot_filter → task_filter →               │        │
                    │ completion (φ_E) → voxelizer               │──▶ VoxelGrid
                    └────────────────────────────────────────────┘        │
                    ┌────────────────────────────────────────────┐        │
 instruction ─────▶ │ 3. bodynav.llm                                │◀───────┘
                    │ schema.parse → grounding → agent code-gen  │
                    │ → toolset sandbox → validation             │──▶ ValueMapSet
                    └────────────────────────────────────────────┘
                    ┌────────────────────────────────────────────┐
                    │ 4. bodynav.planning                           │
                    │ obstacles/gates → value-guided A* →        │
                    │ shortcut → B-spline → re-validate          │──▶ PlanResult
                    └────────────────────────────────────────────┘
                                       │
                    bodynav.sim (SimBridge: IsaacSimBridge/Offline) ──▶ execution
```

The **bidirectional** loop: grounded entities from stage 3's parse feed
*back* into stage 2 as the task mask V_t (where density is preserved),
while robot kinematics shape both the slice and the voxel resolution.
This is why `BodyNavPipeline.build_abstraction` takes the instruction.

## Design decisions

- **Sparse grid on a uniform base lattice.** Coarse voxels carry a larger
  `resolution` attribute instead of a separate octree; `world2voxel`
  stays a single affine map, which keeps the LLM toolset contract simple
  and deterministic.
- **Dyadic resolution ladder.** Eq. 4 yields a continuous r(v); the
  voxelizer snaps it to `r_min * 2^k` capped at r0, so every level nests
  exactly on the base lattice and no voxel straddles a coarser cell
  boundary. With the manuscript's r_min = 2.5 cm / r0 = 50 cm the ladder
  is {2.5, 5, 10, 20, 40} cm — note the top rung is 40 cm, not 50. Where
  a coarse cell would overlap points assigned a finer level, the coarse
  cell is dropped and the finer voxels stand, so the grid stays a
  partition.
- **Pure-numpy PLY reader.** `bodynav.io.ply` memory-maps the vertex block
  and subsamples by stride, so the multi-gigabyte real-world scan loads
  without open3d (which is now optional, used only for non-PLY inputs).
- **Toolset-only sandbox.** Agent-generated code executes in a namespace
  containing exactly the `ApprovedToolset` registry + numpy. Determinism
  comes from temperature 0 + fixed seeds (configs/llm.yaml).
- **Completion is auditable.** Synthetic geometry is flagged per voxel
  (`Voxel.synthetic`, `confidence=c(o)`) so the planner and any reviewer
  can separate observed from inferred surfaces; a `CompletionReport`
  feeds the real-world evaluation table.
- **Simulator-agnostic core.** Only `bodynav.sim` touches ROS 2 / Isaac
  Sim, with imports deferred; `OfflineBridge` runs the pipeline on saved
  scans (the as-built lab-floor dataset path).
- **Baselines** (Pure-Planner, Motion-Primitives, LLM-as-Planner) will
  live under `benchmarks/` and reuse the same grid/graph, differing only
  in the LLM role, mirroring the controlled comparison in the paper.

## Manuscript ↔ code map

| Manuscript | Code |
|---|---|
| Eq. 1 (prompt projection) | `scene_graph/segmentation.py:project_prompts` |
| Eq. 2 (height slice) | `abstraction/robot_filter.py:height_slice` ✅ |
| Eq. 3 (workspace kernel) | `abstraction/robot_filter.py:workspace_kernel_mask` ✅ |
| Eq. 4 (+ φ_E) (adaptive resolution) | `abstraction/voxelizer.py` ✅ |
| Eq. 5 (completeness score) | `abstraction/completion.py:completeness_score` |
| Eqs. 6–8 (V_A, V_H, V_C) | `llm/value_maps.py` |
| Eq. 9 (fusion) | `llm/value_maps.py:fuse` |
| Eqs. 10–11 (cost/heuristic) | `planning/astar.py` |
| Sec. III-E.1 (collision model) | `planning/obstacles.py` |
| Sec. IV (platforms) | `robots/profiles.py`, `configs/robots/` ✅ |
| Sec. IV (metrics) | `benchmarks/metrics.py` |

Every equation above is implemented and covered by `tests/`; the ✅ marks predate
that and are kept only because they point at the file that owns each one.

## Running stage 2 standalone

Stage 2 needs no scene graph and no LLM, so it can be pointed straight at
a saved scan to produce the reduction numbers for Sec. III-B.1 and
Table V:

```
python scripts/run_abstraction.py data/labfloor/labfloor.ply \
    --robot husky_ur5 --max-points 20000000 --json outputs/labfloor_stats.json
```

`LabeledPoints.from_xyz` supplies the unlabeled-cloud bootstrap; once
stage 1 lands, the same script takes real per-point semantics and the
φ_T / φ_E cues stop being zero.
