# Open-vocabulary embodied grounding with large language models for safe robot path planning

Site staff should be able to direct different robots in their own words, and
those words must become paths that respect each robot's body and the safety
distances people state. This repository plans paths for a wheeled Husky+UR5, a
Unitree H1 humanoid and a Crazyflie 2.1 quadrotor from **one 3D scan** and **one
sentence**:

1. the scan becomes a hierarchical 3D scene graph (floors → areas → spots);
2. a large language model maps freely worded goals, hazards and distances onto
   the scene graph's own categories, under a schema that rejects invented ones;
3. the scene is abstracted into a voxel map whose resolution follows the
   robot's operating height band and clearance;
4. value maps over the grounded entities guide an A\* search, and the stated
   safety margins are enforced as **hard** constraints, independent of any code
   the model writes.

This is a code-only release. The scans and derived data are on a public data share (links withheld during review)
(see [Data](#data)); the run outputs are rebuilt by the scripts.

## Pipeline

```
point cloud  +  instruction  +  robot profile
        │
        ├─ 1  scene graph        bodynav/scene_graph/   floors → areas → spots, attributes
        ├─ 2  abstraction        bodynav/abstraction/   height band, task corridor,
        │                                               adaptive voxels r(x)
        ├─ 3  completion         bodynav/abstraction/completion.py
        ├─ 4  grounding + LLM    bodynav/llm/           parse → ground onto scene categories →
        │                                               value-map program → sandbox → validation
        └─ 5  planning           bodynav/planning/      feasibility- and margin-gated,
                                                        value-guided A* → shortcut → smooth
                    │
                    ▼
        path + value maps + scene graph        bodynav/sim/ → Isaac Sim / ROS 2 (optional)
```

## Install

Python 3.10 or newer, Linux (tested on Ubuntu 22.04).

```bash
python3 -m venv .venv                       # or: virtualenv .venv
.venv/bin/pip install -e ".[llm,dev]"
```

To reproduce the exact package set used for the paper (Python 3.10.12):

```bash
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install --no-deps -e .
```

Check the install with the test suite (no network, no GPU, no API key, no
Isaac Sim):

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Isaac Sim is optional. It is only needed for the collision and execution
stage; those scripts run under Isaac's own interpreter
(`~/isaacsim/python.sh`), never the virtualenv.

## Data

The scans and everything derived from them are shared publicly (links withheld during double-anonymized review):

| Site | Contents | Link |
|---|---|---|
| Scene 1 | simulated industrial hall, point cloud | *withheld during double-anonymized review* |
| Scene 2 | simulated two-level bay, point cloud | *withheld during double-anonymized review* |
| Laboratory floor | as-built building scan (real site) | *withheld during double-anonymized review* |
| Derived data | `data/` for all three sites (clouds as `.ply`, voxel grids, task grids, labels, free space, paths, value maps) and the large Isaac stages | *withheld during double-anonymized review* |

Download the derived-data folder's `data/` into the repository root, so that
the tree reads:

```
data/
  labfloor/labfloor.ply           exported scan, building frame, Z-up
  labfloor/point_labels.npz       structure labels
  labfloor/free_space.npz         free-space prior
  labfloor/grid_<robot>.npz       adaptive voxel map per robot
  scene1/  scene2/                the same layout for the simulated sites
```

`data/`, `results/` and `figures/` are gitignored: they are downloaded or
rebuilt.

## Language models

`configs/llm.yaml` names the backends. Paid models read their keys from the
environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); every paid call is priced
and booked in `results/llm_spend.json`, and calls are refused once the ledger
reaches `budget.cap_usd`. Open-weight models (Qwen3.5-9B, Gemma-3-12B,
Llama-3.1-8B, 4-bit) are served locally with vLLM:

```bash
scripts/serve_local_llm.sh qwen3.5-9b          # OpenAI-compatible server on :8000
```

`--offline` selects a scripted backend that only returns canned replies; it
exists for the tests and does not parse instructions.

## Quickstart: one instruction, one robot

With `data/labfloor/` in place, build the scene graph once, then plan:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_scene_graph.py \
    --cloud data/labfloor/labfloor.ply \
    --graph results/labfloor/scene_graph_nodes.json \
    --json  results/labfloor/scene_graph.json

PYTHONPATH=. .venv/bin/python scripts/plan_path.py \
    --graph results/labfloor/scene_graph_nodes.json \
    --grid data/labfloor/grid_husky_ur5.npz --robot husky_ur5 \
    --backend qwen3.5-9b --grounding llm \
    --instruction "Head over to the equipment bay but keep a good 2.5 m away from the clutter."
```

The run's JSON records the grounded goal and hazards, the margins enforced,
the path and its distances to the named hazards.

## Reproducing the experiments

Stage by stage, per site (`labfloor`, `scene1`, `scene2`). Runtimes are for one
laptop with an RTX 5070 Ti.

```bash
# 1  scene graph (as in the quickstart)

# 2  body-conditioned abstraction, one map per robot      (7 min Husky, 28 min H1)
PYTHONPATH=. .venv/bin/python scripts/run_stage2.py \
    --cloud data/labfloor/labfloor.ply \
    --graph results/labfloor/scene_graph_nodes.json \
    --robot husky_ur5 --robot unitree_h1
#    --reference adds the 2.5 cm grading model, --uniform the matched-memory control

# 3  scan completion and emphasis
PYTHONPATH=. .venv/bin/python scripts/run_completion.py \
    --cloud data/labfloor/labfloor.ply \
    --labels data/labfloor/point_labels.npz --robot husky_ur5

# 4  template benchmark: instructions x robots x seeds x configurations
PYTHONPATH=. .venv/bin/python scripts/run_batch.py --site labfloor --seeds 3
#    long batches: run a few instructions per process and merge with
#    scripts/merge_batches.py

# 5  single-factor ablations (replay stored grounding and programs)
scripts/run_revision_controls.sh labfloor
#    grounded-costmap baseline: the stored grounding on a conventional
#    costmap with the grounded hazards inflated by their margins
scripts/run_costmap_gate.sh labfloor

# 6  tables
PYTHONPATH=. .venv/bin/python scripts/make_tables.py
```

Open-vocabulary study (99 instructions, each rewritten as a paraphrase, a
free-form request and an implicit request; `benchmarks/open_vocab/`):

The instruction set ships as `benchmarks/open_vocab/instruction_set.yaml`.
`scripts/build_open_vocab_set.py` rebuilds it from `rewrites.yaml` and the
template benchmark's `results/<site>/batch_all.json` (step 4 on all three
sites), checking margins, vocabulary and coverage on the way.

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_open_vocab_parse.py \
    --backend qwen3.5-9b --grounding llm                            # understanding only
PYTHONPATH=. .venv/bin/python scripts/run_open_vocab_planning.py \
    --backend claude-sonnet --grounding llm --seeds 3 --jobs 6      # full planning runs
PYTHONPATH=. .venv/bin/python scripts/analyze_open_vocab.py        # summary.json + report
```

`--grounding lexical` runs the keyword baseline (a curated synonym table and
token overlap); `--grounding llm` lists the scene's categories in the prompt
and lets the model map each phrase onto one of them.

Simulation stage (Isaac interpreter, optional):

```bash
PYTHONPATH=. .venv/bin/python scripts/build_colliders.py --grid data/labfloor/grid_husky_ur5.npz --robot husky_ur5
PYTHONPATH=. .venv/bin/python scripts/usd_write_colliders.py
~/isaacsim/python.sh scripts/build_exec_stage.py
~/isaacsim/python.sh scripts/execute_path.py --robot husky_ur5
```

## Layout

| Path | Contents |
|---|---|
| `bodynav/core/` | shared data models: `VoxelGrid`, `TaskSpec`, `ValueMapSet`, `PlanResult` |
| `bodynav/scene_graph/` | segmentation, floor/area/spot hierarchy, graph and query API |
| `bodynav/abstraction/` | height band, task corridor, adaptive voxels, completion |
| `bodynav/llm/` | instruction schema, grounding (lexical and LLM), toolset, sandbox, validation, backends with a spend cap |
| `bodynav/planning/` | collision model, feasibility and margin gates, value-guided A\*, smoothing |
| `bodynav/robots/` | platform profiles: width, height band, reach, slope limit |
| `bodynav/eval/` | baselines, task sampling, metrics, result readers |
| `bodynav/sim/` | Isaac Sim / ROS 2 bridge, with an offline no-op backend |
| `configs/` | `default.yaml`, `llm.yaml`, `robots/*.yaml` |
| `benchmarks/` | task templates, adversarial instructions, the open-vocabulary rewrites |
| `scripts/` | every stage above, the ablations, the open-vocabulary study and the tables |
| `tests/` | unit and smoke tests; no network, no Isaac, no API key |

## Robot profiles

| | Husky+UR5 | Unitree H1 | Crazyflie 2.1 |
|---|---|---|---|
| `min_lateral_clearance` (m) | 0.85 | 0.60 | 0.50 |
| `height_band` (m) | 0.0–1.25 | 0.0–1.85 | 0.5–4.0 |
| `r_reach` (m) | 0.85 | 0.80 | — |
| `max_slope_deg` | 30 | 20 | — (aerial) |

Stored run files name our method's configurations `beam`, `beam-no-agent` and
`beam-fixed-grid`; `bodynav/eval/results.py` reads them under the current names.

## Limitations

- Paths are planned and checked against collision geometry; closed-loop
  execution on the physical robots is future work.
- Safety is graded against a 2.5 cm reference model built from the same
  segmented points, not against independent ground truth.
- Grounding is scored by category, not by instance.

## Third-party foundations

Some stages build on open-source work, notably
[HOV-SG](https://github.com/hovsg/HOV-SG) and
[Segment Anything](https://github.com/facebookresearch/segment-anything).
[`docs/third_party.md`](docs/third_party.md) records what is a dependency, what
is adapted and what is reference-only, with licences.

## Citation

Citation details will be added when the paper is published.

## License

MIT, see [`LICENSE`](LICENSE). Third-party components keep their own licences.
