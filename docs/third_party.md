# Third-party foundations

Some BodyNav stages build on prior open-source work — either as installed
dependencies, as adapted code, or as reference implementations we study
while writing our own. This file is the single place that tracks which is
which, so licensing and attribution stay clean as the module fills in.

## Policy

1. **Prefer pip dependencies** (declared in `pyproject.toml`) over copying
   code.
2. **Vendored/adapted code** goes under `third_party/<name>/` with the
   upstream `LICENSE` file preserved, and the adapting module states what
   changed. Keep adaptations minimal and isolated behind our interfaces.
3. **Reference-only** usage (we read their implementation, then write our
   own) gets a docstring credit in the module that follows it, and a row
   here.
4. Before the repo goes public: re-check every row's license
   compatibility with our chosen license (currently TBD in
   `pyproject.toml`).

## Current status

| Upstream | License | How we use it | Where |
|---|---|---|---|
| [HOV-SG](https://github.com/hovsg/HOV-SG) (Werby et al., RSS 2024) | MIT | **Reference** for hierarchical open-vocabulary 3D scene graphs: cross-view mask consolidation, open-vocabulary attribute assignment, graph hierarchy. Candidate for direct reuse of the merging/feature utilities once stage 1 is implemented — decide per component and update this row. | `bodynav/scene_graph/` |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 | **Dependency** (`segmentation` extra): mask generation prompted by our 3D-centric pipeline. | `bodynav/scene_graph/segmentation.py` |
| [Open3D](https://www.open3d.org/) | MIT | **Dependency**: point-cloud I/O, registration, reconstruction. | `bodynav/io/` |
| [NetworkX](https://networkx.org/) | BSD-3 | **Dependency**: scene-graph backbone. | `bodynav/scene_graph/graph.py` |
| VLMaps / LM-Nav / SayCan | various | **Conceptual comparison only** (baselines in the paper); no code reuse planned. | `benchmarks/` |

> Differences from HOV-SG worth preserving when we implement stage 1
> (they are the paper's contribution, not oversights): BodyNav prompts SAM
> *in 3D* (furthest-point-sampled prompts projected per frame) rather
> than per-image; the hierarchy uses site semantics (work zones,
> corridors, restricted zones) rather than room categories; hazard nodes
> carry continuous safety parameters consumed by planning.
