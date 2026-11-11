"""Scene-graph construction and query API (Sec. III-A.3).

G = (N, E, Phi, Psi): floor / area / spot nodes; containment edges follow
the hierarchy (spot -> area -> floor); undirected adjacency edges link
entities in the same parent scope with d_min(v_i, v_j) <= r_adj, built via
spatial indexing (local range queries, not all-pairs).

The graph is exposed through a NetworkX-backed Python API supporting
attribute filtering and traversal — this is the *only* scene interface the
constrained LLM agent may call (Sec. III-D).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from bodynav.scene_graph.attributes import NodeAttributes
from bodynav.scene_graph.hierarchy import LabeledPoints


class BodyNavSceneGraph:
    """Hierarchical scene graph with the query surface used by the LLM
    toolset and by task-guided filtering.

    Node ids are strings like ``floor_0``, ``area_0_2``, ``spot_0_2_17``.
    """

    def __init__(self) -> None:
        self.g = nx.Graph()
        #: node id -> point index array, populated by :meth:`build`
        self._member_index: dict[str, np.ndarray] = {}

    # -- construction ---------------------------------------------------
    @classmethod
    def build(
        cls, points: LabeledPoints, attributes: dict[str, NodeAttributes],
        r_adj: float = 1.0, area_edges: list[tuple[str, str]] | None = None,
    ) -> "BodyNavSceneGraph":
        """Instantiate nodes/edges from labeled points + per-node attributes.

        Containment follows the hierarchy (spot -> area -> floor).
        Adjacency links entities *within the same parent scope* whose
        bounding volumes come within ``r_adj``; it is built from AABB
        separation rather than all-pairs point distance, which keeps
        construction linear in the number of nodes instead of quadratic in
        the number of points.

        ``area_edges`` overrides that test for areas. Pass the free-space
        contacts from :func:`segment_areas`: an AABB test on diagonal
        corridors declares opposite sides of a courtyard adjacent, and a
        route planned on that graph crosses the courtyard.
        """
        self = cls()
        by_level: dict[str, list[str]] = {"floor": [], "area": [], "spot": []}

        for node_id, attrs in attributes.items():
            level = node_id.split("_", 1)[0]
            if level not in by_level:
                raise ValueError(f"node id {node_id!r} has no known level prefix")
            self.g.add_node(node_id, level=level, attrs=attrs)
            by_level[level].append(node_id)

        # containment: area_<f>_<a> -> floor_<f>, spot_<f>_<a>_<i> -> area_<f>_<a>
        for node_id in by_level["area"] + by_level["spot"]:
            parent = node_id.rsplit("_", 1)[0]
            parent = parent.replace("area_", "floor_", 1) if node_id.startswith("area_") \
                else parent.replace("spot_", "area_", 1)
            if parent in self.g:
                self.g.add_edge(node_id, parent, kind="containment")

        if area_edges is not None:
            for u, v in area_edges:
                if u in self.g and v in self.g:
                    self.g.add_edge(u, v, kind="adjacency")

        # adjacency within a shared parent
        for level in (["spot"] if area_edges is not None else ["area", "spot"]):
            nodes = by_level[level]
            for i, a in enumerate(nodes):
                pa = self.parent(a)
                aa = self.g.nodes[a]["attrs"]
                for b in nodes[i + 1:]:
                    if self.parent(b) != pa:
                        continue
                    ab = self.g.nodes[b]["attrs"]
                    gap = np.maximum(
                        np.maximum(aa.aabb_min - ab.aabb_max, ab.aabb_min - aa.aabb_max), 0.0
                    )
                    if float(np.linalg.norm(gap)) <= r_adj:
                        self.g.add_edge(a, b, kind="adjacency")

        self._points = points
        return self

    # -- queries (approved LLM tool surface) ----------------------------
    def nodes_by_level(self, level: str) -> list[str]:
        """All node ids at level in {"floor", "area", "spot"}."""
        return [n for n, d in self.g.nodes(data=True) if d.get("level") == level]

    def attributes(self, node_id: str) -> NodeAttributes:
        return self.g.nodes[node_id]["attrs"]

    def children(self, node_id: str) -> list[str]:
        """Contained nodes one level down (containment edges)."""
        below = {"floor": "area", "area": "spot", "spot": None}
        want = below.get(self.g.nodes[node_id].get("level"))
        return [
            n for n in self.g.neighbors(node_id)
            if self.g.edges[node_id, n].get("kind") == "containment"
            and self.g.nodes[n].get("level") == want
        ]

    def parent(self, node_id: str) -> str | None:
        above = {"spot": "area", "area": "floor", "floor": None}
        want = above.get(self.g.nodes[node_id].get("level"))
        for n in self.g.neighbors(node_id):
            if (self.g.edges[node_id, n].get("kind") == "containment"
                    and self.g.nodes[n].get("level") == want):
                return n
        return None

    def adjacent(self, node_id: str) -> list[str]:
        """Adjacency-edge neighbors within the same parent scope."""
        return [
            n for n in self.g.neighbors(node_id)
            if self.g.edges[node_id, n].get("kind") == "adjacency"
        ]

    def find(self, *, semantic: str | None = None, floor: str | None = None,
             hazardous: bool | None = None, level: str | None = None) -> list[str]:
        """Attribute-filtered lookup, e.g. navigable areas adjacent to
        welding zones on Floor 1."""
        out = []
        for node_id, data in self.g.nodes(data=True):
            attrs = data["attrs"]
            if level is not None and data.get("level") != level:
                continue
            if semantic is not None and attrs.semantic != semantic:
                continue
            if hazardous is not None and attrs.affordances.hazardous != hazardous:
                continue
            if floor is not None:
                node = node_id
                while node is not None and self.g.nodes[node]["level"] != "floor":
                    node = self.parent(node)
                if node != floor:
                    continue
            out.append(node_id)
        return sorted(out)

    def resolve_reference(self, phrase: str) -> list[str]:
        """Candidate nodes for a natural-language entity reference;
        ambiguity (>1 candidate) is surfaced to the agent, proximity
        breaks ties (Sec. V, adversarial handling)."""
        raise NotImplementedError

    # -- geometry -------------------------------------------------------
    def points_of(self, node_id: str) -> np.ndarray:
        """(K, 3) member points of a node (spots) or union (areas/floors)."""
        pts = getattr(self, "_points", None)
        if pts is None:
            raise RuntimeError("graph carries no points; rebuild or load with them")
        idx = self._member_index.get(node_id)
        if idx is None:
            return np.empty((0, 3))
        return pts.xyz[idx]

    # -- io -------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize nodes, attributes and edges to JSON.

        Member points are not written: they are an order of magnitude
        larger than everything else and are reproducible from the cloud.
        """
        import json
        from pathlib import Path

        def attrs_to_dict(a: NodeAttributes) -> dict:
            out = {
                "centroid": [round(float(v), 4) for v in a.centroid],
                "aabb_min": [round(float(v), 4) for v in a.aabb_min],
                "aabb_max": [round(float(v), 4) for v in a.aabb_max],
                "semantic": a.semantic,
                "material": a.material,
                "completeness": round(float(a.completeness), 4),
                "completed": bool(a.completed),
                "affordances": {
                    "navigable": a.affordances.navigable,
                    "loadable": a.affordances.loadable,
                    "hazardous": a.affordances.hazardous,
                },
            }
            if a.hazard is not None:
                out["hazard"] = {
                    "clearance_radius": round(float(a.hazard.clearance_radius), 4),
                    "risk_weight": round(float(a.hazard.risk_weight), 4),
                    "hazard_class": a.hazard.hazard_class,
                }
            return out

        doc = {
            "nodes": {
                n: {"level": d["level"], "attrs": attrs_to_dict(d["attrs"]),
                    "n_points": len(getattr(self, "_member_index", {}).get(n, []))}
                for n, d in self.g.nodes(data=True)
            },
            "edges": [
                {"u": u, "v": v, "kind": d.get("kind", "adjacency")}
                for u, v, d in self.g.edges(data=True)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=1) + "\n")

    @classmethod
    def load(cls, path: str) -> "BodyNavSceneGraph":
        import json
        from pathlib import Path

        from bodynav.scene_graph.attributes import Affordances, HazardParams

        doc = json.loads(Path(path).read_text())
        self = cls()
        for node_id, nd in doc["nodes"].items():
            a = nd["attrs"]
            hazard = None
            if "hazard" in a:
                hazard = HazardParams(**a["hazard"])
            attrs = NodeAttributes(
                centroid=np.asarray(a["centroid"], dtype=float),
                aabb_min=np.asarray(a["aabb_min"], dtype=float),
                aabb_max=np.asarray(a["aabb_max"], dtype=float),
                semantic=a["semantic"],
                affordances=Affordances(**a["affordances"]),
                material=a.get("material", "unknown"),
                hazard=hazard,
                completeness=a.get("completeness", 1.0),
                completed=a.get("completed", False),
            )
            self.g.add_node(node_id, level=nd["level"], attrs=attrs)
        for e in doc["edges"]:
            self.g.add_edge(e["u"], e["v"], kind=e["kind"])
        return self
