"""Neo4j backend: export a LightRAG graph, and a drop-in replacement for GraphStore's graph access.

`Neo4jStore` bulk-loads adjacency and attributes once (fast walks, results identical to NetworkX), while vectors
and provenance still come from the LightRAG working directory. Connection settings: NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD (never hard-coded).
"""
from __future__ import annotations

import os
from collections import defaultdict

from .store import SEP, GraphStore


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                                auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))


def export(gs: GraphStore, label: str = "Entity", batch: int = 2000) -> dict:
    """Write nodes and RELATED edges under a workspace label (re-runnable: MERGE)."""
    nodes = [{"name": n, "type": d.get("entity_type", ""), "description": d.get("description", ""),
              "source_id": d.get("source_id", "")} for n, d in gs.G.nodes(data=True)]
    edges = [{"u": u, "v": v, "description": d.get("description", ""), "keywords": d.get("keywords", ""),
              "source_id": d.get("source_id", ""), "weight": float(d.get("weight", 1.0) or 1.0)}
             for u, v, d in gs.G.edges(data=True)]
    with _driver() as drv, drv.session() as s:
        s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE")
        for i in range(0, len(nodes), batch):
            s.run(f"UNWIND $rows AS r MERGE (n:`{label}` {{name: r.name}}) "
                  "SET n.entity_type = r.type, n.description = r.description, n.source_id = r.source_id",
                  rows=nodes[i:i + batch])
        for i in range(0, len(edges), batch):
            s.run(f"UNWIND $rows AS r MATCH (a:`{label}` {{name: r.u}}), (b:`{label}` {{name: r.v}}) "
                  "MERGE (a)-[e:RELATED]-(b) SET e.description = r.description, e.keywords = r.keywords, "
                  "e.source_id = r.source_id, e.weight = r.weight", rows=edges[i:i + batch])
    return {"nodes": len(nodes), "edges": len(edges), "label": label}


class Neo4jStore(GraphStore):
    """Same interface as GraphStore; graph topology and attributes read from Neo4j."""

    def __init__(self, workdir: str, label: str = "Entity", max_degree: int = 150):
        self.dir = __import__("pathlib").Path(workdir)
        self.max_degree = max_degree
        self.adj: dict[str, list[str]] = defaultdict(list)
        self.attrs: dict[str, dict] = {}
        self.eattrs: dict[frozenset, dict] = {}
        with _driver() as drv, drv.session() as s:
            for r in s.run(f"MATCH (n:`{label}`) RETURN n.name AS n, n.entity_type AS t, n.description AS d, "
                           "n.source_id AS s"):
                self.attrs[r["n"]] = {"entity_type": r["t"] or "", "description": r["d"] or "",
                                      "source_id": r["s"] or ""}
            for r in s.run(f"MATCH (a:`{label}`)-[e:RELATED]-(b:`{label}`) WHERE a.name < b.name "
                           "RETURN a.name AS u, b.name AS v, e.description AS d, e.keywords AS k, e.source_id AS s"):
                self.adj[r["u"]].append(r["v"])
                self.adj[r["v"]].append(r["u"])
                self.eattrs[frozenset((r["u"], r["v"]))] = {"description": r["d"] or "", "keywords": r["k"] or "",
                                                            "source_id": r["s"] or ""}
        for n in self.adj:                      # deterministic neighbour order = deterministic results
            self.adj[n].sort()
        self.deg = {n: len(self.adj.get(n, [])) for n in self.attrs}
        from .store import norm

        self.name_index = {norm(n): n for n in self.attrs}

    def neighbors(self, n):
        return iter(self.adj.get(n, ()))

    def has(self, n):
        return n in self.attrs

    def nodes(self):
        return iter(self.attrs)

    def node_desc(self, n):
        return (self.attrs.get(n, {}).get("description") or "").split(SEP)[0]

    def node_type(self, n):
        return self.attrs.get(n, {}).get("entity_type", "")

    def edge(self, u, v):
        d = self.eattrs.get(frozenset((u, v)), {})
        return {"desc": (d.get("description") or "").split(SEP)[0], "keywords": d.get("keywords", ""),
                "source_id": d.get("source_id", "")}

    def resolve(self, name):
        from .store import norm

        return name if name in self.attrs else self.name_index.get(norm(name))

    def node_docs(self, n):
        ch = self.chunks
        return {ch[c]["doc"] for c in self.chunk_ids(self.attrs.get(n, {}).get("source_id", "")) if c in ch}
