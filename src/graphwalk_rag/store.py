"""Read a LightRAG working directory as a queryable graph + vector database.

Files used (all written by LightRAG; built once, read many times):

    graph_chunk_entity_relation.graphml   entities + relations, each with a text description (the evidence)
    vdb_entities.json / vdb_relationships.json / vdb_chunks.json   nano-vectordb stores (one base64 float32
                                          matrix in row order + a metadata list)
    kv_store_text_chunks.json             chunk text, from which provenance headers are parsed

Ids follow LightRAG: entity `ent-md5(name)`, relation `rel-md5(src + tgt)`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import cached_property
from pathlib import Path

import networkx as nx
import numpy as np

SEP = "<SEP>"
_HEADER = re.compile(r"\[\s*Source:\s*(?P<doc>[^|\]]+?)\s*(?:\|\s*(?P<what>[^\]]*?))?\s*\]")
_PAGES = re.compile(r"\d+")


def mdid(text: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


def norm(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split())


def load_nano_vdb(path: Path) -> tuple[dict[str, int], np.ndarray, list[dict]]:
    """(id -> row, unit-normalised matrix, metadata rows) from a nano-vectordb JSON file."""
    d = json.loads(Path(path).read_text())
    dim = int(d["embedding_dim"])
    mat = np.frombuffer(base64.b64decode(d["matrix"]), dtype=np.float32).reshape(-1, dim).copy()
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    rows = [{k: v for k, v in r.items() if k != "vector"} for r in d["data"]]
    return {r["__id__"]: i for i, r in enumerate(rows)}, mat, rows


def parse_header(text: str) -> dict:
    """'[Source: report.pdf | Page 3-4]' -> {'doc': 'report.pdf', 'pages': [3, 4], 'card': False}.
    Whitespace-normalised first: chunk re-splitting can insert line breaks inside the header."""
    m = _HEADER.search(" ".join(text[:400].split()))
    if not m:
        return {}
    what = (m.group("what") or "").lower()
    pages = [int(p) for p in _PAGES.findall(what)] if "page" in what else []
    if len(pages) == 2 and pages[1] > pages[0]:
        pages = list(range(pages[0], pages[1] + 1))
    return {"doc": m.group("doc").strip(), "pages": pages, "card": "document card" in what}


class GraphStore:
    def __init__(self, workdir: str | Path, max_degree: int = 150):
        self.dir = Path(workdir)
        self.max_degree = max_degree
        self.G: nx.Graph = nx.read_graphml(self.dir / "graph_chunk_entity_relation.graphml")
        if self.G.is_directed():
            self.G = self.G.to_undirected()
        self.deg = dict(self.G.degree())
        self.name_index = {norm(n): n for n in self.G.nodes}

    # ------------------------------------------------------------------ graph access (shared with Neo4jStore)
    def neighbors(self, n: str):
        return self.G.neighbors(n)

    def nodes(self):
        return iter(self.G.nodes)

    def has(self, n: str) -> bool:
        return n in self.G

    def degree(self, n: str) -> int:
        return self.deg.get(n, 0)

    def is_hub(self, n: str) -> bool:
        return self.deg.get(n, 0) > self.max_degree

    def node_desc(self, n: str) -> str:
        return (self.G.nodes[n].get("description") or "").split(SEP)[0] if n in self.G else ""

    def node_type(self, n: str) -> str:
        return self.G.nodes[n].get("entity_type", "") if n in self.G else ""

    def edge(self, u: str, v: str) -> dict:
        d = self.G.get_edge_data(u, v) or {}
        return {"desc": (d.get("description") or "").split(SEP)[0], "keywords": d.get("keywords", ""),
                "source_id": d.get("source_id", "")}

    def resolve(self, name: str) -> str | None:
        return name if name in self.G else self.name_index.get(norm(name))

    # ------------------------------------------------------------------ vectors (lazy: they are large)
    @cached_property
    def entities(self) -> tuple[dict[str, int], np.ndarray, list[dict]]:
        return load_nano_vdb(self.dir / "vdb_entities.json")

    @cached_property
    def relations(self) -> tuple[dict[str, int], np.ndarray, list[dict]]:
        return load_nano_vdb(self.dir / "vdb_relationships.json")

    @cached_property
    def chunk_vdb(self) -> tuple[dict[str, int], np.ndarray, list[dict]]:
        return load_nano_vdb(self.dir / "vdb_chunks.json")

    def entity_vec(self, n: str) -> np.ndarray | None:
        idx, mat, _ = self.entities
        i = idx.get(mdid(n, "ent-"))
        return None if i is None else mat[i]

    def relation_vec(self, u: str, v: str) -> np.ndarray | None:
        idx, mat, _ = self.relations
        i = idx.get(mdid(u + v, "rel-"))
        if i is None:
            i = idx.get(mdid(v + u, "rel-"))
        return None if i is None else mat[i]

    def search_entities(self, qvec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        _, mat, rows = self.entities
        scores = mat @ (qvec / (np.linalg.norm(qvec) + 1e-9))
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(rows[i]["entity_name"], float(scores[i])) for i in top if rows[i]["entity_name"] in self.G]

    # ------------------------------------------------------------------ provenance
    @cached_property
    def chunks(self) -> dict[str, dict]:
        """chunk id -> {'text', 'doc', 'pages', 'card', 'full_doc_id'}. A continuation piece without its own
        header inherits document and last page from the previous chunk of the same full document."""
        kv = json.loads((self.dir / "kv_store_text_chunks.json").read_text())
        out: dict[str, dict] = {}
        last: dict[str, dict] = {}
        for cid, v in sorted(kv.items(), key=lambda x: (x[1].get("full_doc_id", ""),
                                                         x[1].get("chunk_order_index", 0))):
            text = v.get("content", "")
            meta = parse_header(text)
            fd = v.get("full_doc_id", "")
            if not meta and fd in last:
                meta = {"doc": last[fd]["doc"], "pages": last[fd]["pages"][-1:], "card": False}
            if not meta:
                fp = v.get("file_path", "") or ""
                meta = {"doc": fp.split("#")[0] or "unknown", "pages": [], "card": False}
            meta.update(text=text, full_doc_id=fd)
            out[cid] = meta
            last[fd] = meta
        return out

    def chunk_ids(self, attr_source_id: str) -> list[str]:
        return [c for c in (attr_source_id or "").split(SEP) if c]

    def node_docs(self, n: str) -> set[str]:
        ch = self.chunks
        return {ch[c]["doc"] for c in self.chunk_ids(self.G.nodes[n].get("source_id", "")) if c in ch}

    def edge_chunks(self, u: str, v: str) -> list[dict]:
        ch = self.chunks
        return [ch[c] for c in self.chunk_ids(self.edge(u, v)["source_id"]) if c in ch]
