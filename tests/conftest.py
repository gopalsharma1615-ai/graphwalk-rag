"""A tiny LightRAG-format working directory, written from scratch (no LightRAG, no API)."""
from __future__ import annotations

import base64
import hashlib
import json

import networkx as nx
import numpy as np
import pytest

DIM = 16


def vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for tok in text.lower().split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    return v / (np.linalg.norm(v) or 1.0)


def _vdb(path, rows: list[dict], texts: list[str]) -> None:
    mat = np.stack([vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, DIM), np.float32)
    path.write_text(json.dumps({"embedding_dim": DIM, "data": rows,
                                "matrix": base64.b64encode(mat.tobytes()).decode()}))


def md(s: str, p: str) -> str:
    return p + hashlib.md5(s.encode()).hexdigest()


def write_workdir(d, G: nx.Graph, chunks: dict[str, dict]) -> None:
    nx.write_graphml(G, d / "graph_chunk_entity_relation.graphml")
    ents = [{"__id__": md(n, "ent-"), "entity_name": n, "source_id": a.get("source_id", ""),
             "file_path": a.get("file_path", "x.pdf")} for n, a in G.nodes(data=True)]
    _vdb(d / "vdb_entities.json", ents, [f"{n} {a.get('description', '')}" for n, a in G.nodes(data=True)])
    rels = [{"__id__": md(u + v, "rel-"), "src_id": u, "tgt_id": v, "file_path": "x.pdf"} for u, v in G.edges]
    _vdb(d / "vdb_relationships.json", rels, [a.get("description", "") for _, _, a in G.edges(data=True)])
    rows = [{"__id__": cid, "file_path": "x.pdf"} for cid in chunks]
    _vdb(d / "vdb_chunks.json", rows, [c["content"] for c in chunks.values()])
    (d / "kv_store_text_chunks.json").write_text(json.dumps(chunks))


@pytest.fixture()
def film_graph(tmp_path):
    """Anchor 'Ada Brooks' -> film -> director -> birthplace: a 3-hop chain with distractors and one hub."""
    G = nx.Graph()
    facts = [
        ("Ada Brooks", "Night Harbor", "Ada Brooks starred in the film Night Harbor", "c1"),
        ("Night Harbor", "Lena Ortiz", "Night Harbor was directed by Lena Ortiz", "c2"),
        ("Lena Ortiz", "Valparaiso", "Lena Ortiz was born in the city of Valparaiso", "c3"),
        ("Ada Brooks", "Drama Award", "Ada Brooks won the Drama Award", "c1"),
        ("Drama Award", "Gala Night", "The Drama Award is presented at Gala Night", "c4"),
        ("Night Harbor", "Ocean Score", "Night Harbor features the Ocean Score soundtrack", "c2"),
    ]
    for u, v, desc, c in facts:
        for n in (u, v):
            G.add_node(n, entity_type="thing", description=f"{n} entity", source_id=c)
        G.add_edge(u, v, description=desc, keywords="", source_id=c, weight=1.0)
    G.add_node("United States", entity_type="place", description="country", source_id="c1")
    for n in ("Ada Brooks", "Lena Ortiz", "Gala Night", "Ocean Score", "Valparaiso", "Drama Award", "Night Harbor"):
        G.add_edge("United States", n, description=f"{n} is linked to the United States", keywords="",
                   source_id="c1", weight=1.0)
    chunks = {
        "c0": {"content": "[Source: films.pdf | Document card]\nDocument: films", "full_doc_id": "d1",
               "chunk_order_index": 0},
        "c1": {"content": "[Source: films.pdf | Page 1]\nAda Brooks...", "full_doc_id": "d1", "chunk_order_index": 1},
        "c2": {"content": "[Source: films.pdf | Page 2-3]\nNight Harbor...", "full_doc_id": "d1",
               "chunk_order_index": 2},
        "c3": {"content": "continuation without a header", "full_doc_id": "d1", "chunk_order_index": 3},
        "c4": {"content": "[Source: awards.pdf | Page 7]\nGala...", "full_doc_id": "d2", "chunk_order_index": 0},
    }
    write_workdir(tmp_path, G, chunks)
    return tmp_path
