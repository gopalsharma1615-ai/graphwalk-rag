"""Choose among candidate paths: condition coverage first, relation relevance second, optional local rerank."""
from __future__ import annotations

import itertools

import numpy as np


def _unit(m) -> np.ndarray:
    m = np.asarray(m, np.float32)
    m = m.reshape(1, -1) if m.ndim == 1 else m
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def coverage(gs, path: list[str], themes: np.ndarray, tau: float = 0.2) -> float:
    """Share of high-level themes matched (cosine >= tau) by some edge of the path: does the chain satisfy
    every condition the question states?"""
    if len(path) < 2 or themes is None or len(themes) == 0:
        return 0.0
    T = _unit(themes)
    hit: set[int] = set()
    for u, v in itertools.pairwise(path):
        r = gs.relation_vec(u, v)
        if r is not None:
            hit.update(np.nonzero(T @ r >= tau)[0].tolist())
    return len(hit) / len(T)


def relevance(gs, path: list[str], guide: np.ndarray, w_len: float = 0.08, w_hub: float = 0.15,
              hub_deg: int = 100) -> float:
    """Mean best-match relevance of each relation to the guide vectors, minus length and hub penalties."""
    if len(path) < 2:
        return -1e9
    G = _unit(guide)
    rels = []
    for u, v in itertools.pairwise(path):
        r = gs.relation_vec(u, v)
        rels.append(float((G @ r).max()) if r is not None else 0.0)
    hubs = sum(gs.degree(n) > hub_deg for n in path)
    return float(np.mean(rels)) - w_len * (len(path) - 2) - w_hub * hubs


def rank_paths(gs, paths: list[list[str]], query_vec: np.ndarray, theme_vecs: np.ndarray) -> list[list[str]]:
    guide = np.vstack([_unit(query_vec), _unit(theme_vecs)]) if len(theme_vecs) else _unit(query_vec)
    scored = [((coverage(gs, p, theme_vecs), relevance(gs, p, guide)), p) for p in paths]
    return [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)]


def path_text(gs, path: list[str], edge_chars: int = 48, term_chars: int = 140) -> str:
    """Compact chain for a cross-encoder: every relation, plus what the terminal node IS."""
    hops = " ".join(f"{u} -[{gs.edge(u, v)['desc'][:edge_chars]}]-> {v}" for u, v in itertools.pairwise(path))
    td = gs.node_desc(path[-1])[:term_chars]
    return f"{hops}. {path[-1]}: {td}" if td else hops


def rerank_paths(gs, question: str, ranked: list[list[str]], reranker, pool: int = 60,
                 per_terminal: int = 3) -> list[list[str]]:
    """Cross-encode a terminal-diverse pool. The cheap key plateaus on chains sharing one terminal; without the
    per-terminal cap the reranker never sees the alternatives it is supposed to choose between."""
    per: dict[str, int] = {}
    sel, rest = [], []
    for p in ranked:
        if len(sel) < pool and per.get(p[-1], 0) < per_terminal:
            per[p[-1]] = per.get(p[-1], 0) + 1
            sel.append(p)
        else:
            rest.append(p)
    if not sel or reranker is None:
        return ranked
    scores = reranker.score(question, [path_text(gs, p) for p in sel])
    return [p for p, _ in sorted(zip(sel, scores), key=lambda x: -x[1])] + rest


def verbalize(gs, path: list[str], with_sources: bool = True) -> str:
    """A path as an ordered fact chain - the reader gets evidence in order, never a neighbourhood dump."""
    lines = []
    for i, (u, v) in enumerate(itertools.pairwise(path), 1):
        e = gs.edge(u, v)
        src = ""
        if with_sources:
            docs = sorted({c["doc"] for c in gs.edge_chunks(u, v)})
            src = f"  [source: {', '.join(docs[:2])}]" if docs else ""
        lines.append(f"{i}. {u} -> {v}: {e['desc']}{src}")
    return "\n".join(lines)
