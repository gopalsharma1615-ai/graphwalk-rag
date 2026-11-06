"""Turn a question into graph anchors (forward seeds) and answer candidates (backward seeds).

Anchors: low-level keywords (the things the question NAMES) matched to node names, plus dense entity search on
each keyword to rescue partial or odd names. One anchor per keyword is the floor, so multi-entity questions keep
their second anchor.

Answer candidates - the part that decides whether meet-in-the-middle works in open mode (answer unknown):

1. Rank the anchor's hub-guarded k-hop REACHABLE set, never the whole graph. Global retrieval of "the answer"
   mostly returns topic entities, because a question's embedding is dominated by its anchor; the reachable set
   is guaranteed connectable and, on multi-hop sets mined from the graph, contains the gold answer ~99% of the time.
2. Cosine-prefilter against answer probes (question + high-level themes), then cross-encode with the JOINED
   HIGH-LEVEL KEYWORDS ("produced by, mascots") rather than the full question, which again surfaces the anchor's
   neighbours. Blend cosine and cross-encoder in RANK space - their score scales are not comparable, and letting
   the cross-encoder overwrite the cosine order demotes answers cosine had already found.
3. Give every distance shell its own slots: a ball of radius k is dominated by its outermost shell, so a single
   depth-blind top-m leaves short meets with no candidates at all.
"""
from __future__ import annotations

import numpy as np

from .store import norm
from .traversal import reach

_STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "for", "to", "is", "was", "who", "what",
         "which", "where", "when"}


def string_seeds(gs, text: str, max_seeds: int = 6, min_len: int = 4) -> list[str]:
    """Node names occurring as whole-token spans of `text`, longest first."""
    q = f" {norm(text)} "
    hits = sorted(((len(nn), node) for nn, node in gs.name_index.items()
                   if len(nn) >= min_len and nn not in _STOP and f" {nn} " in q), reverse=True)
    out: list[str] = []
    for _, node in hits:
        if node not in out:
            out.append(node)
        if len(out) >= max_seeds:
            break
    return out


def anchors(gs, low_level: list[str], embed, max_anchors: int = 5, amb: float = 0.93,
            amb_max: int = 3) -> list[dict]:
    """[{node, keyword, tier: exact|dense, score}].

    1. One anchor per keyword: its exact node if the name resolves, else its best dense hit.
    2. Ambiguity allowance: a keyword can name several entities at once (a film, its soundtrack, its sequel).
       Dense hits scoring >= `amb` x the exact node's OWN similarity to the keyword are kept as extra anchors -
       self-relative, because absolute cosine tracks description length, not match quality.
    Nothing else is padded in: extra anchors widen the reachable set the answer must be found in, and an anchor
    sitting next to the answer removes the answer from consideration altogether.
    """
    out: list[dict] = []
    seen: set[str] = set()
    vecs = embed(low_level) if low_level else []
    for kw, v in zip(low_level, vecs):
        v = np.asarray(v, np.float32)
        node = gs.resolve(kw)
        hits = [(n, sc) for n, sc in gs.search_entities(v, k=8) if not gs.is_hub(n)]
        if node and not gs.is_hub(node):
            if node not in seen:
                out.append({"node": node, "keyword": kw, "tier": "exact", "score": 1.0})
                seen.add(node)
            ev = gs.entity_vec(node)
            self_sim = float(ev @ (v / (np.linalg.norm(v) + 1e-9))) if ev is not None else 1.0
            sibs = [(n, sc) for n, sc in hits if n not in seen and sc >= amb * self_sim][:amb_max]
        else:
            sibs = [(n, sc) for n, sc in hits if n not in seen][:1]
        for n, sc in sibs:
            out.append({"node": n, "keyword": kw, "tier": "dense", "score": sc})
            seen.add(n)
    return out[:max(max_anchors, len(low_level))]


def answer_candidates(gs, anchor_nodes: list[str], k: int, probes: np.ndarray, m: int = 30, pool: int = 80,
                      rerank_text: str | None = None, reranker=None, alpha: float = 0.7, shell_slots: int = 20,
                      dist_min: int = 2) -> dict[int, list[str]]:
    """{depth k' (2..k): up to m answer candidates within k' hops}, ranked as described in the module doc."""
    dist = reach(gs, anchor_nodes, k)
    names = [n for n, d in dist.items() if d >= dist_min] or [n for n, d in dist.items() if d >= 1]
    rows, keep = [], []
    for n in names:
        v = gs.entity_vec(n)
        if v is not None:
            rows.append(v)
            keep.append(n)
    if not keep:
        return {}
    P = np.asarray(probes, np.float32).reshape(-1, len(rows[0]))
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    cos = (np.stack(rows) @ P.T).max(axis=1)
    order = np.argsort(-cos)
    sel = set(order[:pool].tolist())
    depth = np.array([dist[n] for n in keep])
    for d in range(dist_min, k + 1):          # every shell keeps its own top slots
        ix = np.nonzero(depth == d)[0]
        if ix.size:
            sel.update(ix[np.argsort(-cos[ix])[:shell_slots]].tolist())
    pre = [keep[i] for i in sorted(sel, key=lambda i: -cos[i])]
    cos_rank = {n: r for r, n in enumerate(pre, 1)}
    key = cos_rank.get
    if reranker is not None and rerank_text:
        scores = reranker.score(rerank_text, [f"{n} ({gs.node_type(n)}). {gs.node_desc(n)[:280]}" for n in pre])
        ce_rank = {n: r for r, (n, _) in enumerate(sorted(zip(pre, scores), key=lambda x: -x[1]), 1)}

        def key(n):
            return alpha * cos_rank[n] + (1 - alpha) * ce_rank[n]
    ranked = sorted(pre, key=key)
    return {kk: [n for n in ranked if dist[n] <= kk][:m] for kk in range(2, k + 1)}
