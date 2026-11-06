"""Document-routed guided walk, for long structured corpora (regulations, policies, contracts, statutes).

On such corpora meet-in-the-middle has a different failure: questions rarely have a single crisp answer entity,
and the graph is dominated by hub entities (the regulator, "the company", defined terms) through which an
unguided walk drifts. Measured on a 101-question regulatory set, only 19% of top-20 path edges were about the
question - 43% on the first hop and ~10% from the second hop on.

So every hop is filtered by the question:

  1. ROUTE  question vs one "document card" chunk per document: cosine top-N, then cross-encoder, then the top
            documents. A soft preference (a score bonus), never a hard filter.
  2. SEEDS  union of keyword anchors, question-embedding entity search, and the most question-similar entities
            extracted from the routed documents, then cross-encoder, then the top seeds (never a hub).
  3. WALK   beam search up to max_hops. Each hop: frontier edges (no hubs, no cycles), cosine(question, relation)
            prefilter (+ bonus for edges extracted from a routed document), cross-encoder on
            "u -- v: description", keep edges scoring >= edge_min, keep the best `beam` paths by mean score.
  4. RANK   kept paths by mean edge relevance (ties: shorter first). Evidence = the pages behind the edges.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .seeding import anchors as find_anchors


@dataclass
class GuideConfig:
    route_cos: int = 30
    route_docs: int = 3
    seeds: int = 6
    seed_pool: int = 30
    max_hops: int = 3
    prefilter: int = 80
    edge_min: float = 0.3
    beam: int = 25
    doc_bonus: float = 0.05
    top: int = 20


class GuidedWalk:
    def __init__(self, gs, chat, embed, reranker, cfg: GuideConfig | None = None):
        self.gs, self.chat, self.embed, self.rr = gs, chat, embed, reranker
        self.cfg = cfg or GuideConfig()
        _idx, mat, rows = gs.chunk_vdb
        meta = gs.chunks
        self.cards = [(i, r["__id__"]) for i, r in enumerate(rows) if meta.get(r["__id__"], {}).get("card")]
        self.chunk_mat = mat
        self.doc_entities: dict[str, list[str]] = defaultdict(list)
        for n in gs.nodes():
            for d in gs.node_docs(n):
                self.doc_entities[d].append(n)

    def _ce(self, q: str, texts: list[str]) -> list[float]:
        return self.rr.score(q, texts) if texts else []

    def route(self, question: str, qvec: np.ndarray) -> list[str]:
        if not self.cards:
            return []
        rows = np.array([i for i, _ in self.cards])
        sims = self.chunk_mat[rows] @ (qvec / (np.linalg.norm(qvec) + 1e-9))
        top = np.argsort(-sims)[: self.cfg.route_cos]
        ids = [self.cards[i][1] for i in top]
        texts = [self.gs.chunks[c]["text"][:1500] for c in ids]
        order = sorted(zip(ids, self._ce(question, texts)), key=lambda x: -x[1])
        docs: list[str] = []
        for cid, _ in order:
            d = self.gs.chunks[cid]["doc"]
            if d not in docs:
                docs.append(d)
            if len(docs) >= self.cfg.route_docs:
                break
        return docs

    def walk(self, question: str) -> dict:
        gs, cfg = self.gs, self.cfg
        kw = self.chat.keywords(question)
        qvec = self.embed([question])[0]
        qn = qvec / (np.linalg.norm(qvec) + 1e-9)
        docs = self.route(question, qvec)
        pool = {a["node"] for a in find_anchors(gs, kw["low_level"], self.embed)}
        pool |= {n for n, _ in gs.search_entities(qvec, k=cfg.seed_pool)}
        for d in docs:
            ents = [n for n in self.doc_entities.get(d, []) if not gs.is_hub(n)]
            vecs = [(n, gs.entity_vec(n)) for n in ents]
            vecs = [(n, v) for n, v in vecs if v is not None]
            vecs.sort(key=lambda x: -float(x[1] @ qn))
            pool |= {n for n, _ in vecs[: cfg.seed_pool]}
        pool = [n for n in pool if gs.has(n) and not gs.is_hub(n)]
        seed_scores = self._ce(question, [f"{n}: {gs.node_desc(n)[:200]}" for n in pool])
        seeds = [n for n, _ in sorted(zip(pool, seed_scores), key=lambda x: -x[1])[: cfg.seeds]]

        routed = set(docs)
        beam = [([s], []) for s in seeds]          # (path, edge scores)
        kept: list[tuple[list[str], list[float]]] = []
        for _ in range(cfg.max_hops):
            cand = []
            for path, sc in beam:
                for nb in gs.neighbors(path[-1]):
                    if nb in path or gs.is_hub(nb):
                        continue
                    r = gs.relation_vec(path[-1], nb)
                    cos = float(r @ qn) if r is not None else 0.0
                    if routed and {c["doc"] for c in gs.edge_chunks(path[-1], nb)} & routed:
                        cos += cfg.doc_bonus
                    cand.append((cos, path, sc, nb))
            cand.sort(key=lambda x: -x[0])
            cand = cand[: cfg.prefilter]
            texts = [f"{p[-1]} -- {nb}: {gs.edge(p[-1], nb)['desc'][:300]}" for _, p, _, nb in cand]
            ce = self._ce(question, texts)
            nxt = [(p + [nb], sc + [s]) for (_, p, sc, nb), s in zip(cand, ce) if s >= cfg.edge_min]
            nxt.sort(key=lambda x: -float(np.mean(x[1])))
            beam = nxt[: cfg.beam]
            kept += beam
            if not beam:
                break
        kept.sort(key=lambda x: (-float(np.mean(x[1])), len(x[0])))
        paths = [p for p, _ in kept[: cfg.top]]
        pages = []
        for p in paths:
            for u, v in itertools.pairwise(p):
                for c in gs.edge_chunks(u, v):
                    for pg in c["pages"] or [None]:
                        if (c["doc"], pg) not in pages:
                            pages.append((c["doc"], pg))
        return {"question": question, "keywords": kw, "routed_docs": docs, "seeds": seeds, "paths": paths,
                "evidence_pages": pages}
