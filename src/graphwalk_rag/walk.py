"""GraphWalk: answer multi-hop questions by walking the graph, not by retrieving a bag of chunks.

    question ─► keywords (low-level = anchors, high-level = relation themes)
             ─► anchors ─► answer candidates from the anchors' reachable set (cosine ⊕ local cross-encoder)
             ─► bidirectional meet-in-the-middle for k = 2..max_hops ─► rank by (theme coverage, relevance)
             ─► optional path rerank ─► top-K verbalized chains ─► reader LLM ─► answer

The reader gets the K best chains and picks the one that answers the question: a reader is a far better
final-hop judge than cosine similarity.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .paths import rank_paths, rerank_paths, verbalize
from .seeding import anchors as find_anchors
from .seeding import answer_candidates
from .traversal import bidirectional_meet, enumerate_from

READER_PROMPT = """Answer the question using ONLY the evidence chains below. Each chain is an ordered sequence of
facts from a knowledge graph. Pick the chain that actually answers the question, follow it, and give the final
answer as a short phrase. If no chain answers it, reply exactly INSUFFICIENT.

Question: {question}

Evidence chains:
{chains}

Answer:"""

JUDGE_PROMPT = """Grade the predicted answer against the gold answer for the question.
CORRECT if it refers to the same entity or value (aliases, extra detail and minor wording differences are fine).
INCORRECT otherwise, including when the prediction refuses, hedges or gives several candidates.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Reply with one word: CORRECT or INCORRECT."""

ABSTAIN = re.compile(r"\b(insufficient|not (enough|sufficient) (information|evidence)|cannot (be )?determine|"
                     r"unable to (answer|determine)|no (chain|information))\b", re.IGNORECASE)


def is_abstention(text: str) -> bool:
    return not text.strip() or bool(ABSTAIN.search(text))


def judge(chat, question: str, gold: str, pred: str) -> bool:
    """LLM judge with a deterministic guard: an abstention is INCORRECT. (Judges have been observed grading
    'INSUFFICIENT' as correct when the gold answer is absent from the context.)"""
    if is_abstention(pred):
        return False
    return chat(JUDGE_PROMPT.format(question=question, gold=gold, pred=pred), max_tokens=5).upper().startswith(
        "CORRECT")


@dataclass
class WalkConfig:
    max_hops: int = 6
    search: str = "bidir"          # bidir | enum
    answers_per_depth: int = 30    # backward seeds per depth (dst_m)
    rerank_pool: int = 80          # cosine prefilter size before the answer cross-encoder (saturates beyond)
    alpha: float = 0.7             # rank blend: alpha*cosine_rank + (1-alpha)*ce_rank
    path_rerank_pool: int = 0      # 0 = off; >0 cross-encodes a terminal-diverse pool of paths
    top_k: int = 8                 # chains handed to the reader
    max_paths: int = 6000


@dataclass
class WalkResult:
    question: str
    answer: str
    keywords: dict
    anchors: list[str]
    paths: list[list[str]]
    candidates: int
    timings: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


class GraphWalk:
    def __init__(self, gs, chat, embed, reranker=None, cfg: WalkConfig | None = None):
        self.gs, self.chat, self.embed, self.reranker = gs, chat, embed, reranker
        self.cfg = cfg or WalkConfig()

    def retrieve(self, question: str, hops: int | None = None) -> WalkResult:
        cfg, gs, t = self.cfg, self.gs, {}
        t0 = time.time()
        kw = self.chat.keywords(question)
        qvec = self.embed([question])[0]
        themes = self.embed(kw["high_level"]) if kw["high_level"] else np.zeros((0, len(qvec)), np.float32)
        t["keywords"] = time.time() - t0

        t0 = time.time()
        anc = [a["node"] for a in find_anchors(gs, kw["low_level"], self.embed)]
        t["anchors"] = time.time() - t0
        depths = [hops] if hops else list(range(2, cfg.max_hops + 1))
        paths: list[list[str]] = []
        trace: dict = {"per_depth": {}}
        t0 = time.time()
        if cfg.search == "bidir" and anc:
            probes = np.vstack([qvec[None, :], themes]) if len(themes) else qvec[None, :]
            cands = answer_candidates(gs, anc, max(depths), probes, m=cfg.answers_per_depth,
                                      pool=cfg.rerank_pool, rerank_text=", ".join(kw["high_level"]) or question,
                                      reranker=self.reranker, alpha=cfg.alpha)
            t["answer_candidates"] = time.time() - t0
            t0 = time.time()
            for k in depths:
                tr: dict = {}
                paths += bidirectional_meet(gs, anc, cands.get(k, []), k, cap=cfg.max_paths, trace=tr)
                trace["per_depth"][k] = tr
        elif anc:
            for k in depths:
                for a in anc:
                    paths += enumerate_from(gs, a, k, cap=cfg.max_paths)
        t["traversal"] = time.time() - t0

        t0 = time.time()
        ranked = rank_paths(gs, paths, qvec, themes)
        if cfg.path_rerank_pool and self.reranker is not None:
            ranked = rerank_paths(gs, question, ranked, self.reranker, pool=cfg.path_rerank_pool)
        t["selection"] = time.time() - t0
        return WalkResult(question, "", kw, anc, ranked[:cfg.top_k], len(paths), t, trace)

    def answer(self, question: str, hops: int | None = None) -> WalkResult:
        r = self.retrieve(question, hops)
        if not r.paths:
            r.answer = "INSUFFICIENT"
            return r
        chains = "\n\n".join(f"Chain {i}:\n{verbalize(self.gs, p)}" for i, p in enumerate(r.paths, 1))
        t0 = time.time()
        r.answer = self.chat(READER_PROMPT.format(question=question, chains=chains), max_tokens=64)
        r.timings["reader"] = time.time() - t0
        return r
