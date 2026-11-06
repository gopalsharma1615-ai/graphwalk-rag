"""Single-first router: decomposition is a rescue, not a default.

Self-ask decomposition (resolve the bridge entity, then re-query with it named) rescues bridge questions whose
answer edge single-shot retrieval cannot reach. But applied to every question it damages the many that
single-pass already answers. Measured on 952 HotpotQA questions over one LightRAG graph (strict context-only
judge): single-pass 70.2%, always-decompose 62.8%, single-first router 76.9% - decomposition fired on only 18%
of questions, so its ~3x cost is contained.

Policy: answer single-pass; only when the answer abstains, decompose and retry.
"""
from __future__ import annotations

import asyncio
import re

from .walk import is_abstention

DECOMPOSE_PROMPT = """Split the question into two steps for a retriever. Step 1 asks for the bridge entity that
the question depends on; step 2 is the original question rewritten with a placeholder [BRIDGE] for it.
Return JSON only: {{"step1": "...", "step2": "..."}}

Question: {question}"""

ANSWER_PROMPT = """Answer using ONLY the context. If it does not contain the answer, reply exactly INSUFFICIENT.
Answer with a short phrase.

Context:
{context}

Question: {question}
Answer:"""


class LightRAGContext:
    """Context provider over a LightRAG instance (any mode), retrieval only - no LightRAG answer generation."""

    def __init__(self, rag, mode: str = "mix", top_k: int = 20, chunk_top_k: int = 30):
        self.rag, self.mode, self.top_k, self.chunk_top_k = rag, mode, top_k, chunk_top_k

    def __call__(self, question: str) -> str:
        from lightrag import QueryParam

        p = QueryParam(mode=self.mode, only_need_context=True, top_k=self.top_k, chunk_top_k=self.chunk_top_k)
        return asyncio.get_event_loop().run_until_complete(self.rag.aquery(question, param=p)) or ""


def single_pass(chat, context_fn, question: str, max_ctx_chars: int = 60000) -> str:
    return chat(ANSWER_PROMPT.format(context=context_fn(question)[:max_ctx_chars], question=question), max_tokens=64)


def decompose(chat, context_fn, question: str) -> tuple[str, dict]:
    import json

    raw = chat(DECOMPOSE_PROMPT.format(question=question), max_tokens=200)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        steps = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        steps = {}
    s1, s2 = steps.get("step1"), steps.get("step2")
    if not s1 or not s2:
        return "INSUFFICIENT", {"steps": steps}
    bridge = single_pass(chat, context_fn, s1)
    if is_abstention(bridge):
        return "INSUFFICIENT", {"steps": steps, "bridge": bridge}
    q2 = s2.replace("[BRIDGE]", bridge)
    return single_pass(chat, context_fn, q2), {"steps": steps, "bridge": bridge, "hop2": q2}


def single_first(chat, context_fn, question: str) -> dict:
    first = single_pass(chat, context_fn, question)
    if not is_abstention(first):
        return {"answer": first, "route": "single"}
    ans, info = decompose(chat, context_fn, question)
    return {"answer": ans, "route": "decompose", "first": first, **info}
