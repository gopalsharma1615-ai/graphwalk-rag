"""Build a LightRAG knowledge graph that keeps its provenance.

What this layer adds on top of LightRAG (lightrag-hku, MIT):

* Provenance headers. Every chunk starts with `[Source: <file> | Page N]`. That header is what lets a walk cite
  pages, and it also keeps identical boilerplate paragraphs from different documents distinct (LightRAG
  deduplicates by content hash). A provenance loss is silent and expensive: if chunks end up with
  file_path='unknown_source', corpus filters switch off and every page metric reads zero. `verify_provenance`
  checks all stores after a build.
* One "document card" chunk per file (title + opening text), used by the guided walk to route questions.
* One LightRAG document per file, pre-split at our chunk boundaries, so the stores are rewritten once per file
  rather than once per chunk, and a resumed build skips finished files.
* Ontology guidance: domain entity types are passed to the extraction prompt.
* A dry run that prices the build BEFORE any paid call, from measured unit costs.

Measured on a 371-document / 17,326-page regulatory build with gpt-4o-mini (extraction + one gleaning pass +
description summaries): 2.73 LLM calls, ~7,060 input and ~1,144 output tokens per chunk -> ~$0.0018 per chunk,
~$2.03 per 1,000 pages, $35.1 in total.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SEP = "\n<<<CHUNK>>>\n"

# per-chunk unit costs measured on a full build (see module doc); override with your own measurements
UNIT = {"calls": 2.73, "input_tokens": 7060, "output_tokens": 1144}
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-5-mini": (0.25, 2.00),
          "gpt-4o": (2.50, 10.00)}   # USD per 1M input / output tokens


@dataclass
class DocChunks:
    name: str
    pages: int
    chunks: list[str]


def _tok():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s))


def read_pages(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(path) as d:
            return [p.get_text("text") for p in d]
    return [path.read_text(encoding="utf-8", errors="ignore")]


def chunk_document(path: Path, max_tokens: int = 1000, count=None) -> DocChunks:
    """Paragraph-boundary chunks <= max_tokens, each prefixed with its provenance header, plus a document card."""
    count = count or _tok()
    pages = read_pages(path)
    name = path.name
    paras: list[tuple[int, str]] = []
    for pno, text in enumerate(pages, 1):
        for p in re.split(r"\n\s*\n", text):
            p = " ".join(p.split())
            if p:
                paras.append((pno, p))
    chunks: list[str] = []
    cur: list[tuple[int, str]] = []

    def flush():
        if not cur:
            return
        lo, hi = cur[0][0], cur[-1][0]
        where = f"Page {lo}" if lo == hi else f"Page {lo}-{hi}"
        chunks.append(f"[Source: {name} | {where}]\n" + "\n\n".join(t for _, t in cur))
        cur.clear()

    def pieces(p: str) -> list[str]:
        """A paragraph over budget is split at sentences, then at words."""
        if count(p) <= max_tokens:
            return [p]
        out, buf = [], ""
        for unit in re.split(r"(?<=[.!?])\s+", p):
            for w in (unit.split() if count(unit) > max_tokens else [unit]):
                cand = f"{buf} {w}".strip()
                if buf and count(cand) > max_tokens:
                    out.append(buf)
                    buf = w
                else:
                    buf = cand
        return out + ([buf] if buf else [])

    for pno, para in paras:
        for p in pieces(para):
            if cur and count("\n\n".join(t for _, t in cur) + p) > max_tokens:
                flush()
            cur.append((pno, p))
    flush()
    opening = " ".join(t for _, t in paras[:12])[:1500]
    card = f"[Source: {name} | Document card]\nDocument: {path.stem}\n{opening}"
    return DocChunks(name, len(pages), [card] + chunks)


def estimate(docs: list[DocChunks], model: str = "gpt-4o-mini", count=None) -> dict:
    count = count or _tok()
    n_chunks = sum(len(d.chunks) for d in docs)
    src_tokens = sum(count(c) for d in docs for c in d.chunks)
    pin, pout = PRICES.get(model, PRICES["gpt-4o-mini"])
    tin, tout = n_chunks * UNIT["input_tokens"], n_chunks * UNIT["output_tokens"]
    usd = tin / 1e6 * pin + tout / 1e6 * pout
    pages = sum(d.pages for d in docs)
    return {"documents": len(docs), "pages": pages, "chunks": n_chunks, "source_tokens": src_tokens,
            "llm_calls": round(n_chunks * UNIT["calls"]), "llm_input_tokens": tin, "llm_output_tokens": tout,
            "model": model, "estimated_usd": round(usd, 2),
            "usd_per_1000_pages": round(1000 * usd / pages, 2) if pages else None}


def entity_guidance(entity_types: list[str]) -> str:
    """LightRAG (>= 1.5) takes entity-type guidance as one prose block, not a list."""
    lines = "\n".join(f"- {t}" for t in entity_types if t.strip())
    return ("Classify each entity using one of the following types. If no type fits, use `Other`.\n\n" + lines)


def _llm_funcs(model: str, embed_model: str, embed_dim: int):
    """OpenAI-compatible LLM + embedding functions for LightRAG (OpenAI, vLLM, Ollama ...)."""
    import numpy as np
    from lightrag.utils import EmbeddingFunc

    base_url, api_key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY") or os.environ.get(
        "OPENAI_API_KEY")
    extra = json.loads(os.environ["LLM_EXTRA_BODY"]) if os.environ.get("LLM_EXTRA_BODY") else None

    async def complete(prompt, system_prompt=None, history_messages=None, **kw):
        from openai import AsyncOpenAI

        kw.pop("hashing_kv", None)
        kw.pop("keyword_extraction", None)
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(
            history_messages or []) + [{"role": "user", "content": prompt}]
        r = await client.chat.completions.create(model=model, messages=msgs, extra_body=extra,
                                                 **{k: v for k, v in kw.items() if k in ("temperature",
                                                                                       "max_tokens")})
        return re.sub(r"<think>.*?</think>\s*", "", r.choices[0].message.content or "", flags=re.DOTALL)

    if embed_model.startswith("local:"):
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(embed_model[6:], device="cpu")

        async def embed(texts):
            return np.asarray(st.encode(list(texts), normalize_embeddings=True), np.float32)
    else:
        e_base = os.environ.get("EMBED_BASE_URL") or base_url
        e_key = os.environ.get("EMBED_API_KEY") or api_key

        async def embed(texts):
            from openai import AsyncOpenAI

            r = await AsyncOpenAI(base_url=e_base, api_key=e_key).embeddings.create(model=embed_model,
                                                                                    input=list(texts))
            return np.asarray([d.embedding for d in r.data], np.float32)

    return complete, EmbeddingFunc(embedding_dim=embed_dim, max_token_size=8192, func=embed)


async def build(workdir: str, docs: list[DocChunks], model: str, embed_model: str, embed_dim: int,
                entity_types: list[str] | None = None, llm_async: int = 8) -> None:
    from lightrag import LightRAG

    complete, embed = _llm_funcs(model, embed_model, embed_dim)
    addon = {"language": "English"}
    if entity_types:
        addon["entity_types_guidance"] = entity_guidance(entity_types)
    rag = LightRAG(working_dir=workdir, llm_model_func=complete, llm_model_name=model, embedding_func=embed,
                   addon_params=addon, llm_model_max_async=llm_async)
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except ImportError:
        pass
    for i, d in enumerate(docs, 1):
        await rag.ainsert([CHUNK_SEP.join(d.chunks)], split_by_character=CHUNK_SEP,
                          split_by_character_only=True, file_paths=[d.name])
        print(f"[{i}/{len(docs)}] {d.name}: {len(d.chunks)} chunks", flush=True)
    await rag.finalize_storages()


def verify_provenance(workdir: str | Path) -> dict:
    """Check every store for lost provenance. Returns counts; `ok` is False if anything is unknown."""
    from .store import parse_header

    w = Path(workdir)
    report: dict = {}
    kv = json.loads((w / "kv_store_text_chunks.json").read_text())
    headed = sum(bool(parse_header(v.get("content", ""))) for v in kv.values())
    report["chunks"] = len(kv)
    report["chunks_with_header"] = headed
    for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
        p = w / name
        if p.exists():
            rows = json.loads(p.read_text())["data"]
            report[f"{name}:unknown_source"] = sum((r.get("file_path") or "unknown_source") == "unknown_source"
                                                   for r in rows)
    unknown = sum(v for k, v in report.items() if k.endswith("unknown_source"))
    report["ok"] = unknown == 0 and headed >= 0.9 * max(1, len(kv))
    return report


def run_build(paths: list[Path], workdir: str, model: str, embed_model: str, embed_dim: int, dry_run: bool,
              entity_types: list[str] | None, max_tokens: int) -> dict:
    count = _tok()
    docs = [chunk_document(p, max_tokens, count) for p in paths]
    est = estimate(docs, model, count)
    if dry_run:
        return est
    asyncio.run(build(workdir, docs, model, embed_model, embed_dim, entity_types))
    return {"estimate": est, "provenance": verify_provenance(workdir)}
