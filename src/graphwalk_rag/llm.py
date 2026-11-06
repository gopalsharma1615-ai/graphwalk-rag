"""Model access. Everything is OpenAI-compatible, so the same code runs against OpenAI, vLLM, Ollama or any
compatible server: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (and EMBED_* for embeddings). Rerankers are local.

Embeddings MUST come from the same model that embedded the graph - an entity matrix built with
text-embedding-3-small cannot be queried with bge vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

KEYWORD_PROMPT = """Extract search keywords from the question for a knowledge-graph retriever.

- "low_level": the specific named things the question mentions (people, works, organisations, places, terms).
  These are looked up as graph nodes. Copy names exactly as written.
- "high_level": the relations and the answer type the question asks about ("directed by", "birthplace",
  "capital city"). These guide which edges to follow. Do not repeat the low-level names here.

Return JSON only: {{"low_level": [...], "high_level": [...]}}

Question: {question}"""


class _Cache:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.data: dict = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text())

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.flush()

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))


class Chat:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 cache: str | None = None, extra_body: dict | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL") or None,
                             api_key=api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        self.cache = _Cache(cache)
        # e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Qwen3 on vLLM
        self.extra_body = extra_body or (json.loads(os.environ["LLM_EXTRA_BODY"])
                                         if os.environ.get("LLM_EXTRA_BODY") else None)

    def __call__(self, prompt: str, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0) -> str:
        key = hashlib.sha1(f"{self.model}|{system}|{prompt}|{max_tokens}|{temperature}".encode()).hexdigest()
        if (hit := self.cache.get(key)) is not None:
            return hit
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                temperature=temperature, extra_body=self.extra_body)
        out = (r.choices[0].message.content or "").strip()
        out = re.sub(r"<think>.*?</think>\s*", "", out, flags=re.DOTALL)
        self.cache.put(key, out)
        return out

    def keywords(self, question: str) -> dict:
        raw = self(KEYWORD_PROMPT.format(question=question), max_tokens=200)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        return {"low_level": [str(x) for x in d.get("low_level", [])][:6],
                "high_level": [str(x) for x in d.get("high_level", [])][:6]}


class Embed:
    """OpenAI-compatible embeddings (default) or a local sentence-transformers model (`local:<name>`)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 cache: str | None = None, query_prefix: str = ""):
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.query_prefix = query_prefix
        self.cache = _Cache(cache)
        if self.model.startswith("local:"):
            from sentence_transformers import SentenceTransformer

            self.local = SentenceTransformer(self.model[6:], device="cpu")
            self.client = None
        else:
            from openai import OpenAI

            self.local = None
            self.client = OpenAI(base_url=base_url or os.environ.get("EMBED_BASE_URL") or None,
                                 api_key=api_key or os.environ.get("EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    def __call__(self, texts: list[str]) -> np.ndarray:
        texts = [self.query_prefix + t for t in texts]
        keys = [hashlib.sha1(f"{self.model}|{t}".encode()).hexdigest() for t in texts]
        missing = [t for t, k in zip(texts, keys) if self.cache.get(k) is None]
        if missing:
            if self.local is not None:
                vecs = self.local.encode(missing, normalize_embeddings=True).tolist()
            else:
                r = self.client.embeddings.create(model=self.model, input=missing)
                vecs = [d.embedding for d in r.data]
            for t, v in zip(missing, vecs):
                self.cache.data[hashlib.sha1(f"{self.model}|{t}".encode()).hexdigest()] = v
            self.cache.flush()
        return np.asarray([self.cache.get(k) for k in keys], np.float32)


class LocalReranker:
    """Local cross-encoder (sentence-transformers). Default: ms-marco MiniLM (fast); bge-reranker-v2-m3 is more
    precise and ~8x slower on CPU."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 256):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, max_length=max_length, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16,
                                                     show_progress_bar=False)]
