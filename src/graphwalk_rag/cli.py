"""graphwalk-rag command line.

  graphwalk-rag build docs/ --workdir graph/ --dry-run           price the build first (no API calls)
  graphwalk-rag build docs/ --workdir graph/ [--entity-types types.txt]
  graphwalk-rag verify --workdir graph/                           provenance check across all stores
  graphwalk-rag ask "question" --workdir graph/ [--mode walk|guided|mix|naive|local|global|hybrid|router]
  graphwalk-rag export-neo4j --workdir graph/ --label MyGraph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _files(src: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in src:
        p = Path(s)
        out += sorted(x for x in p.rglob("*") if x.suffix.lower() in (".pdf", ".txt", ".md")) if p.is_dir() else [p]
    return out


def cmd_build(a) -> int:
    from .build import run_build

    types = None
    if a.entity_types:
        t = Path(a.entity_types)
        types = [x.strip() for x in (t.read_text().splitlines() if t.exists() else a.entity_types.split(","))
                 if x.strip()]
    res = run_build(_files(a.sources), a.workdir, a.model, a.embed_model, a.embed_dim, a.dry_run, types,
                    a.chunk_tokens, a.llm_async, a.llm_timeout)
    print(json.dumps(res, indent=2))
    return 0


def cmd_verify(a) -> int:
    from .build import verify_provenance

    rep = verify_provenance(a.workdir)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


def _rag(a):
    import asyncio

    from lightrag import LightRAG

    from .build import _llm_funcs

    complete, embed = _llm_funcs(a.model, a.embed_model, a.embed_dim)
    rag = LightRAG(working_dir=a.workdir, llm_model_func=complete, llm_model_name=a.model, embedding_func=embed)
    asyncio.get_event_loop().run_until_complete(rag.initialize_storages())
    return rag


def cmd_ask(a) -> int:
    from .llm import Chat, Embed, LocalReranker

    chat = Chat(a.model)
    if a.mode in ("walk", "guided"):
        from .store import GraphStore

        gs = GraphStore(a.workdir)
        embed = Embed(a.embed_model)
        rr = LocalReranker(a.reranker)
        if a.mode == "walk":
            from .paths import verbalize
            from .walk import GraphWalk, WalkConfig

            r = GraphWalk(gs, chat, embed, rr, WalkConfig(max_hops=a.max_hops)).answer(a.question)
            print(f"answer: {r.answer}\nanchors: {r.anchors}\ncandidate paths: {r.candidates}\n")
            for i, p in enumerate(r.paths[:3], 1):
                print(f"chain {i}:\n{verbalize(gs, p)}\n")
        else:
            from .guided import GuidedWalk

            r = GuidedWalk(gs, chat, embed, rr).walk(a.question)
            print(json.dumps({k: r[k] for k in ("routed_docs", "seeds", "evidence_pages")}, indent=2, default=str))
        return 0
    from .router import LightRAGContext, single_first, single_pass

    if a.mode == "router":
        print(json.dumps(single_first(chat, LightRAGContext(_rag(a), "mix"), a.question), indent=2))
    else:
        print(single_pass(chat, LightRAGContext(_rag(a), a.mode), a.question))
    return 0


def cmd_export(a) -> int:
    from .neo4j_store import export
    from .store import GraphStore

    print(json.dumps(export(GraphStore(a.workdir), a.label)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="graphwalk-rag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--workdir", required=True)
        s.add_argument("--model", default="gpt-4o-mini", help="LLM (OpenAI-compatible; see LLM_BASE_URL)")
        s.add_argument("--embed-model", default="text-embedding-3-small", help="or local:<sentence-transformer>")
        s.add_argument("--embed-dim", type=int, default=1536)

    b = sub.add_parser("build")
    b.add_argument("sources", nargs="+")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--entity-types", help="file with one type per line, or a comma list")
    b.add_argument("--chunk-tokens", type=int, default=1000)
    b.add_argument("--llm-async", type=int, default=8, help="parallel LLM calls; match vLLM --max-num-seqs")
    b.add_argument("--llm-timeout", type=int, default=240, help="seconds per LLM call (LightRAG worker limit is 2x)")
    common(b)
    b.set_defaults(fn=cmd_build)

    v = sub.add_parser("verify")
    v.add_argument("--workdir", required=True)
    v.set_defaults(fn=cmd_verify)

    q = sub.add_parser("ask")
    q.add_argument("question")
    q.add_argument("--mode", default="walk",
                   choices=["walk", "guided", "router", "mix", "naive", "local", "global", "hybrid"])
    q.add_argument("--max-hops", type=int, default=6)
    q.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    common(q)
    q.set_defaults(fn=cmd_ask)

    e = sub.add_parser("export-neo4j")
    e.add_argument("--workdir", required=True)
    e.add_argument("--label", default="Entity")
    e.set_defaults(fn=cmd_export)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
