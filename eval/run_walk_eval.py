"""End-to-end multi-hop QA accuracy for GraphWalk (paid: keyword extraction, reader and judge call an LLM).

    python eval/run_walk_eval.py --workdir graph/ --questions hop_eval_70.json --out results/walk_bidir.json \
        [--search bidir|enum] [--max-hops 8] [--gold-depth]

Questions need `question`, `gold_answer` and optionally `num_hops` (used with --gold-depth to run a single
depth, the oracle setting). Resumable: finished ids are skipped. Put a budget on the key first, e.g. with
openai-budget-guard.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from graphwalk_rag.llm import Chat, Embed, LocalReranker
from graphwalk_rag.store import GraphStore
from graphwalk_rag.walk import GraphWalk, WalkConfig, judge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search", default="bidir", choices=["bidir", "enum"])
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--gold-depth", action="store_true")
    ap.add_argument("--reader", default="gpt-4o-mini")
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--cache", default=".cache")
    a = ap.parse_args()

    out = Path(a.out)
    done = {r["id"]: r for r in json.loads(out.read_text())["rows"]} if out.exists() else {}
    gs = GraphStore(a.workdir)
    reader = Chat(a.reader, cache=f"{a.cache}/chat.json")
    judge_llm = Chat(a.judge, cache=f"{a.cache}/judge.json")
    walk = GraphWalk(gs, reader, Embed(a.embed_model, cache=f"{a.cache}/embed.json"), LocalReranker(a.reranker),
                     WalkConfig(search=a.search, max_hops=a.max_hops))
    qs = json.loads(Path(a.questions).read_text())
    for q in qs:
        qid = q.get("example_id") or q.get("id") or q["question"][:40]
        if qid in done:
            continue
        hops = int(q["num_hops"]) if a.gold_depth and q.get("num_hops") else None
        r = walk.answer(q["question"], hops=hops)
        ok = judge(judge_llm, q["question"], q["gold_answer"], r.answer)
        done[qid] = {"id": qid, "hops": int(q.get("num_hops") or 0), "answer": r.answer, "gold": q["gold_answer"],
                     "correct": ok, "candidates": r.candidates, "anchors": r.anchors, "top_path": r.paths[:1]}
        rows = list(done.values())
        by = defaultdict(list)
        for x in rows:
            by[x["hops"]].append(x["correct"])
        summary = {"n": len(rows), "accuracy": round(100 * statistics.mean(x["correct"] for x in rows), 1),
                   "by_hops": {h: round(100 * statistics.mean(v), 1) for h, v in sorted(by.items())},
                   "mean_candidates": round(statistics.mean(x["candidates"] for x in rows), 1)}
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"{qid}: {'OK ' if ok else 'BAD'} {r.answer[:60]!r} (gold {q['gold_answer']!r}) | "
              f"running {summary['accuracy']}% of {summary['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
