"""Offline structural evaluation - no API calls, no embeddings.

Given a multi-hop set with gold paths (gold anchor = first node, gold answer = last node, hops = path length),
measure how each search strategy behaves when the endpoints are known:

* recall:      is the gold path among the candidates?
* candidates:  how many paths the reader/selector must choose among (the distractor load)

This isolates the traversal from seeding and answer-candidate recall, and is what shows why meet-in-the-middle
dissolves the selection problem: blind enumeration drowns in its own fan-out and hits its cap before reaching deep
gold paths, while bidir only ever builds anchor->answer paths.

    python eval/offline_paths.py --workdir graph/ --questions hop_eval.json --cap 4000
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from graphwalk_rag.store import GraphStore
from graphwalk_rag.traversal import bidirectional_meet, enumerate_from


def gold_nodes(q: dict) -> list[str]:
    g = q.get("gold_nodes")
    return ast.literal_eval(g) if isinstance(g, str) else list(g)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--cap", type=int, default=4000, help="path cap for enum (and bidir)")
    ap.add_argument("--max-degree", type=int, default=150)
    ap.add_argument("--out")
    a = ap.parse_args()

    gs = GraphStore(a.workdir, max_degree=a.max_degree)
    qs = json.loads(Path(a.questions).read_text())
    rows = []
    for q in qs:
        nodes = gold_nodes(q)
        if len(nodes) < 2 or not all(gs.has(n) for n in nodes):
            continue
        src, dst, k = nodes[0], nodes[-1], len(nodes) - 1
        t0 = time.time()
        enum = [p for p in enumerate_from(gs, src, k, cap=a.cap)]
        t_enum = time.time() - t0
        enum_hit = any(p == nodes for p in enum)
        enum_to_dst = [p for p in enum if p[-1] == dst]
        t0 = time.time()
        bid = bidirectional_meet(gs, src, [dst], k, cap=a.cap)
        t_bid = time.time() - t0
        rows.append({"id": q.get("example_id"), "hops": k, "enum_candidates": len(enum), "enum_hit": enum_hit,
                     "enum_to_answer": len(enum_to_dst), "bidir_candidates": len(bid),
                     "bidir_hit": any(p == nodes for p in bid), "enum_s": round(t_enum, 3),
                     "bidir_s": round(t_bid, 3)})

    def agg(rs):
        return {"n": len(rs),
                "enum_recall": round(100 * statistics.mean(r["enum_hit"] for r in rs), 1),
                "bidir_recall": round(100 * statistics.mean(r["bidir_hit"] for r in rs), 1),
                "enum_mean_candidates": round(statistics.mean(r["enum_candidates"] for r in rs), 1),
                "bidir_mean_candidates": round(statistics.mean(r["bidir_candidates"] for r in rs), 1),
                "enum_mean_s": round(statistics.mean(r["enum_s"] for r in rs), 3),
                "bidir_mean_s": round(statistics.mean(r["bidir_s"] for r in rs), 3)}

    by_hop = defaultdict(list)
    for r in rows:
        by_hop[r["hops"]].append(r)
    report = {"overall": agg(rows), "by_hops": {h: agg(rs) for h, rs in sorted(by_hop.items())},
              "cap": a.cap, "max_degree": a.max_degree}
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
