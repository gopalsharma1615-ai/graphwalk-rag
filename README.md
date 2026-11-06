# graphwalk-rag

[![CI](https://github.com/gopalsharma1615-ai/graphwalk-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/gopalsharma1615-ai/graphwalk-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

**Graph RAG that walks the graph.** You build a knowledge graph that keeps its provenance, store it in NetworkX
or Neo4j, and answer multi-hop questions by following evidence *paths* instead of retrieving a bag of chunks.

> **Thesis.** Multi-hop failures are a *reachability* problem. The edge that answers the question exists, but it
> sits several hops from the question and is not textually similar to it, so single-shot retrieval never
> reaches it. A purpose-built walker fixes the root cause.

| | what | when to use it |
|---|---|---|
| **Build** | LightRAG extraction plus provenance headers, document cards, ontology guidance, resumable per-file inserts, a dry-run cost estimate, and a provenance check | any corpus |
| **Store** | GraphML / NetworkX, nano-vectordb matrices, and a Neo4j exporter with a drop-in `Neo4jStore` | NetworkX for research, Neo4j for serving |
| **GraphWalk** | bidirectional **meet-in-the-middle** from anchors to answer candidates | entity-centric multi-hop QA (3–8 hops) |
| **Guided walk** | document-routed, question-filtered beam walk with page-level evidence | long structured corpora: regulations, policies, contracts |
| **Single-first router** | answer single-pass, and decompose only on abstention | default policy in front of any retriever |

## How GraphWalk works

```
question ─► LLM keywords: low-level (named things) → ANCHORS · high-level (relations, answer type) → THEMES
         ─► answer candidates = anchors' hub-guarded k-hop REACHABLE set, ranked by
              cosine(entity, question+themes) ⊕ local cross-encoder on the JOINED THEMES (rank-space blend)
         ─► bidirectional meet: forward ⌈k/2⌉ from anchors ⋈ backward ⌊k/2⌋ from answer candidates,
              both halves distance-pruned toward each other (provably the same result, far less work)
         ─► rank paths by (theme coverage, relation relevance) ─► top-K chains ─► reader LLM ─► answer
```

The design choices, each measured:

- **Meet in the middle, don't enumerate.** Blind enumeration builds every length-*k* walk and then has to find
  the needle among thousands. Meet-in-the-middle builds only paths that connect an anchor to a plausible answer.
- **Seed answers from the reachable set, not the whole graph.** A question's embedding is dominated by its
  anchor, so global "find the answer" retrieval returns topic entities (about 20% answer recall). The anchor's
  reachable set contains the gold answer 99% of the time.
- **Rerank answer candidates on the joined high-level keywords, not the question.** The full question again
  pulls in the anchor's neighbours. On one query the gold answer went from rank 16/238 to rank 1/238.
- **Blend cosine and cross-encoder in rank space.** Letting the cross-encoder overwrite the cosine order demoted
  answers that cosine had already found. A bigger rerank pool *hurts*, because the cross-encoder saturates.
- **Give every distance shell its own slots.** A ball of radius *k* is dominated by its outer shell, so a
  depth-blind top-*m* left short meets with no candidates at all.
- **Hub guard everywhere.** Never traverse through nodes of degree > 150. Generic connectors are the #1 source
  of distractor paths.
- **Let the reader choose among K chains.** The reader is a better final-hop judge than cosine; top-K reading
  added **+12 points** (28% → 40%).

## Quickstart

```bash
pip install "graphwalk-rag[all] @ git+https://github.com/gopalsharma1615-ai/graphwalk-rag"
export OPENAI_API_KEY=...            # or LLM_BASE_URL=http://your-vllm:8000/v1 LLM_MODEL=Qwen/Qwen3-14B

graphwalk-rag build docs/ --workdir graph/ --dry-run     # price it first: calls, tokens, $ (no API calls)
graphwalk-rag build docs/ --workdir graph/ --entity-types "Regulation,Section,Institution,Requirement"
graphwalk-rag verify --workdir graph/                    # provenance intact in all 6 stores?

graphwalk-rag ask "Which city was the director of the film starring X born in?" --workdir graph/
graphwalk-rag ask "What capital must a deposit-taking lender hold?" --workdir graph/ --mode guided
graphwalk-rag ask "..." --workdir graph/ --mode router   # LightRAG mix, decompose only on abstention
graphwalk-rag export-neo4j --workdir graph/ --label MyGraph     # NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
```

```python
from graphwalk_rag import GraphStore, GraphWalk, WalkConfig
from graphwalk_rag.llm import Chat, Embed, LocalReranker

gs = GraphStore("graph/")                          # or graphwalk_rag.neo4j_store.Neo4jStore("graph/", label=...)
walk = GraphWalk(gs, Chat("gpt-4o-mini"), Embed("text-embedding-3-small"), LocalReranker(),
                 WalkConfig(max_hops=6, answers_per_depth=30, rerank_pool=80))
r = walk.answer("...")
print(r.answer, r.anchors, r.candidates, r.trace["per_depth"])
```

Everything is OpenAI-compatible (OpenAI, vLLM, Ollama). Rerankers are local cross-encoders, and no API key is
needed for reranking. Queries must be embedded with the same model that embedded the graph.

## Results

### Traversal, offline (this code, reproducible without any API)

Clean multi-hop set (70 questions mined from real paths of a 52,474-node / 60,475-edge graph built over a
10,000-document HotpotQA corpus). The gold endpoints are given, so this isolates the search itself. The output
is in [`results/offline_paths_clean70.json`](results/offline_paths_clean70.json) and is reproduced by
`python eval/offline_paths.py --cap 4000`.

| hops | n | enum recall | **bidir recall** | enum candidates | **bidir candidates** |
|---|---|---|---|---|---|
| 3 | 13 | 100 | 100 | 79 | 1.5 |
| 4 | 32 | 100 | 100 | 382 | 2.8 |
| 5 | 12 | 91.7 | 100 | 1,775 | 4.9 |
| 6 | 8 | 62.5 | 100 | 3,001 | 7.4 |
| 7 | 4 | 25.0 | 100 | 3,325 | 22.2 |
| **all** | 70 | 88.6 | **100** | 1,084 | **5.4 (≈200× fewer)** |

Blind enumeration drowns in its own fan-out and hits its cap before it reaches deep gold paths.
Meet-in-the-middle never builds the fan-out at all. A randomized test checks that the pruned meet returns
exactly the brute-force path set (`tests/test_traversal.py`).

### End-to-end multi-hop QA (reference runs)

> **Provenance.** These end-to-end numbers come from the reference runs of the original implementation of this
> design (reader gpt-4o-mini, judge gpt-4o with abstention scored as incorrect, graph and query embeddings
> text-embedding-3-small, local MiniLM cross-encoder). This repository is a clean re-implementation, and
> `eval/` contains the harness to re-run them.

Clean-70 accuracy by search configuration (single-pass LightRAG `mix` on the same questions is about 47%):

| configuration | accuracy | avg candidates | what it fixed |
|---|---|---|---|
| `enum` (blind) | 47% | ~1,500 | (selection breaks, deep hops explode) |
| `typed` (relation-similarity gate τ) | 47% | ~1,300 | nothing: τ prunes about 8%, a no-op |
| `bidir`, answers from global dense search | ~47% | ~200 | selection solved, answer recall broke |
| `bidir`, answers from the reachable set | 47% | ~200 | candidates connectable, 99% answer ceiling |
| **`bidir` + reachable + cross-encoder on themes** | **53%** | ~420 | answer recall (+6) |

By hop count (best configuration): 3 hops 54% · **4 hops 66%** · 5 hops 25% · 6 hops 38% · **7 hops 75%**.

Component metrics:
- Anchor recall: 88%.
- Gold path structurally reachable: 87–95% (3–5 hops).
- Gold answer in the reachable set: 99%.
- Gold path selected when it is among the candidates: 72–84% for `bidir`, vs 39% for `enum`.

The ceiling is now answer-candidate recall in deep, huge reachable pools of 1.4k–26k nodes.

### Retrieval vs naive RAG, and the single-first router (reference runs, HotpotQA)

1,021 HotpotQA questions over the 10,000-document corpus, gold-document positions verified per query:

| both gold documents retrieved within top-k | 5 | 20 | 50 | 200 |
|---|---|---|---|---|
| naive vector RAG | 59% | 81% | 88% | 95% |
| LightRAG `mix` (graph + chunks) | 58% | 82% | 91% | 98% |

The graph's edge shows up at deep cutoffs, which is where multi-hop evidence lives. Answer accuracy on 952
questions (strict judge, context-only answers):

| policy | accuracy | decomposition used on |
|---|---|---|
| single-pass `mix` | 70.2% | 0% |
| always decompose (self-ask) | 62.8% | 100% |
| **single-first router** (decompose only on abstention) | **76.9%** | **18%** |
| oracle (best of both, per question) | 81.5% | n/a |

Decomposition is a **rescue, not a default**. On the 62 hardest questions it lifts accuracy from 17.7% to 58.1%,
but applied everywhere it damages the many questions that single-pass already answers.

### Long regulatory corpus (reference runs, private corpus)

371 regulatory documents (17,326 pages) and 101 questions with gold pages. The corpus is private, so these
numbers are reported for scale and method, not for reproduction.

| | value |
|---|---|
| graph | 46,508 entities / 109,572 relations, built with gpt-4o-mini |
| build cost | **$35.1 total**: 2.73 LLM calls/chunk, $0.0018/chunk, **$2.03 per 1,000 pages** |
| doc recall @5 / @10 / @20 | 0.804 / 0.865 / 0.927 |
| evidence-page recall @20 | 0.85 |
| fully on-prem rebuild (Qwen3 on vLLM + bge-m3) vs cloud build, guided walk | doc hit@20 **0.911 vs 0.861**, page hit@30 0.634 vs 0.604 |
| latency | 8.7 s retrieval + 4.5 s answer |

The guided walk exists because an unguided walk drifted through hub entities. Only 19% of top-20 path edges were
about the question: 43% on the first hop and about 10% from the second hop on.

The same graph code moved from NetworkX to Neo4j through the `Neo4jStore` adapter and reproduced 25/25 query
results exactly. The full graph loads in 7 s.

### The two graphs, side by side

| | HotpotQA graph | regulatory graph |
|---|---|---|
| mean degree | 2.3 | 4.71 |
| max degree | 211 | 5,830 (the regulator itself) |
| nodes above the hub guard (deg > 150) | 2 | 88 |
| isolated nodes | 3.1% | 11.0% |

Hub structure is why the regulatory corpus needs the guided walk and HotpotQA does not.

## Building a trustworthy multi-hop eval set

Questions mined from graph paths are only as good as their guards. A full re-validation of 100 mined questions
found about 30 defective:
- 16 had wording defects: interrogative/answer-type mismatch, dataset strings leaking into the question,
  degenerate common-noun golds, or the answer stated in the question.
- About 14 had non-unique terminals: a final one-to-many relation with no discriminator, such as "a case decided
  by the court" with 28+ valid answers.

A generator therefore needs three guards:
1. The question's head noun must match the terminal's type.
2. The final relation must be functional, or discriminated.
3. No template or dataset strings may appear in the text.

All accuracy numbers above are on the clean 70.

## Development

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[all,dev]"
pytest -q          # 52 tests: randomized brute-force equivalence, loader, seeding, walk, build, judge guard
ruff check .
```

MIT licence. Built by [Gopal Sharma](https://github.com/gopalsharma1615-ai). Graph extraction uses
[LightRAG](https://github.com/HKUDS/LightRAG) (MIT).
