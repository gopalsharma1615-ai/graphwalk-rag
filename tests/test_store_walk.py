import numpy as np
from conftest import vec

from graphwalk_rag.build import DocChunks, chunk_document, estimate
from graphwalk_rag.paths import coverage, rank_paths, verbalize
from graphwalk_rag.seeding import answer_candidates, string_seeds
from graphwalk_rag.store import GraphStore, parse_header
from graphwalk_rag.walk import GraphWalk, WalkConfig, is_abstention, judge


def test_parse_header_variants():
    assert parse_header("[Source: a.pdf | Page 3-5]\nx") == {"doc": "a.pdf", "pages": [3, 4, 5], "card": False}
    assert parse_header("[Source:\n a.pdf |\n Page 2]") == {"doc": "a.pdf", "pages": [2], "card": False}
    assert parse_header("[Source: a.pdf | Document card]")["card"] is True
    assert parse_header("no header") == {}


def test_store_loads_lightrag_workdir(film_graph):
    gs = GraphStore(film_graph, max_degree=5)
    assert gs.is_hub("United States") and not gs.is_hub("Night Harbor")
    assert gs.entity_vec("Lena Ortiz") is not None
    assert gs.relation_vec("Lena Ortiz", "Night Harbor") is not None          # either direction
    assert gs.resolve("lena ortiz") == "Lena Ortiz"
    assert gs.chunks["c2"]["pages"] == [2, 3]
    assert gs.chunks["c3"]["doc"] == "films.pdf" and gs.chunks["c3"]["pages"] == [3]   # inherited
    assert gs.chunks["c0"]["card"] is True
    assert gs.node_docs("Gala Night") == {"awards.pdf"}
    assert string_seeds(gs, "Which city was the director of a film with Ada Brooks born in?") == ["Ada Brooks"]


def test_answer_candidates_by_depth(film_graph):
    gs = GraphStore(film_graph, max_degree=5)
    probes = np.stack([vec("born in the city of"), vec("city")])
    c = answer_candidates(gs, ["Ada Brooks"], 3, probes, m=5)
    assert set(c) == {2, 3}
    assert "Valparaiso" in c[3] and "Valparaiso" not in c[2]     # 3 hops away
    assert all(n != "United States" for lvl in c.values() for n in lvl)


def test_coverage_prefers_condition_complete_path(film_graph):
    gs = GraphStore(film_graph, max_degree=5)
    themes = np.stack([vec("directed by"), vec("born in the city")])
    gold = ["Ada Brooks", "Night Harbor", "Lena Ortiz", "Valparaiso"]
    other = ["Ada Brooks", "Drama Award", "Gala Night"]
    assert coverage(gs, gold, themes, tau=0.5) > coverage(gs, other, themes, tau=0.5)
    assert rank_paths(gs, [other, gold], vec("director birthplace city"), themes)[0] == gold
    assert "directed by Lena Ortiz" in verbalize(gs, gold) and "films.pdf" in verbalize(gs, gold)


class FakeChat:
    def __init__(self):
        self.prompts = []

    def keywords(self, q):
        return {"low_level": ["Ada Brooks"], "high_level": ["directed by", "born in the city"]}

    def __call__(self, prompt, **kw):
        self.prompts.append(prompt)
        if prompt.startswith("Grade"):
            return "CORRECT"
        return "Valparaiso" if "Valparaiso" in prompt else "INSUFFICIENT"


def test_graphwalk_end_to_end(film_graph):
    gs = GraphStore(film_graph, max_degree=5)
    chat = FakeChat()
    gw = GraphWalk(gs, chat, lambda texts: np.stack([vec(t) for t in texts]), None, WalkConfig(max_hops=3))
    r = gw.answer("In which city was the director of the film starring Ada Brooks born?")
    assert r.anchors[0] == "Ada Brooks"
    assert ["Ada Brooks", "Night Harbor", "Lena Ortiz", "Valparaiso"] in r.paths
    assert r.answer == "Valparaiso"
    assert all("United States" not in p for p in r.paths)


def test_judge_guards_abstention():
    chat = FakeChat()
    assert is_abstention("INSUFFICIENT") and is_abstention("I cannot determine this from the evidence")
    assert judge(chat, "q", "Paris", "INSUFFICIENT") is False      # never reaches the LLM judge
    assert judge(chat, "q", "Paris", "Paris, France") is True


def test_chunk_document_headers_and_card(tmp_path):
    p = tmp_path / "policy.txt"
    p.write_text("Scope\n\nThis policy applies to all lenders.\n\n" + "More text here. " * 200)
    d = chunk_document(p, max_tokens=120, count=lambda s: len(s.split()))
    assert d.chunks[0].startswith("[Source: policy.txt | Document card]")
    assert all(c.startswith("[Source: policy.txt | Page 1]") for c in d.chunks[1:])
    assert len(d.chunks) > 3


def test_estimate_uses_unit_costs():
    docs = [DocChunks("a.pdf", 500, ["x"] * 1000)]
    e = estimate(docs, "gpt-4o-mini", count=lambda s: 1)
    assert e["llm_calls"] == 2730
    assert abs(e["estimated_usd"] - (1000 * 7060 * 0.15 + 1000 * 1144 * 0.60) / 1e6) < 0.01
