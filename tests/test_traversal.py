import random

import networkx as nx
import pytest

from graphwalk_rag.traversal import bidirectional_meet, enumerate_from, reach


class G:
    """Minimal store over a NetworkX graph (the interface GraphStore exposes to traversal)."""

    def __init__(self, g, max_degree=10**9):
        self.g, self.max_degree, self.deg = g, max_degree, dict(g.degree())

    def neighbors(self, n):
        return self.g.neighbors(n)

    def has(self, n):
        return n in self.g

    def is_hub(self, n):
        return self.deg.get(n, 0) > self.max_degree

    def degree(self, n):
        return self.deg.get(n, 0)


def brute(g, src, dsts, k):
    out = set()
    for d in dsts:
        if d == src:
            continue
        for p in nx.all_simple_paths(g, src, d, cutoff=k):
            if len(p) - 1 == k:
                out.add(tuple(p))
    return out


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_meet_equals_brute_force(seed, k):
    rnd = random.Random(seed)
    g = nx.gnm_random_graph(28, 60, seed=seed)
    gs = G(g)
    src = rnd.randrange(28)
    dsts = rnd.sample(range(28), 4)
    want = brute(g, src, dsts, k)
    got_pruned = {tuple(p) for p in bidirectional_meet(gs, src, dsts, k)}
    got_blind = {tuple(p) for p in bidirectional_meet(gs, src, dsts, k, prune=False)}
    assert got_pruned == want
    assert got_blind == want


def test_pruning_does_less_work():
    g = nx.gnm_random_graph(200, 600, seed=3)
    gs = G(g)
    tp, tb = {}, {}
    a = bidirectional_meet(gs, 0, [150, 151], 6, trace=tp)
    b = bidirectional_meet(gs, 0, [150, 151], 6, prune=False, trace=tb)
    assert {tuple(p) for p in a} == {tuple(p) for p in b}
    assert tp["forward_halves"] < tb["forward_halves"]
    assert tp["backward_enumerated"] < tb["backward_enumerated"]


def test_hubs_are_never_traversed():
    g = nx.path_graph(["a", "hub", "c"])
    g.add_edges_from(("hub", f"x{i}") for i in range(5))
    gs = G(g, max_degree=3)
    assert bidirectional_meet(gs, "a", ["c"], 2) == []
    assert reach(gs, ["a"], 3) == {"a": 0}


def test_reach_distances():
    gs = G(nx.path_graph(5))
    assert reach(gs, [0], 2) == {0: 0, 1: 1, 2: 2}
    assert reach(gs, [0, 4], 1) == {0: 0, 1: 1, 4: 0, 3: 1}


def test_enumerate_is_capped():
    gs = G(nx.complete_graph(9))
    assert len(enumerate_from(gs, 0, 4, cap=50)) == 50
