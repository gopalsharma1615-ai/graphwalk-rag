"""Path search over the entity graph.

The thesis: multi-hop failures are a REACHABILITY problem. The edge that answers the question exists, but it sits
several hops from the question and is not textually similar to it, so single-shot retrieval never reaches it.

* `bidirectional_meet` (recommended): grow a forward frontier from the anchor(s) to depth ceil(k/2) and a backward
  frontier from candidate ANSWER nodes to depth floor(k/2); join where they share a midpoint. Only paths that
  connect an anchor to a plausible answer are ever materialised - cost ~ b^(k/2) per side instead of b^k.
  Both frontiers are distance-pruned toward each other with admissible bounds, so the result is identical to the
  blind enumerate-then-join version with far less work.
* `enumerate_from` / `beam_paths`: the blind and greedy baselines it is measured against.

Every search uses the same guards: simple paths only, and no traversal THROUGH a hub (degree > store.max_degree) -
generic connectors ("United States", "film") are the #1 source of distractor paths.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable

INF = 1 << 30


def reach(gs, sources: Iterable[str], k: int) -> dict[str, int]:
    """Hub-guarded multi-source BFS: shortest distance (<= k) from any source. Hubs are never entered, so the
    set is exactly what a guarded walk could connect to."""
    dist = {s: 0 for s in sources if gs.has(s)}
    q = deque(dist)
    while q:
        u = q.popleft()
        if dist[u] >= k:
            continue
        for nb in gs.neighbors(u):
            if nb not in dist and not gs.is_hub(nb):
                dist[nb] = dist[u] + 1
                q.append(nb)
    return dist


def _paths(gs, src: str, k: int, cap: int, allowed: Callable[[str, int], bool] | None = None) -> list[list[str]]:
    """All simple hub-guarded paths of exactly k edges from src. `allowed(node, hops_left)` prunes branches."""
    if not gs.has(src):
        return []
    out: list[list[str]] = []
    path, on = [src], {src}

    def dfs(depth: int) -> None:
        if len(out) >= cap:
            return
        if depth == k:
            out.append(list(path))
            return
        for nb in gs.neighbors(path[-1]):
            if nb in on or gs.is_hub(nb):
                continue
            if allowed is not None and not allowed(nb, k - depth - 1):
                continue
            path.append(nb)
            on.add(nb)
            dfs(depth + 1)
            path.pop()
            on.discard(nb)
            if len(out) >= cap:
                return

    dfs(0)
    return out


def enumerate_from(gs, src: str, k: int, cap: int = 4000) -> list[list[str]]:
    """Blind baseline: every simple length-k walk from src, capped."""
    return _paths(gs, src, k, cap)


def bidirectional_meet(gs, anchors, answers: Iterable[str], k: int, cap: int = 6000,
                       half_cap: int = 20000, prune: bool = True, trace: dict | None = None) -> list[list[str]]:
    """Simple paths of exactly k edges from any anchor to any answer candidate, by meet-in-the-middle.

    prune=True restricts each half to nodes that can still reach the other side within the hops left
    (admissible: a simple path of r edges cannot reach a node whose shortest distance exceeds r), so pruned
    branches provably cannot meet. prune=False is the blind version, kept for A/B equivalence tests.
    """
    many = isinstance(anchors, (list, tuple, set, frozenset))
    srcs = [a for a in (list(anchors) if many else [anchors]) if gs.has(a)]
    dsts = [d for d in answers if gs.has(d)]
    if not srcs or not dsts or k < 1:
        return []
    kf, kb = (k + 1) // 2, k // 2

    if prune:
        back_zone = reach(gs, dsts, kb)                 # nodes within kb of an answer = possible midpoints
        to_zone = reach(gs, list(back_zone), kf)        # distance to that zone, for the forward prune
        fwd_ok = (lambda n, left: to_zone.get(n, INF) <= left)
    else:
        back_zone, fwd_ok = None, None

    fwd: list[list[str]] = []
    seen: set[tuple] = set()
    for s in srcs:
        for p in _paths(gs, s, kf, half_cap, fwd_ok):
            if back_zone is not None and back_zone.get(p[-1], INF) > kb:
                continue
            if tuple(p) not in seen:
                seen.add(tuple(p))
                fwd.append(p)
    mids = {p[-1] for p in fwd}

    to_mid = reach(gs, list(mids), kb) if prune else None
    bwd_ok = (lambda n, left: to_mid.get(n, INF) <= left) if prune else None
    back_by_mid: dict[str, list[list[str]]] = defaultdict(list)
    enumerated = 0
    for d in dsts:
        for p in _paths(gs, d, kb, half_cap, bwd_ok):      # answer .. midpoint
            enumerated += 1
            if p[-1] in mids:
                back_by_mid[p[-1]].append(p)

    out: list[list[str]] = []
    seen_out: set[tuple] = set()
    for pf in fwd:
        fset = set(pf)
        for pb in back_by_mid.get(pf[-1], ()):
            tail = pb[-2::-1]                               # midpoint excluded, ends at the answer
            if fset.intersection(tail):
                continue
            full = pf + tail
            if tuple(full) in seen_out:
                continue
            seen_out.add(tuple(full))
            out.append(full)
            if len(out) >= cap:
                break
        if len(out) >= cap:
            break
    if trace is not None:
        trace.update({"k": k, "forward_halves": len(fwd), "midpoints": len(mids),
                      "backward_enumerated": enumerated,
                      "backward_landed": sum(len(v) for v in back_by_mid.values()), "joined": len(out)})
    return out


def beam_paths(gs, seeds: list[str], k: int, edge_score: Callable[[str, str], float], beam: int = 40,
               per_node: int = 8) -> list[list[str]]:
    """Greedy baseline: keep the best `beam` partial paths by mean edge score at every hop. Fast, but it prunes
    the gold path whenever an early hop looks irrelevant to the question - which is typical for bridge hops."""
    frontier = [([s], 0.0) for s in seeds if gs.has(s)]
    done: list[tuple[list[str], float]] = []
    for depth in range(k):
        nxt = []
        for path, total in frontier:
            cands = [nb for nb in gs.neighbors(path[-1]) if nb not in path and not gs.is_hub(nb)]
            scored = sorted(((edge_score(path[-1], nb), nb) for nb in cands), reverse=True)[:per_node]
            nxt += [(path + [nb], total + s) for s, nb in scored]
        frontier = sorted(nxt, key=lambda x: -x[1] / len(x[0]))[:beam]
        done += [(p, t) for p, t in frontier if len(p) - 1 == depth + 1]
    return [p for p, _ in sorted(done, key=lambda x: -x[1] / (len(x[0]) - 1))]
