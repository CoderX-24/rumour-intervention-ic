"""
degree_preserving_null.py

Directly tests the circularity concern: is r(degree, impact) ~= 0.97-0.99
just a mechanical consequence of tree geometry (a node's children are
definitionally part of its own subtree), or does it reflect something
beyond what ANY tree with the same degree sequence would show?

Method (Pruefer-sequence rewiring):
  A classical bijection (Pruefer 1918) maps labeled trees on n nodes to
  sequences of length n-2, where each node i appears in the sequence
  exactly (degree_i - 1) times. This means: if you take a real tree's
  degree sequence, build its Pruefer sequence, RANDOMLY SHUFFLE that
  sequence, and decode it back into a tree, you get a uniformly random
  tree with the EXACT SAME degree sequence as the original -- but
  completely randomized topology (who is whose parent/child is
  scrambled, subject only to preserving how many edges each node has).

  This is a strictly stronger control than synthetic_baseline.py's
  uniform/Poisson-branching generators, which only matched thread SIZE,
  not the actual degree sequence. If r(degree, impact) collapses toward
  the null under DEGREE-PRESERVING rewiring, that is strong evidence the
  original result is substantially mechanical/tautological. If r stays
  far above the degree-preserving null, that is strong evidence of a
  real effect beyond what degree-sequence + tree-structure alone predicts.

Usage:
  python3 degree_preserving_null.py <pheme_data_dir>

Runtime: scales with N_REWIRES x total nodes across all threads; expect
several minutes per event with N_REWIRES=200 on a laptop.
"""

import os
import sys
import json
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_ic import load_event  # reuse existing PHEME loader

EVENTS = [
    ("charliehebdo-all-rnr-threads", "Charlie Hebdo"),
    ("ottawashooting-all-rnr-threads", "Ottawa Shooting"),
    ("germanwings-crash-all-rnr-threads", "Germanwings"),
    ("sydneysiege-all-rnr-threads", "Sydney Siege"),
    ("ferguson-all-rnr-threads", "Ferguson"),
]

N_REWIRES = 200  # random degree-preserving rewires per event
MIN_THREAD_SIZE = 4  # Pruefer sequence needs n>=2; skip trivial threads
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def tree_to_degree_sequence(root, children_map, all_nodes):
    """Total (undirected) degree of each node = out-degree + (1 if not root)."""
    deg = {}
    for n in all_nodes:
        out_deg = len(children_map.get(n, []))
        deg[n] = out_deg + (0 if n == root else 1)
    return deg


def tree_to_pruefer(root, children_map, all_nodes):
    """
    Standard Pruefer encoding for a labeled tree (undirected edge list
    derived from the rooted parent/child structure). Returns the Pruefer
    sequence (list of node labels), using integer-relabeled nodes
    0..n-1 internally for stability, with a mapping back to original ids.
    """
    nodes = list(all_nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    edges = []
    for parent, kids in children_map.items():
        if parent not in idx:
            continue
        for k in kids:
            if k in idx:
                edges.append((idx[parent], idx[k]))

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    degree = {i: len(adj[i]) for i in range(n)}
    # Leaves = degree-1 nodes (standard Pruefer encoding)
    import heapq
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    adj_work = {i: set(adj[i]) for i in range(n)}
    deg_work = dict(degree)
    seq = []
    removed = set()

    for _ in range(n - 2):
        leaf = heapq.heappop(leaves)
        while leaf in removed:
            leaf = heapq.heappop(leaves)
        removed.add(leaf)
        (neighbor,) = adj_work[leaf] - removed if adj_work[leaf] - removed else (None,)
        if neighbor is None:
            # fallback: pick any remaining neighbor
            remaining = [x for x in adj_work[leaf] if x not in removed]
            neighbor = remaining[0]
        seq.append(neighbor)
        deg_work[neighbor] -= 1
        adj_work[neighbor].discard(leaf)
        if deg_work[neighbor] == 1 and neighbor not in removed:
            heapq.heappush(leaves, neighbor)

    return seq, nodes, idx


def pruefer_to_tree(seq, n):
    """Decode a Pruefer sequence (0..n-1 labels) back into an edge list."""
    degree = [1] * n
    for x in seq:
        degree[x] += 1

    import heapq
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    edges = []
    deg = list(degree)
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, x))
        deg[leaf] -= 1
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(leaves, x)

    # Final two remaining nodes with degree 1
    remaining = [i for i in range(n) if deg[i] == 1 and not any(i in e for e in edges[-1:])]
    remaining = [i for i in range(n) if deg[i] >= 1]
    u, v = remaining[0], remaining[1] if len(remaining) > 1 else remaining[0]
    edges.append((u, v))
    return edges


def root_edges(edges, root_idx, n):
    """Convert an undirected edge list into a rooted children map via BFS."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    children = defaultdict(list)
    visited = {root_idx}
    queue = [root_idx]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                children[u].append(v)
                queue.append(v)
    return children


def subtree_sizes(root_idx, children):
    """Descendant count + self for every node, via post-order traversal."""
    sizes = {}

    def dfs(u):
        s = 1
        for c in children.get(u, []):
            s += dfs(c)
        sizes[u] = s
        return s

    dfs(root_idx)
    return sizes


def real_degree_impact_r(root, children_map, all_nodes):
    sizes = {}

    def dfs(u):
        s = 1
        for c in children_map.get(u, []):
            s += dfs(c)
        sizes[u] = s
        return s

    dfs(root)
    # Defensive: only use nodes actually reachable from root. A handful of
    # PHEME threads contain nodes not connected via reply edges to the
    # declared root (e.g. duplicate/malformed structure.json entries);
    # silently including them caused a KeyError. We drop them here rather
    # than crash, and this affects a small minority of nodes/threads.
    reachable = all_nodes if len(sizes) == len(all_nodes) else [n for n in all_nodes if n in sizes]
    degs = [len(children_map.get(n, [])) for n in reachable]
    imps = [sizes[n] for n in reachable]
    if len(set(degs)) < 2 or len(set(imps)) < 2:
        return None
    r, _ = spearmanr(degs, imps)
    return r


def rewired_degree_impact_r(root, children_map, all_nodes):
    """One degree-preserving rewire + its degree-impact Spearman r."""
    n = len(all_nodes)
    if n < 4:
        return None
    try:
        seq, nodes, idx = tree_to_pruefer(root, children_map, all_nodes)
        random.shuffle(seq)
        edges = pruefer_to_tree(seq, n)
        root_idx = idx[root]
        new_children = root_edges(edges, root_idx, n)
        sizes = subtree_sizes(root_idx, new_children)
        degs = [len(new_children.get(i, [])) for i in range(n)]
        imps = [sizes.get(i, 1) for i in range(n)]
        if len(set(degs)) < 2 or len(set(imps)) < 2:
            return None
        r, _ = spearmanr(degs, imps)
        return r
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 degree_preserving_null.py <pheme_data_dir>")
        sys.exit(1)
    base = sys.argv[1]

    print("=" * 70)
    print("DEGREE-PRESERVING NULL MODEL (Pruefer-sequence rewiring)")
    print("Tests whether r(degree, impact) survives when tree topology is")
    print("randomized but each node's EXACT degree is held fixed.")
    print("=" * 70)

    all_results = {}

    for event_dir, label in EVENTS:
        event_path = os.path.join(base, event_dir)
        if not os.path.isdir(event_path):
            print(f"\n{label}: directory not found -- skipping")
            continue

        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, thread_sizes, threads = load_event(event_path)
        print(f"done ({len(threads)} threads)")

        real_rs = []
        rewired_rs = []

        for root, thread_graph in threads:
            children_map = defaultdict(list)
            for u, v in thread_graph.edges():
                children_map[u].append(v)
            all_nodes = list(thread_graph.nodes())
            if len(all_nodes) < MIN_THREAD_SIZE:
                continue

            r_real = real_degree_impact_r(root, children_map, all_nodes)
            if r_real is not None:
                real_rs.append(r_real)

        # Pool-level analysis: rather than per-thread r (often degenerate on
        # small threads), pool (degree, impact) pairs across ALL nodes in ALL
        # threads for the event, matching the original analysis's approach.
        real_degs, real_imps = [], []
        for root, thread_graph in threads:
            children_map = defaultdict(list)
            for u, v in thread_graph.edges():
                children_map[u].append(v)
            all_nodes = list(thread_graph.nodes())
            if len(all_nodes) < MIN_THREAD_SIZE:
                continue
            sizes = {}

            def dfs(u, sizes=sizes, cm=children_map):
                s = 1
                for c in cm.get(u, []):
                    s += dfs(c, sizes, cm)
                sizes[u] = s
                return s

            dfs(root)
            reachable = [nd for nd in all_nodes if nd in sizes]
            for nd in reachable:
                real_degs.append(len(children_map.get(nd, [])))
                real_imps.append(sizes[nd])

        real_r, _ = spearmanr(real_degs, real_imps)

        rewired_r_samples = []
        for rewire_i in range(N_REWIRES):
            pooled_degs, pooled_imps = [], []
            for root, thread_graph in threads:
                children_map = defaultdict(list)
                for u, v in thread_graph.edges():
                    children_map[u].append(v)
                all_nodes = list(thread_graph.nodes())
                if len(all_nodes) < MIN_THREAD_SIZE:
                    continue
                n = len(all_nodes)
                idx_map = {nd: i for i, nd in enumerate(all_nodes)}
                try:
                    seq, nodes, idx = tree_to_pruefer(root, children_map, all_nodes)
                    random.shuffle(seq)
                    edges = pruefer_to_tree(seq, n)
                    root_idx = idx[root]
                    new_children = root_edges(edges, root_idx, n)
                    sizes = subtree_sizes(root_idx, new_children)
                    reachable_idx = [i for i in range(n) if i in sizes]
                    for i in reachable_idx:
                        pooled_degs.append(len(new_children.get(i, [])))
                        pooled_imps.append(sizes[i])
                except Exception:
                    continue
            if len(pooled_degs) > 10:
                rr, _ = spearmanr(pooled_degs, pooled_imps)
                rewired_r_samples.append(rr)
            if (rewire_i + 1) % 50 == 0:
                print(f"    ...{rewire_i + 1}/{N_REWIRES} rewires done")

        rewired_r_samples = np.array(rewired_r_samples)
        rewired_mean = rewired_r_samples.mean()
        ci_low, ci_high = np.percentile(rewired_r_samples, [2.5, 97.5])
        gap = real_r - rewired_mean

        print(f"\n  Real r(degree, impact):           {real_r:.4f}")
        print(f"  Degree-preserving null mean r:    {rewired_mean:.4f}  "
              f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Gap (real - degree-preserving):   {gap:+.4f}")
        verdict = (
            "INFORMATIVE (real structure exceeds degree-preserving null)"
            if real_r > ci_high
            else "CIRCULARITY CONFIRMED (real r within degree-preserving null range)"
        )
        print(f"  Verdict: {verdict}")

        all_results[label] = {
            "real_r": float(real_r),
            "degree_preserving_null_mean_r": float(rewired_mean),
            "degree_preserving_null_ci": [float(ci_low), float(ci_high)],
            "gap": float(gap),
            "verdict": verdict,
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_informative = sum(1 for v in all_results.values() if "INFORMATIVE" in v["verdict"])
    n_total = len(all_results)
    print(f"Informative (real > degree-preserving null) in {n_informative}/{n_total} events")

    if n_informative == n_total:
        print(
            "\nThe degree-preserving null model -- the strongest available control, "
            "holding each node's exact degree fixed while randomizing topology -- "
            "still shows real PHEME trees produce higher degree-impact correlation "
            "than randomized trees with the identical degree sequence. This is "
            "evidence AGAINST the circularity concern being the primary driver of "
            "the headline result: something about the real ARRANGEMENT of degrees "
            "in PHEME trees (not just the degree values themselves) predicts impact."
        )
    else:
        print(
            "\nAt least one event's real r falls within the degree-preserving null "
            "range. This is evidence FOR the circularity concern: the observed "
            "degree-impact correlation may be substantially explained by degree "
            "sequence and tree size alone, not by real PHEME topology. If this "
            "holds broadly, the paper's headline framing needs to be revised "
            "toward a weaker or more qualified claim."
        )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "degree_preserving_null_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
