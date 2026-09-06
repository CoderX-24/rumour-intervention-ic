"""
test_pruefer_roundtrip.py

Standalone correctness check for the Pruefer encode/decode machinery in
degree_preserving_null.py, which is now load-bearing for THREE analyses
(degree_preserving_null.py, assortativity_analysis.py, assortativity_by_size.py).
This does not require PHEME data -- it builds synthetic trees of known
shape and verifies invariants that MUST hold if the rewiring is correct.

Checks, for many random trees of varying size/shape:
  1. tree_to_pruefer -> pruefer_to_tree round-trips to a tree with the
     EXACT SAME degree sequence as the original (this is the entire point
     of using Pruefer sequences as a degree-preserving null).
  2. The decoded edge set forms a valid tree: exactly n-1 edges, connected,
     acyclic (i.e. nx.is_tree passes).
  3. After random.shuffle(seq) + decode (the actual null-generation step
     used throughout the pipeline), the same two invariants still hold on
     the REWIRED tree, across many repeated shuffles.
  4. Degenerate small cases (n=2, n=3) are handled without crashing or
     silently producing a wrong-length edge list, since the "closing edge"
     logic in pruefer_to_tree looked fragile on inspection.

This is a diagnostic only -- it does not touch degree_preserving_null.py.
If everything below prints PASS, the null-generation machinery your
Table 1 / assortativity results depend on is verified correct on both
typical and edge-case tree shapes matching PHEME's observed size range.

Usage:
  python3 test_pruefer_roundtrip.py
"""

import sys
import os
import random
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degree_preserving_null import tree_to_pruefer, pruefer_to_tree, root_edges

random.seed(0)


def random_tree_children_map(n, shape="random"):
    """Build a random rooted tree (as a children_map) with n nodes, root = 0."""
    nodes = list(range(n))
    children_map = defaultdict(list)
    if shape == "star":
        for i in range(1, n):
            children_map[0].append(i)
    elif shape == "path":
        for i in range(1, n):
            children_map[i - 1].append(i)
    else:  # random attachment
        attached = [0]
        for i in range(1, n):
            parent = random.choice(attached)
            children_map[parent].append(i)
            attached.append(i)
    return children_map, nodes


def degree_sequence(children_map, all_nodes, root):
    """Undirected degree = out_deg + (1 if not root else 0)."""
    deg = {}
    for n in all_nodes:
        out_deg = len(children_map.get(n, []))
        deg[n] = out_deg + (0 if n == root else 1)
    return deg


def build_nx_tree(children_map, all_nodes):
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for p, kids in children_map.items():
        for c in kids:
            G.add_edge(p, c)
    return G


def check_one(n, shape, n_shuffles=20, verbose=False):
    children_map, all_nodes = random_tree_children_map(n, shape=shape)
    root = 0
    orig_deg = degree_sequence(children_map, all_nodes, root)

    failures = []

    # --- Check 1 & 2: plain round-trip (no shuffle) reproduces an
    # isomorphic-degree-sequence valid tree ---
    try:
        seq, nodes, idx = tree_to_pruefer(root, children_map, all_nodes)
    except Exception as e:
        return [f"n={n} shape={shape}: tree_to_pruefer raised {e!r}"]

    try:
        edges = pruefer_to_tree(list(seq), n)
    except Exception as e:
        return [f"n={n} shape={shape}: pruefer_to_tree (no shuffle) raised {e!r}"]

    if len(edges) != n - 1:
        failures.append(
            f"n={n} shape={shape}: no-shuffle decode produced {len(edges)} edges, expected {n - 1}"
        )
    else:
        G_decoded = nx.Graph()
        G_decoded.add_nodes_from(range(n))
        G_decoded.add_edges_from(edges)
        if not nx.is_tree(G_decoded):
            failures.append(f"n={n} shape={shape}: no-shuffle decode is not a valid tree")
        else:
            decoded_deg = dict(G_decoded.degree())
            # degree by original index label mapping
            orig_deg_by_idx = {idx[k]: v for k, v in orig_deg.items()}
            if decoded_deg != orig_deg_by_idx:
                failures.append(
                    f"n={n} shape={shape}: no-shuffle degree sequence mismatch "
                    f"(orig={sorted(orig_deg_by_idx.values())}, "
                    f"decoded={sorted(decoded_deg.values())})"
                )

    # --- Check 3: repeated shuffles preserve degree sequence + validity ---
    for trial in range(n_shuffles):
        seq2 = list(seq)
        random.shuffle(seq2)
        try:
            edges2 = pruefer_to_tree(seq2, n)
        except Exception as e:
            failures.append(f"n={n} shape={shape} shuffle#{trial}: pruefer_to_tree raised {e!r}")
            continue
        if len(edges2) != n - 1:
            failures.append(
                f"n={n} shape={shape} shuffle#{trial}: produced {len(edges2)} edges, expected {n - 1}"
            )
            continue
        G2 = nx.Graph()
        G2.add_nodes_from(range(n))
        G2.add_edges_from(edges2)
        if not nx.is_tree(G2):
            failures.append(f"n={n} shape={shape} shuffle#{trial}: rewired result is not a valid tree")
            continue
        decoded_deg2 = dict(G2.degree())
        orig_deg_by_idx = {idx[k]: v for k, v in orig_deg.items()}
        if decoded_deg2 != orig_deg_by_idx:
            failures.append(
                f"n={n} shape={shape} shuffle#{trial}: rewired degree sequence mismatch "
                f"(orig={sorted(orig_deg_by_idx.values())}, "
                f"rewired={sorted(decoded_deg2.values())})"
            )

    return failures


def main():
    print("=" * 70)
    print("PRUEFER ROUND-TRIP CORRECTNESS TEST")
    print("Verifies degree_preserving_null.py's encode/decode + shuffle")
    print("preserves the exact degree sequence and produces valid trees.")
    print("=" * 70)

    test_cases = []
    # Degenerate / small edge cases -- these are the ones most likely to
    # break given the somewhat fragile-looking closing-edge logic in
    # pruefer_to_tree.
    for n in [2, 3, 4, 5]:
        test_cases.append((n, "random"))
    # Typical PHEME-scale sizes, varying shape (star = high-degree root,
    # path = pure chain / worst case for degree skew, random = typical).
    for n in [10, 25, 50, 100, 300]:
        for shape in ["star", "path", "random"]:
            test_cases.append((n, shape))

    all_failures = []
    n_checked = 0
    for n, shape in test_cases:
        n_checked += 1
        failures = check_one(n, shape, n_shuffles=20)
        status = "PASS" if not failures else "FAIL"
        print(f"  [{status}] n={n:4d}  shape={shape:8s}")
        all_failures.extend(failures)

    print("\n" + "=" * 70)
    if not all_failures:
        print(f"ALL {n_checked} TEST CASES PASSED (each with 20 shuffle trials).")
        print("The Pruefer rewiring machinery correctly preserves the exact")
        print("degree sequence and produces valid trees across typical PHEME")
        print("thread sizes, degenerate small cases, and extreme shapes")
        print("(star / path). Table 1, degree_preserving_null_results.json,")
        print("and both assortativity scripts can be trusted to be operating")
        print("on genuinely degree-matched random trees.")
    else:
        print(f"{len(all_failures)} FAILURES across {n_checked} test cases:")
        for f in all_failures[:50]:
            print(f"  - {f}")
        if len(all_failures) > 50:
            print(f"  ... and {len(all_failures) - 50} more")
        print("\nDo NOT trust downstream null-model results until these are fixed.")
    print("=" * 70)

    sys.exit(1 if all_failures else 0)


if __name__ == "__main__":
    main()
