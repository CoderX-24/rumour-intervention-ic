"""
assortativity_by_size.py

Extends assortativity_analysis.py: tests whether the (dis)assortativity
gap (real vs Pruefer-rewired null) itself scales with thread size, the
same way the direct-removal advantage gap does in the paper's existing
Figure 1 (real-vs-null removal advantage grows ~38x from small to large
threads). If the assortativity gap shows the same size-dependent pattern,
that ties this new mechanism result directly to the paper's headline
finding rather than leaving it as an isolated statistic.

Uses the SAME thread-size buckets as the paper / extended_thread_analysis.py:
  small:  <= 10 nodes
  medium: 11-50 nodes
  large:  > 50 nodes

Method: identical to assortativity_analysis.py (pooled parent-child
out-degree Spearman rho, real vs 200 Pruefer-rewired nulls per event),
but pairs/edges are bucketed by the SIZE OF THE THREAD THE EDGE CAME FROM
before pooling and correlating, instead of pooling all edges together.

Usage:
  python3 assortativity_by_size.py <pheme_data_dir>

Output:
  assortativity_by_size_results.json
"""

import os
import sys
import json
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_ic import load_event
from degree_preserving_null import tree_to_pruefer, pruefer_to_tree, root_edges

EVENTS = [
    ("charliehebdo-all-rnr-threads", "Charlie Hebdo"),
    ("ottawashooting-all-rnr-threads", "Ottawa Shooting"),
    ("germanwings-crash-all-rnr-threads", "Germanwings"),
    ("sydneysiege-all-rnr-threads", "Sydney Siege"),
    ("ferguson-all-rnr-threads", "Ferguson"),
    ("prince-toronto-all-rnr-threads", "Prince Toronto"),
    ("putinmissing-all-rnr-threads", "Putin Missing"),
    # Small (14 threads / 222 nodes) — included but flagged, matching
    # compute_all_events.py's "underpowered=True, include=True" convention.
    # Gurlitt is deliberately excluded here too, per documented PHEME
    # data-quality issues (see compute_all_events.py) -- not overridden.
    ("ebola-essien-all-rnr-threads", "Ebola-Essien"),
]

N_REWIRES = 200
MIN_THREAD_SIZE = 4
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BUCKETS = [("small", 0, 10), ("medium", 11, 50), ("large", 51, float("inf"))]


def bucket_for(n_nodes):
    for name, lo, hi in BUCKETS:
        if lo <= n_nodes <= hi:
            return name
    return None


def real_pairs_by_bucket(threads):
    out = {name: ([], []) for name, _, _ in BUCKETS}
    for root, thread_graph in threads:
        all_nodes = list(thread_graph.nodes())
        n = len(all_nodes)
        if n < MIN_THREAD_SIZE:
            continue
        b = bucket_for(n)
        if b is None:
            continue
        children_map = defaultdict(list)
        for u, v in thread_graph.edges():
            children_map[u].append(v)
        out_deg = {nd: len(children_map.get(nd, [])) for nd in all_nodes}
        for parent, kids in children_map.items():
            if parent not in out_deg:
                continue
            for child in kids:
                if child not in out_deg:
                    continue
                out[b][0].append(out_deg[parent])
                out[b][1].append(out_deg[child])
    return out


def rewired_pairs_by_bucket(threads):
    out = {name: ([], []) for name, _, _ in BUCKETS}
    for root, thread_graph in threads:
        all_nodes = list(thread_graph.nodes())
        n = len(all_nodes)
        if n < MIN_THREAD_SIZE:
            continue
        b = bucket_for(n)
        if b is None:
            continue
        children_map = defaultdict(list)
        for u, v in thread_graph.edges():
            children_map[u].append(v)
        try:
            seq, nodes, idx = tree_to_pruefer(root, children_map, all_nodes)
            random.shuffle(seq)
            edges = pruefer_to_tree(seq, n)
            root_idx = idx[root]
            new_children = root_edges(edges, root_idx, n)
            out_deg = {i: len(new_children.get(i, [])) for i in range(n)}
            for parent, kids in new_children.items():
                for child in kids:
                    out[b][0].append(out_deg.get(parent, 0))
                    out[b][1].append(out_deg.get(child, 0))
        except Exception:
            continue
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 assortativity_by_size.py <pheme_data_dir>")
        sys.exit(1)
    base = sys.argv[1]

    print("=" * 70)
    print("PARENT-CHILD ASSORTATIVITY GAP BY THREAD SIZE")
    print("Tests whether the (dis)assortativity gap scales with thread size,")
    print("mirroring the paper's existing 38x small->large removal-advantage")
    print("finding (Figure 1).")
    print("Buckets: small <=10, medium 11-50, large >50 nodes")
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

        real_by_bucket = real_pairs_by_bucket(threads)

        null_rhos_by_bucket = {name: [] for name, _, _ in BUCKETS}
        for i in range(N_REWIRES):
            rewired = rewired_pairs_by_bucket(threads)
            for name, _, _ in BUCKETS:
                p_, c_ = rewired[name]
                if len(p_) > 10 and len(set(p_)) > 1 and len(set(c_)) > 1:
                    r, _ = spearmanr(p_, c_)
                    null_rhos_by_bucket[name].append(r)
            if (i + 1) % 50 == 0:
                print(f"    ...{i + 1}/{N_REWIRES} rewires done")

        event_result = {}
        print(f"\n  {label} results by thread size:")
        for name, lo, hi in BUCKETS:
            p_, c_ = real_by_bucket[name]
            if len(p_) < 10 or len(set(p_)) < 2 or len(set(c_)) < 2:
                print(f"    {name}: insufficient data (n_edges={len(p_)}) -- skipping")
                event_result[name] = {"n_edges_real": len(p_), "insufficient_data": True}
                continue
            real_rho, real_p = spearmanr(p_, c_)
            null_rhos = np.array(null_rhos_by_bucket[name])
            if len(null_rhos) < 5:
                print(f"    {name}: insufficient null samples -- skipping")
                event_result[name] = {"n_edges_real": len(p_), "insufficient_null": True}
                continue
            null_mean = null_rhos.mean()
            ci_low, ci_high = np.percentile(null_rhos, [2.5, 97.5])
            gap = real_rho - null_mean
            print(
                f"    {name:7s} (n={lo}-{hi if hi != float('inf') else 'inf'}): "
                f"real rho={real_rho:+.4f}  null mean={null_mean:+.4f}  "
                f"gap={gap:+.4f}  n_edges={len(p_)}"
            )
            event_result[name] = {
                "n_edges_real": int(len(p_)),
                "real_rho": float(real_rho),
                "real_p_value": float(real_p),
                "null_mean_rho": float(null_mean),
                "null_ci": [float(ci_low), float(ci_high)],
                "gap": float(gap),
            }

        all_results[label] = event_result

    print("\n" + "=" * 70)
    print("SUMMARY: does |gap| grow from small -> large threads?")
    print("=" * 70)
    for label, res in all_results.items():
        gaps = {}
        for name, _, _ in BUCKETS:
            v = res.get(name, {})
            if "gap" in v:
                gaps[name] = v["gap"]
        if len(gaps) == 3:
            print(f"  {label}: small={gaps['small']:+.4f}  medium={gaps['medium']:+.4f}  large={gaps['large']:+.4f}")
        else:
            have = ", ".join(f"{k}={v:+.4f}" for k, v in gaps.items())
            print(f"  {label}: incomplete buckets ({have or 'none'})")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assortativity_by_size_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
