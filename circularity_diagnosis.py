"""
circularity_diagnosis.py

Extends degree_preserving_null.py's finding (real r <= degree-preserving
null r, i.e. the degree-impact correlation is substantially mechanical)
by diagnosing WHERE the mechanical relationship comes from and whether
it is uniform across the dataset.

Two analyses:

  1. DEPTH-CONTROLLED CHECK: nodes closer to the root have mechanically
     larger subtrees available to them (more of the tree is "downstream").
     We test whether real r(degree, impact) and the depth-preserving null
     gap both shrink once depth is controlled for -- via (a) computing
     the correlation WITHIN fixed depth bands only, and (b) a partial
     Spearman correlation of degree vs. impact controlling for depth.
     If most of the real/null gap disappears once depth is controlled,
     depth-from-root -- not degree itself -- was doing the real work.

  2. SIZE-STRATIFIED CHECK: reruns the real-vs-null gap separately for
     small (<=10 nodes), medium (11-50), and large (>50 node) threads,
     to test whether the mechanical artifact identified by
     degree_preserving_null.py is uniform across thread sizes or
     concentrated in large threads (which is where the original paper's
     effect was largest -- if the artifact is also largest there, that
     directly explains why the naive result looked so clean).

Usage:
  python3 circularity_diagnosis.py <pheme_data_dir>

Requires calibrate_ic.py (for load_event) in the same directory.
Reuses the Pruefer-rewiring machinery from degree_preserving_null.py.
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
from degree_preserving_null import (
    tree_to_pruefer,
    pruefer_to_tree,
    root_edges,
    subtree_sizes,
)

EVENTS = [
    ("charliehebdo-all-rnr-threads", "Charlie Hebdo"),
    ("ottawashooting-all-rnr-threads", "Ottawa Shooting"),
    ("germanwings-crash-all-rnr-threads", "Germanwings"),
    ("sydneysiege-all-rnr-threads", "Sydney Siege"),
    ("ferguson-all-rnr-threads", "Ferguson"),
    ("prince-toronto-all-rnr-threads", "Prince Toronto"),
    ("putinmissing-all-rnr-threads", "Putin Missing"),
]

N_REWIRES = 100  # fewer than degree_preserving_null.py's 200, for runtime
MIN_THREAD_SIZE = 4
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SIZE_BINS = [("small (<=10)", 0, 10), ("medium (11-50)", 11, 50), ("large (>50)", 51, float("inf"))]
DEPTH_MAX_BAND = 6  # analyze depth 0..5 individually, pool the rest


def node_depths(root, children_map):
    """BFS depth of every node from root."""
    depth = {root: 0}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for c in children_map.get(u, []):
            if c not in depth:
                depth[c] = depth[u] + 1
                queue.append(c)
    return depth


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation of x,y controlling for z, via the
    standard rank-based residual method: rank-transform all three,
    regress out z (Pearson-style linear residuals on ranks), then
    correlate the residuals.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)

    def residual(a, b):
        # residual of a after regressing out b (simple linear OLS on ranks)
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    rx_resid = residual(rx, rz)
    ry_resid = residual(ry, rz)
    if np.std(rx_resid) == 0 or np.std(ry_resid) == 0:
        return np.nan
    r = np.corrcoef(rx_resid, ry_resid)[0, 1]
    return r


def collect_event_data(threads):
    """
    For every node in every valid thread, collect (degree, impact, depth,
    thread_size). Also returns per-thread (root, children_map, all_nodes)
    for the rewiring step.
    """
    rows = []  # (degree, impact, depth, thread_size)
    thread_records = []

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
        depths = node_depths(root, children_map)
        reachable = [n for n in all_nodes if n in sizes and n in depths]
        thread_size = len(reachable)

        for n in reachable:
            deg = len(children_map.get(n, []))
            rows.append((deg, sizes[n], depths[n], thread_size))

        thread_records.append((root, children_map, reachable))

    return rows, thread_records


def rewired_pooled_with_depth(thread_records):
    """One Pruefer rewire pass over all threads, returning pooled
    (degree, impact, depth, thread_size) rows for the REWIRED trees.
    Depth is recomputed on the rewired tree structure (rooted at the
    same node index)."""
    rows = []
    for root, children_map, reachable in thread_records:
        n = len(reachable)
        if n < MIN_THREAD_SIZE:
            continue
        try:
            seq, nodes, idx = tree_to_pruefer(root, children_map, reachable)
            random.shuffle(seq)
            edges = pruefer_to_tree(seq, n)
            root_idx = idx[root]
            new_children = root_edges(edges, root_idx, n)
            sizes = subtree_sizes(root_idx, new_children)
            depths = node_depths(root_idx, new_children)
            for i in range(n):
                if i in sizes and i in depths:
                    deg = len(new_children.get(i, []))
                    rows.append((deg, sizes[i], depths[i], n))
        except Exception:
            continue
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 circularity_diagnosis.py <pheme_data_dir>")
        sys.exit(1)
    base = sys.argv[1]

    print("=" * 70)
    print("CIRCULARITY DIAGNOSIS: depth-controlled + size-stratified")
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

        rows, thread_records = collect_event_data(threads)
        degs = np.array([r[0] for r in rows])
        imps = np.array([r[1] for r in rows])
        depths = np.array([r[2] for r in rows])
        tsizes = np.array([r[3] for r in rows])

        # ---- 1a. Overall real r (sanity check vs degree_preserving_null.py) ----
        real_r, _ = spearmanr(degs, imps)

        # ---- 1b. Depth-stratified r: within each depth band ----
        print(f"  Depth-stratified r(degree, impact):")
        depth_band_rs = {}
        for d in range(DEPTH_MAX_BAND):
            mask = depths == d
            if mask.sum() > 10 and len(set(degs[mask])) > 1 and len(set(imps[mask])) > 1:
                r_d, _ = spearmanr(degs[mask], imps[mask])
                depth_band_rs[d] = {"r": float(r_d), "n": int(mask.sum())}
                print(f"    depth={d}: r={r_d:.4f}  (n={mask.sum()})")
        deep_mask = depths >= DEPTH_MAX_BAND
        if deep_mask.sum() > 10:
            r_deep, _ = spearmanr(degs[deep_mask], imps[deep_mask])
            depth_band_rs[f">={DEPTH_MAX_BAND}"] = {"r": float(r_deep), "n": int(deep_mask.sum())}
            print(f"    depth>={DEPTH_MAX_BAND}: r={r_deep:.4f}  (n={deep_mask.sum()})")

        # ---- 1c. Partial correlation controlling for depth ----
        partial_r = partial_spearman(degs, imps, depths)
        print(f"  Overall real r:                  {real_r:.4f}")
        print(f"  Partial r(degree,impact | depth): {partial_r:.4f}")
        depth_explains_most = abs(partial_r) < abs(real_r) * 0.5
        print(f"  {'Depth controls explain MOST of the correlation (linear control only)' if depth_explains_most else 'A LINEAR depth control does not collapse the correlation -- degree appears to carry information beyond a linear depth effect; nonlinear depth-adjustment not tested here'}")

        # ---- 2. Size-stratified real-vs-null gap ----
        print(f"\n  Size-stratified real vs. degree-preserving-null gap "
              f"({N_REWIRES} rewires):")
        size_results = {}
        for bin_label, lo, hi in SIZE_BINS:
            mask = (tsizes >= lo) & (tsizes <= hi) if hi != float("inf") else (tsizes >= lo)
            if mask.sum() < 20:
                print(f"    {bin_label}: insufficient data (n={mask.sum()})")
                continue
            real_r_bin, _ = spearmanr(degs[mask], imps[mask])

            # rewired nulls restricted to threads whose size falls in this bin
            bin_thread_records = [
                (root, cm, nodes) for root, cm, nodes in thread_records
                if lo <= len(nodes) <= (hi if hi != float("inf") else len(nodes))
            ]
            null_rs = []
            for _ in range(N_REWIRES):
                rewired_rows = rewired_pooled_with_depth(bin_thread_records)
                if len(rewired_rows) > 10:
                    rd = np.array([r[0] for r in rewired_rows])
                    ri = np.array([r[1] for r in rewired_rows])
                    rr, _ = spearmanr(rd, ri)
                    null_rs.append(rr)
            if null_rs:
                null_mean = float(np.mean(null_rs))
                gap = real_r_bin - null_mean
                print(f"    {bin_label}: real r={real_r_bin:.4f}, null mean r={null_mean:.4f}, "
                      f"gap={gap:+.4f}  (n_nodes={mask.sum()}, n_threads={len(bin_thread_records)})")
                size_results[bin_label] = {
                    "real_r": real_r_bin, "null_mean_r": null_mean, "gap": gap,
                    "n_nodes": int(mask.sum()), "n_threads": len(bin_thread_records),
                }

        all_results[label] = {
            "overall_real_r": float(real_r),
            "partial_r_controlling_depth": float(partial_r),
            "depth_explains_most_of_correlation": bool(depth_explains_most),
            "depth_band_r": depth_band_rs,
            "size_stratified": size_results,
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_depth_dominant = sum(
        1 for v in all_results.values() if v["depth_explains_most_of_correlation"]
    )
    print(f"A LINEAR depth control does not fully explain the correlation in "
          f"{len(all_results) - n_depth_dominant}/{len(all_results)} events "
          f"(nonlinear depth-adjustment not tested)")

    # Check whether the artifact (negative/near-zero gap) is worse for large threads
    MIN_THREADS_FOR_RELIABLE_BIN = 5  # a bin backed by fewer threads than
    # this is reported individually but excluded from the pooled summary,
    # since a single thread can swing the bin's gap arbitrarily.
    large_gaps, small_gaps, excluded_bins = [], [], []
    for label, v in all_results.items():
        if "large (>50)" in v["size_stratified"]:
            entry = v["size_stratified"]["large (>50)"]
            if entry["n_threads"] >= MIN_THREADS_FOR_RELIABLE_BIN:
                large_gaps.append(entry["gap"])
            else:
                excluded_bins.append((label, "large", entry["gap"], entry["n_threads"]))
        if "small (<=10)" in v["size_stratified"]:
            entry = v["size_stratified"]["small (<=10)"]
            if entry["n_threads"] >= MIN_THREADS_FOR_RELIABLE_BIN:
                small_gaps.append(entry["gap"])
            else:
                excluded_bins.append((label, "small", entry["gap"], entry["n_threads"]))

    if excluded_bins:
        print(f"\nExcluded from pooled summary (n_threads < {MIN_THREADS_FOR_RELIABLE_BIN}, "
              f"too few to be reliable):")
        for label, size_cat, gap, n_t in excluded_bins:
            print(f"    {label} / {size_cat}: gap={gap:+.4f} (n_threads={n_t}) -- "
                  f"reported individually, not pooled")
    if large_gaps and small_gaps:
        print(f"Mean gap, large threads (>50 nodes): {np.mean(large_gaps):+.4f}")
        print(f"Mean gap, small threads (<=10 nodes): {np.mean(small_gaps):+.4f}")
        if np.mean(large_gaps) < np.mean(small_gaps):
            print("The mechanical artifact is MORE pronounced in large threads -- "
                  "consistent with the original (circular) finding appearing "
                  "strongest exactly where degree-impact entanglement is worst.")
        else:
            print("The mechanical artifact is not concentrated in large threads; "
                  "size does not obviously explain the original effect's pattern.")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "circularity_diagnosis_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
