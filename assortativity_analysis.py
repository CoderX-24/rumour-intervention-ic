"""
assortativity_analysis.py

Directly tests the "correlated branching" mechanism hypothesis raised in
the Discussion section: does the surviving 10-19% real-vs-degree-preserving-
null removal advantage come from high-degree nodes preferentially attaching
to other high-degree nodes (parent-child degree assortativity), as opposed
to e.g. homophily operating uniformly regardless of local degree structure?

Method:
  For every directed edge (parent -> child) in a thread, take the OUT-DEGREE
  pair (out_deg(parent), out_deg(child)). Pool these pairs across all edges
  in all threads for an event and compute Spearman's rank correlation
  rho_parent_child. Compare real rho against the same statistic computed on
  N_REWIRES Pruefer-sequence-rewired trees with identical degree sequences
  (using the exact rewiring machinery from degree_preserving_null.py, for
  methodological consistency with the paper's primary null).

  A positive real rho that exceeds the null's 95% CI is evidence that real
  reply trees preferentially wire high-out-degree nodes to other high-out-
  degree nodes (hub-to-hub attachment / correlated branching) beyond what
  the degree sequence alone predicts. This directly operationalizes the
  "correlated branching" hypothesis from Section IV.

Usage:
  python3 assortativity_analysis.py <pheme_data_dir>

Output:
  assortativity_results.json
"""

import os
import sys
import json
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_ic import load_event  # reuse existing PHEME loader
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
    ("ebola-essien-all-rnr-threads", "Ebola-Essien"),
]

# Events with too few threads for a stable Prueter-null CI; still run and
# reported, but flagged in output rather than silently trusted like the
# 7 primary/secondary events. Gurlitt is deliberately NOT included here:
# it is excluded pipeline-wide per documented PHEME data-quality issues
# (see compute_all_events.py), and that exclusion is not overridden here.
UNDERPOWERED_EVENTS = {"Ebola-Essien"}

N_REWIRES = 200
MIN_THREAD_SIZE = 4
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def pooled_parent_child_pairs_real(threads):
    """Pool (out_deg(parent), out_deg(child)) across every edge in every thread."""
    parent_degs, child_degs = [], []
    for root, thread_graph in threads:
        children_map = defaultdict(list)
        for u, v in thread_graph.edges():
            children_map[u].append(v)
        all_nodes = list(thread_graph.nodes())
        if len(all_nodes) < MIN_THREAD_SIZE:
            continue
        out_deg = {n: len(children_map.get(n, [])) for n in all_nodes}
        for parent, kids in children_map.items():
            if parent not in out_deg:
                continue
            for child in kids:
                if child not in out_deg:
                    continue
                parent_degs.append(out_deg[parent])
                child_degs.append(out_deg[child])
    return parent_degs, child_degs


def pooled_parent_child_pairs_rewired(threads):
    """One degree-preserving rewire of every thread; pool parent-child out-degree pairs."""
    parent_degs, child_degs = [], []
    for root, thread_graph in threads:
        children_map = defaultdict(list)
        for u, v in thread_graph.edges():
            children_map[u].append(v)
        all_nodes = list(thread_graph.nodes())
        n = len(all_nodes)
        if n < MIN_THREAD_SIZE:
            continue
        try:
            seq, nodes, idx = tree_to_pruefer(root, children_map, all_nodes)
            random.shuffle(seq)
            edges = pruefer_to_tree(seq, n)
            root_idx = idx[root]
            new_children = root_edges(edges, root_idx, n)
            out_deg = {i: len(new_children.get(i, [])) for i in range(n)}
            for parent, kids in new_children.items():
                for child in kids:
                    parent_degs.append(out_deg.get(parent, 0))
                    child_degs.append(out_deg.get(child, 0))
        except Exception:
            continue
    return parent_degs, child_degs


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 assortativity_analysis.py <pheme_data_dir>")
        sys.exit(1)
    base = sys.argv[1]

    print("=" * 70)
    print("PARENT-CHILD DEGREE ASSORTATIVITY (real vs Pruefer-rewired null)")
    print("Tests the 'correlated branching' mechanism hypothesis directly.")
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

        real_p, real_c = pooled_parent_child_pairs_real(threads)
        if len(real_p) < 10 or len(set(real_p)) < 2 or len(set(real_c)) < 2:
            print(f"  Skipping {label}: insufficient edge variance for correlation")
            continue
        real_rho, real_p_val = spearmanr(real_p, real_c)

        null_rhos = []
        for i in range(N_REWIRES):
            np_, nc_ = pooled_parent_child_pairs_rewired(threads)
            if len(np_) > 10 and len(set(np_)) > 1 and len(set(nc_)) > 1:
                r, _ = spearmanr(np_, nc_)
                null_rhos.append(r)
            if (i + 1) % 50 == 0:
                print(f"    ...{i + 1}/{N_REWIRES} rewires done")

        null_rhos = np.array(null_rhos)
        null_mean = null_rhos.mean()
        ci_low, ci_high = np.percentile(null_rhos, [2.5, 97.5])
        gap = real_rho - null_mean

        # Two-sided t-test: is the real value's "location" distinguishable
        # from the null distribution's mean (using null variance as reference)?
        # Reported alongside the direct 95% CI check (the paper's primary
        # decision rule, for consistency with Table 1's method).
        t_stat, t_p = ttest_ind([real_rho], null_rhos, equal_var=False)

        verdict = (
            "SIGNIFICANT POSITIVE ASSORTATIVITY (real exceeds null 95% CI)"
            if real_rho > ci_high
            else (
                "SIGNIFICANT NEGATIVE ASSORTATIVITY (real below null 95% CI)"
                if real_rho < ci_low
                else "NOT DISTINGUISHABLE FROM NULL (real within 95% CI)"
            )
        )

        flag = " [UNDERPOWERED: small N, wide CI expected]" if label in UNDERPOWERED_EVENTS else ""
        print(f"\n  Real parent-child rho:          {real_rho:.4f} (p={real_p_val:.2e}, n_edges={len(real_p)}){flag}")
        print(f"  Null mean rho:                  {null_mean:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Gap (real - null):               {gap:+.4f}")
        print(f"  Verdict: {verdict}")

        all_results[label] = {
            "real_rho": float(real_rho),
            "real_p_value": float(real_p_val),
            "n_edges_real": int(len(real_p)),
            "null_mean_rho": float(null_mean),
            "null_ci": [float(ci_low), float(ci_high)],
            "gap": float(gap),
            "verdict": verdict,
            "underpowered": label in UNDERPOWERED_EVENTS,
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_events = len(all_results)
    n_pos = sum(1 for v in all_results.values() if "POSITIVE" in v["verdict"])
    print(f"Significant positive assortativity in {n_pos}/{n_events} events")
    if n_pos == n_events and n_events > 0:
        print(
            "\nReal reply trees show parent-child out-degree assortativity that "
            "exceeds what the degree sequence alone predicts, in every event. "
            "This is direct evidence for the correlated-branching hypothesis: "
            "high-out-degree nodes preferentially attach to other high-out-degree "
            "nodes, beyond what Pruefer rewiring of the same degree sequence "
            "produces. This mechanism, combined with the thread-size moderator "
            "finding (larger gap in larger threads), is consistent with hub-to-hub "
            "amplification structure being concentrated in larger cascades."
        )
    elif n_pos == 0 and n_events > 0:
        print(
            "\nReal reply trees do NOT show parent-child assortativity beyond "
            "the degree-preserving null in any event. This argues AGAINST the "
            "correlated-branching hypothesis and shifts weight toward the "
            "alternative (homophily / other structural) explanation for the "
            "surviving direct-removal advantage."
        )
    else:
        print(
            "\nMixed result across events -- assortativity is not a uniform "
            "explanation for the surviving direct-removal advantage. Report "
            "per-event results rather than a single pooled claim."
        )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assortativity_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
