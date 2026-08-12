# pagerank_greedy_oracle.py
# Addresses:
#   - PageRank correlation with cascade impact (vs degree and betweenness)
#   - Greedy oracle for 1-2 events (true upper bound on cascade truncation)
#
# Usage: python3 pagerank_greedy_oracle.py <pheme_data_dir>
# Runtime: ~10 minutes

import os, json, sys
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

def build_tree_graph(structure, graph=None):
    if graph is None:
        graph = nx.DiGraph()
    if isinstance(structure, dict):
        for parent, children in structure.items():
            graph.add_node(str(parent))
            if isinstance(children, dict):
                for child in children.keys():
                    graph.add_edge(str(parent), str(child))
                build_tree_graph(children, graph)
            elif isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        build_tree_graph(child, graph)
                    else:
                        graph.add_edge(str(parent), str(child))
    return graph

def load_event(event_path):
    G = nx.DiGraph()
    roots, threads = [], []
    for cat in ["rumours","non-rumours"]:
        cat_path = os.path.join(event_path, cat)
        if not os.path.isdir(cat_path): continue
        for tid in sorted(os.listdir(cat_path)):
            if tid.startswith('.'): continue
            sf = os.path.join(cat_path, tid, 'structure.json')
            if not os.path.exists(sf): continue
            try:
                with open(sf) as f:
                    sd = json.load(f)
                tg = build_tree_graph(sd)
                if tg.number_of_nodes() < 2: continue
                G = nx.compose(G, tg)
                root = str(list(sd.keys())[0])
                roots.append(root)
                threads.append((root, tg))
            except Exception:
                pass
    return G, sorted(roots), threads

def desc_plus_self(tg, node):
    if node not in tg: return set()
    return {node} | nx.descendants(tg, node)

def cascade_after_removal(threads, removed):
    total = 0
    for _, tg in threads:
        surv = set(tg.nodes())
        for n in removed:
            if n in tg:
                surv -= desc_plus_self(tg, n)
        total += len(surv)
    return total

def greedy_oracle(threads, root_set, budget, label):
    """
    True greedy oracle: at each step, select the non-seed node with the
    highest remaining cascade impact, remove it and its descendants,
    recompute impacts. Repeat until budget exhausted.
    O(budget * V) — feasible for small events.
    """
    print(f"  Computing greedy oracle (budget={budget})...", end=" ", flush=True)

    # Build working copy of thread graphs
    remaining = {i: set(tg.nodes()) for i, (_, tg) in enumerate(threads)}
    removed_total = set()
    base = sum(len(nodes) for nodes in remaining.values())

    def current_impact(n):
        total = 0
        for i, (_, tg) in enumerate(threads):
            if n in remaining[i]:
                total += len(desc_plus_self(tg, n) & remaining[i])
        return total

    all_nonseed = set()
    for _, tg in threads:
        all_nonseed.update(n for n in tg.nodes() if n not in root_set)

    candidates = set(all_nonseed)

    for step in range(budget):
        if not candidates:
            break
        # Find highest-impact candidate
        best_node = max(candidates, key=current_impact)
        best_impact = current_impact(best_node)
        if best_impact <= 1:  # Only removes itself
            break
        # Remove it and descendants from remaining
        for i, (_, tg) in enumerate(threads):
            if best_node in remaining[i]:
                to_remove = desc_plus_self(tg, best_node) & remaining[i]
                remaining[i] -= to_remove
                # These removed nodes can no longer be selected
                candidates -= to_remove
        removed_total.add(best_node)

    after_greedy = sum(len(nodes) for nodes in remaining.values())
    greedy_r = 100.0*(base - after_greedy)/base
    print(f"done. Reduction: {greedy_r:.2f}%")
    return greedy_r

if __name__ == "__main__":
    base_dir = sys.argv[1]

    # ── PageRank correlation analysis (all 5 large events) ────────────────────
    print("PageRank Correlation with Cascade Impact")
    print("="*65)
    print("(All 5 large events, 2,000 node sample per event)")
    print()
    print(f"{'Event':<22} {'r(degree)':>10} {'r(pagerank)':>12} {'r(betweenness)':>15} {'PR vs Deg':>10}")
    print("─"*70)

    large_events = [
        ("charliehebdo-all-rnr-threads",     "Charlie Hebdo"),
        ("ottawashooting-all-rnr-threads",   "Ottawa Shooting"),
        ("germanwings-crash-all-rnr-threads","Germanwings"),
        ("sydneysiege-all-rnr-threads",      "Sydney Siege"),
        ("ferguson-all-rnr-threads",         "Ferguson"),
    ]

    import random
    pr_results = {}
    for event_dir, label in large_events:
        G, roots, threads = load_event(os.path.join(base_dir, event_dir))
        root_set = set(roots)
        nonseed  = set(G.nodes()) - root_set
        U = G.to_undirected()

        # Sample 2000 nodes
        sample = sorted(nonseed)
        if len(sample) > 2000:
            random.seed(42)
            sample = random.sample(sample, 2000)

        # Centrality measures
        degrees  = {n: U.degree(n) for n in nonseed if n in U}
        pr = nx.pagerank(G, alpha=0.85)  # directed graph PageRank
        k = min(500, U.number_of_nodes())
        bc = nx.betweenness_centrality(U, k=k, normalized=True, seed=42)

        # Impact per node
        impacts = {}
        for n in sample:
            impacts[n] = sum(len(desc_plus_self(tg, n)) for _, tg in threads if n in tg)

        deg_vals = [degrees.get(n, 0) for n in sample]
        pr_vals  = [pr.get(n, 0)      for n in sample]
        bc_vals  = [bc.get(n, 0)      for n in sample]
        imp_vals = [impacts[n]         for n in sample]

        r_deg, _ = spearmanr(deg_vals, imp_vals)
        r_pr,  _ = spearmanr(pr_vals,  imp_vals)
        r_bc,  _ = spearmanr(bc_vals,  imp_vals)

        pr_results[label] = {"r_degree": r_deg, "r_pagerank": r_pr, "r_betweenness": r_bc}
        better = "PageRank" if r_pr > r_deg else "Degree"
        print(f"{label:<22} {r_deg:>10.4f} {r_pr:>12.4f} {r_bc:>15.4f} {better:>10}")

    means = {
        "degree":      np.mean([v["r_degree"]     for v in pr_results.values()]),
        "pagerank":    np.mean([v["r_pagerank"]    for v in pr_results.values()]),
        "betweenness": np.mean([v["r_betweenness"] for v in pr_results.values()]),
    }
    print(f"{'Mean':<22} {means['degree']:>10.4f} {means['pagerank']:>12.4f} {means['betweenness']:>15.4f}")
    print(f"\nPageRank {('outperforms' if means['pagerank'] > means['degree'] else 'does not outperform')} degree in terms of impact correlation.")
    print(f"(Note: PageRank is computed on the directed graph, weighting nodes by")
    print(f" the importance of who replies to them, not just how many.)")

    # ── Greedy oracle (Germanwings — smallest event, fastest) ──────────────────
    print("\n\nGreedy Oracle Upper Bound")
    print("="*65)
    print("Computing greedy oracle for Germanwings (smallest event, O(k*V) feasible)")
    print("and Ottawa Shooting for comparison.")
    print()

    for event_dir, label in [
        ("germanwings-crash-all-rnr-threads","Germanwings"),
        ("ottawashooting-all-rnr-threads",   "Ottawa Shooting"),
    ]:
        G, roots, threads = load_event(os.path.join(base_dir, event_dir))
        root_set = set(roots)
        nonseed  = set(G.nodes()) - root_set
        budget   = int(0.05 * len(nonseed))
        base     = sum(tg.number_of_nodes() for _, tg in threads)
        U = G.to_undirected()

        # One-shot degree
        degrees  = {n: U.degree(n) for n in nonseed if n in U}
        top_deg  = set(sorted(degrees, key=degrees.get, reverse=True)[:budget])
        deg_r    = 100.0*(base - cascade_after_removal(threads, top_deg))/base

        # One-shot oracle (by pre-computed impact)
        impacts  = {n: sum(len(desc_plus_self(tg,n)) for _,tg in threads if n in tg)
                    for n in nonseed}
        top_ora1 = set(sorted(impacts, key=impacts.get, reverse=True)[:budget])
        oracle1_r = 100.0*(base - cascade_after_removal(threads, top_ora1))/base

        # Greedy oracle
        greedy_r = greedy_oracle(threads, root_set, budget, label)

        print(f"  {label}:")
        print(f"    Budget: {budget} nodes  (5% of {len(nonseed):,} non-seed)")
        print(f"    Degree (one-shot):       {deg_r:.2f}%")
        print(f"    Oracle (one-shot):       {oracle1_r:.2f}%  "
              f"(degree achieves {100*deg_r/oracle1_r:.1f}% of one-shot oracle)")
        print(f"    Oracle (greedy):         {greedy_r:.2f}%  "
              f"(degree achieves {100*deg_r/greedy_r:.1f}% of greedy oracle)")
        print(f"    Gap: degree vs greedy oracle = {greedy_r-deg_r:.2f}pp")
        print()

    with open("pagerank_oracle_results.json","w") as f:
        json.dump({"pagerank_correlations": pr_results}, f, indent=2)
    print("Saved to: pagerank_oracle_results.json")
