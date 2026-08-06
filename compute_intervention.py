# compute_intervention.py
# Stage 2: Cascade truncation intervention analysis
#
# Uses real observed reply trees as ground truth — no spreading model needed.
# For each non-seed node, removing it removes it and all its descendants.
# This gives the exact cascade size reduction that would have resulted from
# removing that node before the event.
#
# Three strategies are compared on non-seed nodes only:
#   1. Random removal      (fresh sample, 1000 trials for CI)
#   2. Degree centrality   (top k non-seed nodes by undirected degree)
#   3. Betweenness centrality (top k non-seed nodes by betweenness)
#
# Output: intervention_results.json (used by transfer_analysis.py)
#
# Usage: python3 compute_intervention.py <pheme_data_dir>

import os, json, sys, random
import networkx as nx
import numpy as np
from scipy.stats import sem

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
    roots = []
    threads = []

    for category in ["rumours", "non-rumours"]:
        cat_path = os.path.join(event_path, category)
        if not os.path.isdir(cat_path):
            continue
        for thread_id in sorted(os.listdir(cat_path)):
            if thread_id.startswith('.'):
                continue
            struct_file = os.path.join(cat_path, thread_id, 'structure.json')
            if not os.path.exists(struct_file):
                continue
            try:
                with open(struct_file, 'r', encoding='utf-8') as f:
                    struct_data = json.load(f)
                thread_graph = build_tree_graph(struct_data)
                n = thread_graph.number_of_nodes()
                if n < 2:
                    continue
                G = nx.compose(G, thread_graph)
                root = str(list(struct_data.keys())[0])
                roots.append(root)
                threads.append((root, thread_graph))
            except Exception:
                pass

    return G, sorted(roots), threads

def descendants_plus_self(tree, node):
    """
    In a directed reply tree, returns node and all nodes reachable from it.
    These are the nodes that would be removed if this node were removed.
    """
    if node not in tree:
        return set()
    return {node} | nx.descendants(tree, node)

def compute_node_impacts(threads, root_set):
    """
    For every non-seed node across all threads, compute how many nodes
    would be removed (node + descendants) if it were removed.
    Returns dict: node_id -> impact (int)
    """
    impacts = {}
    for root, tg in threads:
        for node in tg.nodes():
            if node in root_set:
                continue
            if node not in impacts:
                desc = descendants_plus_self(tg, node)
                impacts[node] = len(desc)
    return impacts

def cascade_size_after_removal(threads, removed_set):
    """
    Total cascade size across all threads after removing a set of nodes.
    Removing a node removes it and all its descendants in each thread.
    """
    total = 0
    for root, tg in threads:
        surviving = set(tg.nodes())
        for node in removed_set:
            if node in tg:
                surviving -= descendants_plus_self(tg, node)
        total += len(surviving)
    return total

def baseline_cascade_size(threads):
    return sum(tg.number_of_nodes() for _, tg in threads)

def run_intervention(threads, root_set, nonseed_nodes,
                     budget, label, n_random_trials=1000):
    """
    Run all three strategies at a given budget.
    Budget is number of non-seed nodes to remove.
    """
    base = baseline_cascade_size(threads)
    nonseed_list = sorted(nonseed_nodes)

    # ── Strategy 1: Degree centrality (non-seeds only) ────────────────────
    print(f"    Computing degree centrality...", end=" ", flush=True)
    # Build undirected graph from all threads
    G_all = nx.DiGraph()
    for _, tg in threads:
        G_all = nx.compose(G_all, tg)
    U_all = G_all.to_undirected()
    degrees = {n: U_all.degree(n) for n in nonseed_nodes if n in U_all}
    top_degree = set(sorted(degrees, key=degrees.get, reverse=True)[:budget])
    deg_size = cascade_size_after_removal(threads, top_degree)
    deg_reduction = 100.0 * (base - deg_size) / base
    print(f"done. Reduction: {deg_reduction:.2f}%")

    # ── Strategy 2: Betweenness centrality (non-seeds only) ───────────────
    print(f"    Computing betweenness centrality (k=500)...", end=" ", flush=True)
    k = min(500, U_all.number_of_nodes())
    bc = nx.betweenness_centrality(U_all, k=k, normalized=True, seed=42)
    bc_nonseed = {n: bc[n] for n in nonseed_nodes if n in bc}
    top_bet = set(sorted(bc_nonseed, key=bc_nonseed.get, reverse=True)[:budget])
    bet_size = cascade_size_after_removal(threads, top_bet)
    bet_reduction = 100.0 * (base - bet_size) / base
    print(f"done. Reduction: {bet_reduction:.2f}%")

    # ── Strategy 3: Random (fresh sample each trial) ──────────────────────
    print(f"    Running {n_random_trials} random trials...", end=" ", flush=True)
    rand_reductions = []
    rng = random.Random(42)
    for trial in range(n_random_trials):
        rng.seed(trial)
        rand_set = set(rng.sample(nonseed_list, min(budget, len(nonseed_list))))
        rand_size = cascade_size_after_removal(threads, rand_set)
        rand_reductions.append(100.0 * (base - rand_size) / base)
    rand_mean = float(np.mean(rand_reductions))
    rand_ci   = float(1.96 * sem(rand_reductions))
    print(f"done. Mean reduction: {rand_mean:.2f}%")

    # ── Ranking ───────────────────────────────────────────────────────────
    strategies = {
        "degree":      deg_reduction,
        "betweenness": bet_reduction,
        "random":      rand_mean,
    }
    ranking = sorted(strategies, key=strategies.get, reverse=True)

    return {
        "budget":          budget,
        "budget_pct":      100.0 * budget / len(nonseed_nodes),
        "baseline":        base,
        "degree": {
            "reduction_pct": deg_reduction,
            "final_size":    deg_size,
        },
        "betweenness": {
            "reduction_pct": bet_reduction,
            "final_size":    bet_size,
        },
        "random": {
            "mean_reduction_pct": rand_mean,
            "ci_95":              rand_ci,
            "final_size":         base - int(base * rand_mean / 100),
        },
        "ranking": ranking,
        "degree_wins": ranking[0] == "degree",
    }

if __name__ == "__main__":
    base = sys.argv[1]

    events = [
        ("charliehebdo-all-rnr-threads",     "Charlie Hebdo"),
        ("ottawashooting-all-rnr-threads",   "Ottawa Shooting"),
        ("germanwings-crash-all-rnr-threads","Germanwings"),
        ("sydneysiege-all-rnr-threads",      "Sydney Siege"),
        ("ferguson-all-rnr-threads",         "Ferguson"),
    ]

    BUDGET_PCT = 0.05  # primary analysis: 5% of non-seed nodes

    print("Cascade Truncation Intervention Analysis")
    print("Budget: 5% of non-seed nodes")
    print("Strategies: degree, betweenness, random (1000 trials)")
    print("="*60)

    all_results = {}

    for event_dir, label in events:
        event_path = os.path.join(base, event_dir)
        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, threads = load_event(event_path)
        root_set     = set(roots)
        all_nodes    = set(G.nodes())
        nonseed_nodes = all_nodes - root_set
        budget = int(BUDGET_PCT * len(nonseed_nodes))

        print(f"done. Non-seed nodes: {len(nonseed_nodes):,} | Budget: {budget}")
        print(f"  Running interventions:")

        result = run_intervention(
            threads, root_set, nonseed_nodes,
            budget=budget, label=label,
            n_random_trials=1000
        )
        result["label"] = label
        all_results[label] = result

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY (5% non-seed budget)")
    print("="*70)
    print(f"{'Event':<20} {'Degree':>10} {'Betweenness':>13} "
          f"{'Random (mean)':>15} {'Ranking'}")
    print("─"*70)

    for label, r in all_results.items():
        print(f"{label:<20} "
              f"{r['degree']['reduction_pct']:>9.2f}% "
              f"{r['betweenness']['reduction_pct']:>12.2f}% "
              f"{r['random']['mean_reduction_pct']:>13.2f}% "
              f"±{r['random']['ci_95']:.2f}%  "
              f"{' > '.join(r['ranking'])}")

    # ── Ranking consistency ───────────────────────────────────────────────
    print("\n\nRANKING CONSISTENCY")
    print("─"*40)
    degree_wins = sum(1 for r in all_results.values() if r['degree_wins'])
    print(f"Degree ranked 1st: {degree_wins}/{len(all_results)} events")

    from scipy.stats import binomtest
    p_binom = binomtest(degree_wins, len(all_results), p=1/3, alternative='greater')
    print(f"Binomial test (H0: p_win=1/3): p = {p_binom.pvalue:.6f}")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "intervention_results.json"
    )
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("Run transfer_analysis.py next.")