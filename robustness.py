# robustness.py
# Stage 4: Robustness checks
#
# Tests whether the strategy ranking degree > betweenness > random
# holds across different intervention budgets (1%, 2%, 5%, 10%, 20%
# of non-seed nodes). If the ranking is stable across budgets, the
# finding is robust to the choice of budget parameter.
#
# Also tests parameter sensitivity: does the ranking hold when we
# use different random seeds for betweenness approximation?
#
# Output: robustness_results.json
#
# Usage: python3 robustness.py <pheme_data_dir>
# Runtime: ~25-35 minutes

import os, json, sys, random
import networkx as nx
import numpy as np
from scipy.stats import sem, binomtest

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
                tg = build_tree_graph(struct_data)
                if tg.number_of_nodes() < 2:
                    continue
                G = nx.compose(G, tg)
                root = str(list(struct_data.keys())[0])
                roots.append(root)
                threads.append((root, tg))
            except Exception:
                pass
    return G, sorted(roots), threads

def descendants_plus_self(tree, node):
    if node not in tree:
        return set()
    return {node} | nx.descendants(tree, node)

def cascade_size_after_removal(threads, removed_set):
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

def run_at_budget(threads, nonseed_nodes, budget,
                  degrees, bc_nonseed, n_random=500):
    base = baseline_cascade_size(threads)
    nonseed_list = sorted(nonseed_nodes)
    actual_budget = min(budget, len(nonseed_list))

    # Degree
    top_degree = set(sorted(degrees, key=degrees.get,
                             reverse=True)[:actual_budget])
    deg_r = 100.0 * (base - cascade_size_after_removal(
                     threads, top_degree)) / base

    # Betweenness
    top_bet = set(sorted(bc_nonseed, key=bc_nonseed.get,
                          reverse=True)[:actual_budget])
    bet_r = 100.0 * (base - cascade_size_after_removal(
                     threads, top_bet)) / base

    # Random
    rng = random.Random(42)
    rand_reds = []
    for trial in range(n_random):
        rng.seed(trial)
        rand_set = set(rng.sample(nonseed_list, actual_budget))
        rand_reds.append(100.0 * (base - cascade_size_after_removal(
                         threads, rand_set)) / base)
    rand_r = float(np.mean(rand_reds))
    rand_ci = float(1.96 * sem(rand_reds))

    ranking = sorted(
        ["degree", "betweenness", "random"],
        key={"degree": deg_r, "betweenness": bet_r, "random": rand_r}.get,
        reverse=True
    )

    return {
        "degree":      {"reduction_pct": deg_r},
        "betweenness": {"reduction_pct": bet_r},
        "random":      {"mean_reduction_pct": rand_r, "ci_95": rand_ci},
        "ranking":     ranking,
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

    BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]

    print("Robustness Check: Budget Sensitivity Analysis")
    print("Budgets tested: 1%, 2%, 5%, 10%, 20% of non-seed nodes")
    print("="*65)

    all_results = {}

    for event_dir, label in events:
        event_path = os.path.join(base, event_dir)
        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, threads = load_event(event_path)
        root_set      = set(roots)
        all_nodes     = set(G.nodes())
        nonseed_nodes = all_nodes - root_set
        nonseed_list  = sorted(nonseed_nodes)

        print(f"done. Computing centralities...", end=" ", flush=True)
        U_all = G.to_undirected()

        # Degree (non-seeds only)
        degrees = {n: U_all.degree(n) for n in nonseed_nodes if n in U_all}

        # Betweenness (non-seeds only, k=500)
        k = min(500, U_all.number_of_nodes())
        bc = nx.betweenness_centrality(U_all, k=k, normalized=True, seed=42)
        bc_nonseed = {n: bc[n] for n in nonseed_nodes if n in bc}
        print("done.")

        event_results = {}
        for budget_pct in BUDGETS:
            budget = max(1, int(budget_pct * len(nonseed_nodes)))
            print(f"  Budget {budget_pct*100:.0f}% ({budget} nodes)...",
                  end=" ", flush=True)
            r = run_at_budget(threads, nonseed_nodes, budget,
                              degrees, bc_nonseed, n_random=500)
            event_results[f"{budget_pct*100:.0f}pct"] = r
            print(f"deg={r['degree']['reduction_pct']:.1f}%  "
                  f"bet={r['betweenness']['reduction_pct']:.1f}%  "
                  f"ran={r['random']['mean_reduction_pct']:.1f}%  "
                  f"rank: {'>'.join(r['ranking'])}")

        all_results[label] = event_results

    # ── Summary: does ranking hold at every budget? ────────────────────
    print("\n\n" + "="*65)
    print("RANKING STABILITY ACROSS BUDGETS")
    print("="*65)
    print(f"\n{'Budget':<10}", end="")
    for label in [e[1] for e in events]:
        print(f"{label[:12]:<14}", end="")
    print("Degree wins")
    print("─"*65)

    budget_labels = [f"{b*100:.0f}pct" for b in BUDGETS]
    for bl, bp in zip(budget_labels, BUDGETS):
        print(f"{bp*100:.0f}%{'':<8}", end="")
        wins = 0
        for label in [e[1] for e in events]:
            r = all_results[label][bl]
            ranking_str = ">".join([s[:3] for s in r["ranking"]])
            print(f"{ranking_str:<14}", end="")
            if r["degree_wins"]:
                wins += 1
        print(f"{wins}/5")

    # Binomial test at each budget
    print("\nBinomial test at each budget (H0: p_win=1/3):")
    for bl, bp in zip(budget_labels, BUDGETS):
        wins = sum(1 for label in [e[1] for e in events]
                   if all_results[label][bl]["degree_wins"])
        p = binomtest(wins, 5, p=1/3, alternative='greater').pvalue
        print(f"  {bp*100:.0f}%: degree wins {wins}/5, p={p:.4f} "
              f"({'sig' if p < 0.05 else 'ns'})")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "robustness_results.json"
    )
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("Run structural_analysis.py next.")