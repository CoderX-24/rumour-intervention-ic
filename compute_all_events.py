# compute_all_events.py
# Runs cascade truncation intervention on ALL 9 PHEME events.
# Events previously excluded under SIR (calibration failures) are included
# here because cascade truncation requires no calibration.
# Events with <50 threads are reported but flagged as underpowered.
#
# Explicit exclusion criteria (stated upfront):
#   - Gurlitt: excluded — data quality issues documented in original dataset
#   - Ebola-Essien: included despite 14 threads (flagged as underpowered)
#   - Prince Toronto: included
#   - Putin Missing: included
#
# Usage: python3 compute_all_events.py <pheme_data_dir>
# Runtime: ~45-60 minutes

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
            if thread_id.startswith('.'): continue
            struct_file = os.path.join(cat_path, thread_id, 'structure.json')
            if not os.path.exists(struct_file): continue
            try:
                with open(struct_file, 'r', encoding='utf-8') as f:
                    struct_data = json.load(f)
                tg = build_tree_graph(struct_data)
                if tg.number_of_nodes() < 2: continue
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

def run_intervention(threads, root_set, nonseed_nodes, budget, label,
                     n_random=1000, underpowered=False):
    base = baseline_cascade_size(threads)
    nonseed_list = sorted(nonseed_nodes)
    actual_budget = min(budget, len(nonseed_list))

    if actual_budget == 0 or base == 0 or len(threads) < 10:
        print(f"    SKIP — too few threads ({len(threads)})")
        return None

    # Degree — on full undirected graph, non-seeds only
    G_all = nx.DiGraph()
    for _, tg in threads:
        G_all = nx.compose(G_all, tg)
    U_full = G_all.to_undirected()

    # M5 fix: betweenness computed on FULL graph (including seeds)
    # but budget applied to non-seed nodes only.
    # This is explicitly stated and justified: seeds are structurally central
    # and their betweenness scores correctly reflect their position;
    # excluding them from the budget means we never select them.
    degrees_nonseed = {n: U_full.degree(n) for n in nonseed_nodes if n in U_full}
    top_degree = set(sorted(degrees_nonseed, key=degrees_nonseed.get,
                             reverse=True)[:actual_budget])
    deg_size = cascade_size_after_removal(threads, top_degree)
    deg_r = 100.0 * (base - deg_size) / base

    k = min(500, U_full.number_of_nodes())
    bc_full = nx.betweenness_centrality(U_full, k=k, normalized=True, seed=42)
    bc_nonseed = {n: bc_full.get(n, 0) for n in nonseed_nodes}
    top_bet = set(sorted(bc_nonseed, key=bc_nonseed.get, reverse=True)[:actual_budget])
    bet_size = cascade_size_after_removal(threads, top_bet)
    bet_r = 100.0 * (base - bet_size) / base

    # Random — fresh sample each trial; reduction is always >= 0 by construction
    rng = random.Random(42)
    rand_reds = []
    for trial in range(n_random):
        rng.seed(trial)
        rand_set = set(rng.sample(nonseed_list, actual_budget))
        size = cascade_size_after_removal(threads, rand_set)
        reduction = 100.0 * (base - size) / base
        # Sanity check: cascade truncation is monotone — reduction must be >= 0
        assert reduction >= 0, f"Negative reduction detected: {reduction}"
        rand_reds.append(reduction)

    rand_mean = float(np.mean(rand_reds))
    rand_ci   = float(1.96 * sem(rand_reds))
    rand_min  = float(np.min(rand_reds))
    rand_max  = float(np.max(rand_reds))

    ranking = sorted(["degree","betweenness","random"],
                     key={"degree":deg_r,"betweenness":bet_r,
                          "random":rand_mean}.get,
                     reverse=True)

    flag = " [UNDERPOWERED — <50 threads]" if underpowered else ""
    print(f"    Degree: {deg_r:.2f}%  Betweenness: {bet_r:.2f}%  "
          f"Random: {rand_mean:.2f}% [{rand_min:.2f},{rand_max:.2f}]"
          f"  → {'>'.join(ranking)}{flag}")

    return {
        "label": label,
        "n_threads": len(threads),
        "n_nodes": G_all.number_of_nodes(),
        "n_nonseed": len(nonseed_nodes),
        "baseline": base,
        "budget": actual_budget,
        "budget_pct": 100.0 * actual_budget / len(nonseed_nodes),
        "underpowered": underpowered,
        "degree":      {"reduction_pct": deg_r},
        "betweenness": {"reduction_pct": bet_r},
        "random": {
            "mean_reduction_pct": rand_mean,
            "ci_95": rand_ci,
            "min": rand_min,
            "max": rand_max,
        },
        "ranking": ranking,
        "degree_wins": ranking[0] == "degree",
    }

if __name__ == "__main__":
    base = sys.argv[1]

    # All 9 PHEME events with explicit inclusion/exclusion justification
    events = [
        # Previously used (5 primary)
        ("charliehebdo-all-rnr-threads",      "Charlie Hebdo",    False, True),
        ("ottawashooting-all-rnr-threads",    "Ottawa Shooting",  False, True),
        ("germanwings-crash-all-rnr-threads", "Germanwings",      False, True),
        ("sydneysiege-all-rnr-threads",       "Sydney Siege",     False, True),
        ("ferguson-all-rnr-threads",          "Ferguson",         False, True),
        # Previously excluded under SIR (calibration failure) — now included
        ("prince-toronto-all-rnr-threads",    "Prince Toronto",   False, True),
        ("putinmissing-all-rnr-threads",      "Putin Missing",    False, True),
        # Small — included but flagged
        ("ebola-essien-all-rnr-threads",      "Ebola-Essien",     True,  True),
        # Excluded — data quality (documented)
        # gurlitt-all-rnr-threads: excluded — see Zubiaga et al. (2016) quality notes
    ]

    print("Cascade Truncation — All Available PHEME Events")
    print("Betweenness computed on full graph, budget restricted to non-seed nodes")
    print("="*70)
    print("NOTE: Gurlitt excluded due to documented data quality issues.")
    print("      All other events included regardless of prior SIR calibration outcome.")
    print()

    all_results = {}
    included = []

    for event_dir, label, underpowered, include in events:
        event_path = os.path.join(base, event_dir)
        if not os.path.isdir(event_path):
            print(f"\n{label}: directory not found — skipping")
            continue
        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, threads = load_event(event_path)
        root_set = set(roots)
        nonseed  = set(G.nodes()) - root_set
        budget   = int(0.05 * len(nonseed))
        print(f"done. Threads: {len(threads)}, Non-seed: {len(nonseed):,}, Budget: {budget}")

        r = run_intervention(threads, root_set, nonseed, budget, label,
                             underpowered=underpowered)
        if r:
            all_results[label] = r
            if not underpowered:
                included.append(r)

    # ── Results table ──────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("ALL EVENTS RESULTS")
    print("="*70)
    print(f"{'Event':<22} {'Threads':>8} {'Degree':>9} {'Betweenness':>13} "
          f"{'Random':>9} {'Ranking'}")
    print("─"*75)
    for label, r in all_results.items():
        flag = "*" if r["underpowered"] else " "
        print(f"{flag}{label:<21} {r['n_threads']:>8} "
              f"{r['degree']['reduction_pct']:>8.2f}% "
              f"{r['betweenness']['reduction_pct']:>12.2f}% "
              f"{r['random']['mean_reduction_pct']:>8.2f}%  "
              f"{'>'.join(r['ranking'])}")
    print("* = underpowered (<50 threads)")

    # ── Primary transferability test (excluding underpowered) ─────────────
    print("\n\nTRANSFERABILITY TEST (powered events only)")
    print("─"*50)
    wins = sum(1 for r in included if r["degree_wins"])
    n    = len(included)
    p    = binomtest(wins, n, p=1/3, alternative='greater').pvalue
    print(f"Degree wins: {wins}/{n}")
    print(f"Binomial p  = {p:.8f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "all_events_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {out}")
