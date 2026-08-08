# compute_intervention_extended.py
# Extended intervention analysis adding:
#   - Charlie Hebdo rumours-only subgraph
#   - Charlie Hebdo non-rumours-only subgraph
#   - Ferguson Wave 1 threads (root tweets posted in first 30 hours)
#   - Ferguson Wave 3 threads (root tweets posted in hours 132-150)
#
# These add 4 semi-independent contexts to the primary 5, strengthening
# the transferability finding. Non-independence is acknowledged: CH splits
# and Ferguson waves share their parent event's network structure.
#
# Usage: python3 compute_intervention_extended.py <pheme_data_dir>
# Runtime: ~25-35 minutes

import os, json, sys, random
from datetime import datetime
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

def get_root_timestamp(thread_path):
    """Get the created_at timestamp of the root/source tweet."""
    for folder in ["source-tweet", "source-tweets"]:
        d = os.path.join(thread_path, folder)
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(d, fname)) as f:
                            tweet = json.load(f)
                        return datetime.strptime(
                            tweet["created_at"], "%a %b %d %H:%M:%S %z %Y"
                        )
                    except Exception:
                        pass
    return None

def load_event_filtered(event_path, categories=None, wave_hours=None):
    """
    Load event graph filtered by:
      categories: list of subdirs to include e.g. ["rumours"] or ["non-rumours"]
      wave_hours: tuple (start_h, end_h) to filter by root tweet timestamp
    Returns G, roots, threads
    """
    G = nx.DiGraph()
    roots = []
    threads = []

    if categories is None:
        categories = ["rumours", "non-rumours"]

    # For wave filtering, find the earliest timestamp across ALL threads
    # (same reference t0 as in the full event)
    t0 = None
    if wave_hours is not None:
        full_path = event_path
        for cat in ["rumours", "non-rumours"]:
            cat_path = os.path.join(full_path, cat)
            if not os.path.isdir(cat_path):
                continue
            for tid in sorted(os.listdir(cat_path)):
                if tid.startswith('.'): continue
                ts = get_root_timestamp(os.path.join(cat_path, tid))
                if ts and (t0 is None or ts < t0):
                    t0 = ts

    for category in categories:
        cat_path = os.path.join(event_path, category)
        if not os.path.isdir(cat_path):
            continue
        for thread_id in sorted(os.listdir(cat_path)):
            if thread_id.startswith('.'): continue
            thread_path = os.path.join(cat_path, thread_id)
            struct_file = os.path.join(thread_path, 'structure.json')
            if not os.path.exists(struct_file):
                continue
            try:
                # Wave filter
                if wave_hours is not None and t0 is not None:
                    ts = get_root_timestamp(thread_path)
                    if ts is None:
                        continue
                    hours = (ts - t0).total_seconds() / 3600
                    if not (wave_hours[0] <= hours <= wave_hours[1]):
                        continue

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

def run_intervention(threads, root_set, nonseed_nodes,
                     budget, label, n_random=1000):
    base = baseline_cascade_size(threads)
    nonseed_list = sorted(nonseed_nodes)
    actual_budget = min(budget, len(nonseed_list))

    if actual_budget == 0 or base == 0:
        print(f"    SKIP — insufficient nodes")
        return None

    # Degree
    G_all = nx.DiGraph()
    for _, tg in threads:
        G_all = nx.compose(G_all, tg)
    U_all = G_all.to_undirected()
    degrees = {n: U_all.degree(n) for n in nonseed_nodes if n in U_all}
    top_degree = set(sorted(degrees, key=degrees.get, reverse=True)[:actual_budget])
    deg_size = cascade_size_after_removal(threads, top_degree)
    deg_r = 100.0 * (base - deg_size) / base

    # Betweenness
    k = min(500, U_all.number_of_nodes())
    bc = nx.betweenness_centrality(U_all, k=k, normalized=True, seed=42)
    bc_nonseed = {n: bc.get(n,0) for n in nonseed_nodes}
    top_bet = set(sorted(bc_nonseed, key=bc_nonseed.get, reverse=True)[:actual_budget])
    bet_size = cascade_size_after_removal(threads, top_bet)
    bet_r = 100.0 * (base - bet_size) / base

    # Random
    rng = random.Random(42)
    rand_reds = []
    for trial in range(n_random):
        rng.seed(trial)
        rand_set = set(rng.sample(nonseed_list, actual_budget))
        rand_reds.append(100.0*(base - cascade_size_after_removal(threads, rand_set))/base)
    rand_mean = float(np.mean(rand_reds))
    rand_ci   = float(1.96 * sem(rand_reds))

    ranking = sorted(["degree","betweenness","random"],
                     key={"degree":deg_r,"betweenness":bet_r,"random":rand_mean}.get,
                     reverse=True)

    print(f"    Degree: {deg_r:.2f}%  Betweenness: {bet_r:.2f}%  "
          f"Random: {rand_mean:.2f}%  → {'>'.join(ranking)}")

    return {
        "label": label,
        "n_threads": len(threads),
        "baseline": base,
        "budget": actual_budget,
        "budget_pct": 100.0 * actual_budget / len(nonseed_nodes),
        "degree":      {"reduction_pct": deg_r},
        "betweenness": {"reduction_pct": bet_r},
        "random":      {"mean_reduction_pct": rand_mean, "ci_95": rand_ci},
        "ranking": ranking,
        "degree_wins": ranking[0] == "degree",
    }

if __name__ == "__main__":
    base = sys.argv[1]

    contexts = []

    # ── Charlie Hebdo: rumours only ───────────────────────────────────────────
    print("\nLoading Charlie Hebdo (rumours only)...", end=" ", flush=True)
    G, roots, threads = load_event_filtered(
        os.path.join(base, "charliehebdo-all-rnr-threads"),
        categories=["rumours"]
    )
    nonseed = set(G.nodes()) - set(roots)
    budget  = int(0.05 * len(nonseed))
    print(f"done. {len(threads)} threads, {len(nonseed):,} non-seed, budget={budget}")
    r = run_intervention(threads, set(roots), nonseed, budget,
                         "CH Rumours-only")
    if r: contexts.append(r)

    # ── Charlie Hebdo: non-rumours only ──────────────────────────────────────
    print("\nLoading Charlie Hebdo (non-rumours only)...", end=" ", flush=True)
    G, roots, threads = load_event_filtered(
        os.path.join(base, "charliehebdo-all-rnr-threads"),
        categories=["non-rumours"]
    )
    nonseed = set(G.nodes()) - set(roots)
    budget  = int(0.05 * len(nonseed))
    print(f"done. {len(threads)} threads, {len(nonseed):,} non-seed, budget={budget}")
    r = run_intervention(threads, set(roots), nonseed, budget,
                         "CH Non-rumours-only")
    if r: contexts.append(r)

    # ── Ferguson Wave 1: root tweets in first 30 hours ────────────────────────
    print("\nLoading Ferguson Wave 1 (0–30h)...", end=" ", flush=True)
    G, roots, threads = load_event_filtered(
        os.path.join(base, "ferguson-all-rnr-threads"),
        wave_hours=(0, 30)
    )
    nonseed = set(G.nodes()) - set(roots)
    budget  = int(0.05 * len(nonseed))
    print(f"done. {len(threads)} threads, {len(nonseed):,} non-seed, budget={budget}")
    r = run_intervention(threads, set(roots), nonseed, budget,
                         "Ferguson Wave 1")
    if r: contexts.append(r)

    # ── Ferguson Wave 3: root tweets in hours 132–150 ─────────────────────────
    print("\nLoading Ferguson Wave 3 (132–150h)...", end=" ", flush=True)
    G, roots, threads = load_event_filtered(
        os.path.join(base, "ferguson-all-rnr-threads"),
        wave_hours=(132, 150)
    )
    nonseed = set(G.nodes()) - set(roots)
    budget  = int(0.05 * len(nonseed))
    print(f"done. {len(threads)} threads, {len(nonseed):,} non-seed, budget={budget}")
    r = run_intervention(threads, set(roots), nonseed, budget,
                         "Ferguson Wave 3")
    if r: contexts.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("EXTENDED CONTEXTS SUMMARY")
    print("="*65)
    print(f"{'Context':<22} {'Threads':>8} {'Degree':>9} {'Betweenness':>13} {'Random':>9}  Ranking")
    print("─"*75)
    for c in contexts:
        print(f"{c['label']:<22} {c['n_threads']:>8} "
              f"{c['degree']['reduction_pct']:>8.2f}% "
              f"{c['betweenness']['reduction_pct']:>12.2f}% "
              f"{c['random']['mean_reduction_pct']:>8.2f}%  "
              f"{'>'.join(c['ranking'])}")

    degree_wins = sum(1 for c in contexts if c["degree_wins"])
    print(f"\nDegree ranked first: {degree_wins}/{len(contexts)} extended contexts")

    # Combined binomial test (all 9 contexts = 5 primary + 4 extended)
    total_contexts = 5 + len(contexts)
    total_wins     = 5 + degree_wins   # 5 primary all won
    p_combined = binomtest(total_wins, total_contexts, p=1/3,
                           alternative='greater').pvalue
    print(f"Combined (5 primary + {len(contexts)} extended): "
          f"{total_wins}/{total_contexts} degree wins, "
          f"binomial p = {p_combined:.8f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "intervention_results_extended.json")
    with open(out, "w") as f:
        json.dump(contexts, f, indent=2)
    print(f"\nSaved to: {out}")
    print("Run transfer_analysis_extended.py next.")
