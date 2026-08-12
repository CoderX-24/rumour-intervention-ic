# partial_tree_intervention.py
# Tests whether degree-centrality targeting works at realistic intervention
# timing windows — not just on the final fully-observed cascade.
#
# Method: for each thread, extract timestamps for every node. Sort nodes
# chronologically. At X% completion (10%, 25%, 50%, 75%, 100%), build the
# partial graph using only nodes timestamped up to that percentile.
# Select the top-k non-seed nodes by degree on the PARTIAL graph.
# Measure cascade reduction on the FULL tree (since that is what actually
# spreads if no intervention occurs).
#
# This directly tests the operational claim: does degree computed on a
# partial cascade identify the same high-impact nodes as degree computed
# on the final cascade?
#
# Usage: python3 partial_tree_intervention.py <pheme_data_dir>
# Runtime: ~25-35 minutes

import os, json, sys, random
from datetime import datetime
import networkx as nx
import numpy as np
from scipy.stats import sem

def parse_ts(s):
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None

def get_node_timestamps(thread_path, root_node):
    """Return dict: node_id -> datetime for all nodes in thread."""
    ts_map = {}
    # Source tweet
    for folder in ["source-tweet", "source-tweets"]:
        d = os.path.join(thread_path, folder)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(d, f)) as fh:
                            t = json.load(fh)
                        nid = str(t.get("id") or t.get("id_str", ""))
                        ts  = parse_ts(t.get("created_at",""))
                        if nid and ts:
                            ts_map[nid] = ts
                    except Exception:
                        pass
    # Reactions
    rd = os.path.join(thread_path, "reactions")
    if os.path.isdir(rd):
        for f in os.listdir(rd):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(rd, f)) as fh:
                        t = json.load(fh)
                    nid = str(t.get("id") or t.get("id_str",""))
                    ts  = parse_ts(t.get("created_at",""))
                    if nid and ts:
                        ts_map[nid] = ts
                except Exception:
                    pass
    return ts_map

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

def load_event_with_timestamps(event_path):
    """Load event graphs plus per-node timestamps."""
    threads_full = []   # (root, full_graph, ts_map)
    for cat in ["rumours", "non-rumours"]:
        cat_path = os.path.join(event_path, cat)
        if not os.path.isdir(cat_path): continue
        for tid in sorted(os.listdir(cat_path)):
            if tid.startswith('.'): continue
            thread_path = os.path.join(cat_path, tid)
            sf = os.path.join(thread_path, 'structure.json')
            if not os.path.exists(sf): continue
            try:
                with open(sf) as f:
                    sd = json.load(f)
                tg = build_tree_graph(sd)
                if tg.number_of_nodes() < 2: continue
                root = str(list(sd.keys())[0])
                ts_map = get_node_timestamps(thread_path, root)
                # Only keep threads where we have timestamps for ≥50% of nodes
                ts_coverage = sum(1 for n in tg.nodes() if n in ts_map)
                if ts_coverage < max(2, 0.5 * tg.number_of_nodes()):
                    continue
                threads_full.append((root, tg, ts_map))
            except Exception:
                pass
    return threads_full

def desc_plus_self(tg, node):
    if node not in tg: return set()
    return {node} | nx.descendants(tg, node)

def cascade_after_removal(threads_full, removed_set):
    return sum(
        len(set(tg.nodes()) - set().union(
            *([desc_plus_self(tg, n) for n in removed_set if n in tg] or [set()])
        ))
        for _, tg, _ in threads_full
    )

def intervention_at_completion(threads_full, root_set, budget_pct, completion_pct, n_rand=500):
    """
    Simulate intervention when cascade is `completion_pct`% complete.
    - Build partial graph: only nodes timestamped in the first completion_pct% of the time range
    - Select top-k by degree on partial graph
    - Measure reduction on FULL tree
    """
    base = sum(tg.number_of_nodes() for _, tg, _ in threads_full)

    # Determine per-thread time cutoffs
    partial_available = set()  # nodes visible at this completion pct
    partial_nonseed   = set()

    for root, tg, ts_map in threads_full:
        # Get timestamps for nodes that have them
        node_times = [(n, ts_map[n]) for n in tg.nodes() if n in ts_map]
        if len(node_times) < 2:
            # No useful timestamps — include all nodes (conservative)
            partial_available.update(tg.nodes())
            partial_nonseed.update(n for n in tg.nodes() if n != root)
            continue

        node_times.sort(key=lambda x: x[1])
        t_start = node_times[0][1]
        t_end   = node_times[-1][1]
        duration = (t_end - t_start).total_seconds()

        if duration == 0 or completion_pct >= 1.0:
            # Include all nodes
            visible = set(n for n, _ in node_times)
        else:
            cutoff_seconds = duration * completion_pct
            visible = set(n for n, t in node_times
                         if (t - t_start).total_seconds() <= cutoff_seconds)

        # Always include root (it's the seed, always present)
        visible.add(root)
        partial_available.update(visible)
        partial_nonseed.update(n for n in visible if n != root and n not in root_set)

    if not partial_nonseed:
        return None

    budget = max(1, int(budget_pct * len(partial_nonseed)))

    # Build partial undirected graph for centrality
    G_partial = nx.DiGraph()
    for _, tg, _ in threads_full:
        for u, v in tg.edges():
            if u in partial_available and v in partial_available:
                G_partial.add_edge(u, v)
    U_partial = G_partial.to_undirected()

    # Degree on partial graph — restricted to partial non-seeds
    deg_partial = {n: U_partial.degree(n) for n in partial_nonseed if n in U_partial}
    top_deg = set(sorted(deg_partial, key=deg_partial.get, reverse=True)[:budget])

    # Random from partial non-seeds
    nonseed_list = sorted(partial_nonseed)
    rng = random.Random(42)
    rand_reds = []
    for trial in range(n_rand):
        rng.seed(trial)
        rs = set(rng.sample(nonseed_list, min(budget, len(nonseed_list))))
        rand_reds.append(100.0*(base - cascade_after_removal(threads_full, rs))/base)

    deg_r  = 100.0*(base - cascade_after_removal(threads_full, top_deg))/base
    rand_r = float(np.mean(rand_reds))

    return {
        "completion_pct": completion_pct,
        "partial_nodes_visible": len(partial_available),
        "partial_nonseed_visible": len(partial_nonseed),
        "budget": budget,
        "degree_reduction": deg_r,
        "random_reduction": rand_r,
        "degree_wins": deg_r > rand_r,
        "degree_advantage_pp": deg_r - rand_r,
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

    COMPLETION_PCTS = [0.10, 0.25, 0.50, 0.75, 1.00]
    BUDGET_PCT = 0.05

    print("Partial-Tree Intervention Timing Analysis")
    print("="*70)
    print("Tests whether degree targeting on a partial cascade predicts")
    print("the same high-impact nodes as degree on the full cascade.")
    print()

    all_results = {}

    for event_dir, label in events:
        event_path = os.path.join(base, event_dir)
        print(f"\nLoading {label} with timestamps...", end=" ", flush=True)
        threads_full = load_event_with_timestamps(event_path)
        root_set = set(r for r, _, _ in threads_full)
        print(f"done. {len(threads_full)} threads with timestamps")

        if len(threads_full) < 10:
            print(f"  Too few timestamped threads — skipping")
            continue

        event_results = []
        print(f"  {'Completion':>12} {'Visible':>10} {'Degree':>9} {'Random':>9} {'Advantage':>11} {'Degree wins?'}")
        print(f"  {'─'*65}")

        for pct in COMPLETION_PCTS:
            r = intervention_at_completion(threads_full, root_set, BUDGET_PCT, pct)
            if r:
                event_results.append(r)
                print(f"  {pct*100:>10.0f}%  {r['partial_nodes_visible']:>10,}  "
                      f"{r['degree_reduction']:>8.2f}%  {r['random_reduction']:>8.2f}%  "
                      f"{r['degree_advantage_pp']:>+10.2f}pp  {'YES' if r['degree_wins'] else 'NO'}")

        all_results[label] = event_results

    print("\n\n" + "="*70)
    print("SUMMARY: Does degree targeting win across all timing windows?")
    print("="*70)
    print(f"\n{'Event':<22}", end="")
    for pct in COMPLETION_PCTS:
        print(f"  {pct*100:.0f}%", end="")
    print("  All win?")
    print("─"*70)

    for label, results in all_results.items():
        print(f"{label:<22}", end="")
        all_win = True
        for r in results:
            wins = r["degree_wins"]
            print(f"  {'✓' if wins else '✗'}", end="   ")
            if not wins: all_win = False
        print(f"  {'YES' if all_win else 'NO'}")

    print("\n\nDegree advantage at each timing window (percentage points above random):")
    print(f"\n{'Event':<22}", end="")
    for pct in COMPLETION_PCTS:
        print(f"  {pct*100:.0f}%   ", end="")
    print()
    print("─"*70)
    for label, results in all_results.items():
        print(f"{label:<22}", end="")
        for r in results:
            print(f"  {r['degree_advantage_pp']:>+5.1f}pp", end="")
        print()

    out = "partial_tree_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {out}")
