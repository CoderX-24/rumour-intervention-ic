# build_networks.py
# Stage 1: Network construction and statistics
# Builds reply-tree graphs from PHEME data, labels seed (root) vs non-seed
# nodes, and reports structural statistics per event.
# Output: network_stats.json (used by all subsequent scripts)
#
# Usage: python3 build_networks.py <pheme_data_dir>

import os, json, sys
import networkx as nx
import numpy as np
from collections import defaultdict

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
    """
    Returns:
        G         : merged DiGraph of all threads
        roots     : list of seed/root node IDs
        thread_sizes : list of per-thread cascade sizes (number of nodes)
    """
    G = nx.DiGraph()
    roots = []
    thread_sizes = []

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
                n_nodes = thread_graph.number_of_nodes()
                if n_nodes < 2:
                    continue  # skip trivial threads
                G = nx.compose(G, thread_graph)
                if isinstance(struct_data, dict) and len(struct_data) > 0:
                    root = str(list(struct_data.keys())[0])
                    roots.append(root)
                    thread_sizes.append(n_nodes)
            except Exception:
                pass

    return G, sorted(roots), thread_sizes

def compute_stats(G, roots, thread_sizes, label):
    U = G.to_undirected()
    root_set = set(roots)
    all_nodes = set(G.nodes())
    non_seeds = all_nodes - root_set

    # degree stats on undirected graph
    degrees = dict(U.degree())
    seed_degrees     = [degrees[n] for n in roots     if n in degrees]
    nonseed_degrees  = [degrees[n] for n in non_seeds if n in degrees]

    # tree depth per thread
    depths = []
    for root in roots:
        try:
            length = nx.single_source_shortest_path_length(G, root)
            depths.append(max(length.values()) if length else 0)
        except Exception:
            pass

    # branching factor: mean out-degree of non-leaf nodes
    out_degrees = [d for n, d in G.out_degree() if d > 0]

    stats = {
        "label":               label,
        "n_nodes":             G.number_of_nodes(),
        "n_edges":             G.number_of_edges(),
        "n_threads":           len(roots),
        "n_seed_nodes":        len(roots),
        "n_nonseed_nodes":     len(non_seeds),
        "nonseed_pct":         100.0 * len(non_seeds) / max(G.number_of_nodes(), 1),
        "thread_size_mean":    float(np.mean(thread_sizes)),
        "thread_size_median":  float(np.median(thread_sizes)),
        "thread_size_max":     int(np.max(thread_sizes)),
        "thread_size_std":     float(np.std(thread_sizes)),
        "depth_mean":          float(np.mean(depths)) if depths else 0,
        "depth_max":           int(np.max(depths))    if depths else 0,
        "branching_factor":    float(np.mean(out_degrees)) if out_degrees else 0,
        "seed_degree_mean":    float(np.mean(seed_degrees))    if seed_degrees    else 0,
        "nonseed_degree_mean": float(np.mean(nonseed_degrees)) if nonseed_degrees else 0,
        "budget_5pct_total":   int(0.05 * G.number_of_nodes()),
        "budget_5pct_nonseed": int(0.05 * len(non_seeds)),
    }
    return stats

def print_stats(s):
    print(f"\n{'='*60}")
    print(f"  {s['label']}")
    print(f"{'='*60}")
    print(f"  Nodes (total):          {s['n_nodes']:>8,}")
    print(f"  Edges:                  {s['n_edges']:>8,}")
    print(f"  Threads (seeds):        {s['n_threads']:>8,}")
    print(f"  Non-seed nodes:         {s['n_nonseed_nodes']:>8,}  ({s['nonseed_pct']:.1f}% of total)")
    print(f"  Thread size mean:       {s['thread_size_mean']:>8.1f}")
    print(f"  Thread size median:     {s['thread_size_median']:>8.1f}")
    print(f"  Thread size max:        {s['thread_size_max']:>8,}")
    print(f"  Thread size std:        {s['thread_size_std']:>8.1f}")
    print(f"  Mean tree depth:        {s['depth_mean']:>8.2f}")
    print(f"  Max tree depth:         {s['depth_max']:>8,}")
    print(f"  Mean branching factor:  {s['branching_factor']:>8.2f}")
    print(f"  Seed degree mean:       {s['seed_degree_mean']:>8.2f}")
    print(f"  Non-seed degree mean:   {s['nonseed_degree_mean']:>8.2f}")
    print(f"  5% budget (total):      {s['budget_5pct_total']:>8,}  nodes")
    print(f"  5% budget (non-seed):   {s['budget_5pct_nonseed']:>8,}  nodes")

if __name__ == "__main__":
    base = sys.argv[1]

    events = [
        ("charliehebdo-all-rnr-threads",     "Charlie Hebdo"),
        ("ottawashooting-all-rnr-threads",   "Ottawa Shooting"),
        ("germanwings-crash-all-rnr-threads","Germanwings"),
        ("sydneysiege-all-rnr-threads",      "Sydney Siege"),
        ("ferguson-all-rnr-threads",         "Ferguson"),
    ]

    all_stats = {}

    for event_dir, label in events:
        event_path = os.path.join(base, event_dir)
        print(f"Loading {label}...", end=" ", flush=True)
        G, roots, thread_sizes = load_event(event_path)
        print(f"done ({G.number_of_nodes():,} nodes)")
        stats = compute_stats(G, roots, thread_sizes, label)
        print_stats(stats)
        all_stats[label] = stats

    # Save for use by subsequent scripts
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network_stats.json")
    with open(out_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n\nNetwork statistics saved to: {out_path}")
    print("Run calibrate_ic.py next.")
