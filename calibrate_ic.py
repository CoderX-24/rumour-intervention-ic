# calibrate_ic.py
# Stage 2: Independent Cascade model calibration
#
# Fits transmission probability p per event by matching the simulated
# cascade size distribution to the real one using Kolmogorov-Smirnov
# distance. This is more appropriate than R² on a time curve because:
#   1. IC is parameter-free regarding time — spread is driven by network
#      structure, not a global clock
#   2. KS distance compares full distributions, not just means
#   3. Avoids the simultaneous-seed-activation problem that invalidated SIR
#
# Output: calibration_results.json (used by mitigation_ic.py)
#
# Usage: python3 calibrate_ic.py <pheme_data_dir>

import os, json, sys, random
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp

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
    thread_sizes = []
    threads = []  # list of (root, thread_graph)

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
                thread_sizes.append(n)
                threads.append((root, thread_graph))
            except Exception:
                pass

    return G, sorted(roots), thread_sizes, threads

def run_ic_single_thread(thread_graph, root, p, seed=None):
    """
    Run IC on a single thread graph starting from root.
    Returns number of nodes activated (cascade size).
    Each active node tries each inactive neighbour once with probability p.
    """
    if seed is not None:
        random.seed(seed)

    U = thread_graph.to_undirected()
    if root not in U:
        return 1

    activated = {root}
    frontier = [root]

    while frontier:
        next_frontier = []
        for node in sorted(frontier):
            for neighbor in sorted(U.neighbors(node)):
                if neighbor not in activated:
                    if random.random() < p:
                        activated.add(neighbor)
                        next_frontier.append(neighbor)
        frontier = next_frontier

    return len(activated)

def simulate_cascade_sizes(threads, p, n_runs=200):
    """
    Simulate IC across all threads n_runs times.
    Returns array of all simulated cascade sizes (one per thread per run).
    """
    all_sizes = []
    for run in range(n_runs):
        for i, (root, tg) in enumerate(threads):
            size = run_ic_single_thread(tg, root, p, seed=run * 10000 + i)
            all_sizes.append(size)
    return np.array(all_sizes)

def calibrate(threads, real_sizes, label):
    """
    Grid search over p in [0.01, 0.99].
    Minimise KS statistic between simulated and real cascade size distributions.
    """
    real_sizes = np.array(real_sizes)
    p_grid = np.linspace(0.01, 0.99, 100)

    best_p = None
    best_ks = np.inf
    best_mean_sim = None
    results = []

    print(f"\n  Grid search (100 values of p)...")
    for p in p_grid:
        sim_sizes = simulate_cascade_sizes(threads, p, n_runs=50)
        ks_stat, _ = ks_2samp(real_sizes, sim_sizes)
        mean_sim = np.mean(sim_sizes)
        results.append((p, ks_stat, mean_sim))
        if ks_stat < best_ks:
            best_ks = ks_stat
            best_p = p
            best_mean_sim = mean_sim

    # Report
    real_mean   = np.mean(real_sizes)
    real_median = np.median(real_sizes)
    real_max    = np.max(real_sizes)

    print(f"\n  {label}")
    print(f"  {'─'*50}")
    print(f"  Best p:                {best_p:.4f}")
    print(f"  KS statistic:          {best_ks:.4f}  {'[PASS]' if best_ks < 0.10 else '[FAIL — distributions differ]'}")
    print(f"  Real cascade size:     mean={real_mean:.1f}, median={real_median:.1f}, max={real_max}")
    print(f"  Simulated cascade size:mean={best_mean_sim:.1f}")
    print(f"  Ratio (sim/real):      {best_mean_sim/real_mean:.3f}")

    return {
        "label":        label,
        "best_p":       float(best_p),
        "ks_stat":      float(best_ks),
        "passes":       bool(best_ks < 0.10),
        "real_mean":    float(real_mean),
        "real_median":  float(real_median),
        "real_max":     int(real_max),
        "sim_mean":     float(best_mean_sim),
        "ratio":        float(best_mean_sim / real_mean),
        "all_results":  [(float(p), float(ks), float(m)) for p, ks, m in results]
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

    print("Independent Cascade Calibration")
    print("Metric: Kolmogorov-Smirnov distance on cascade size distribution")
    print("Pass threshold: KS < 0.10")
    print("="*60)

    calibration_results = {}

    for event_dir, label in events:
        event_path = os.path.join(base, event_dir)
        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, thread_sizes, threads = load_event(event_path)
        print(f"done ({len(threads)} threads)")
        result = calibrate(threads, thread_sizes, label)
        calibration_results[label] = result

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Event':<20} {'p*':>6} {'KS':>8} {'Ratio':>8} {'Pass?':>8}")
    print("─"*55)
    for label, r in calibration_results.items():
        print(f"{label:<20} {r['best_p']:>6.4f} {r['ks_stat']:>8.4f} "
              f"{r['ratio']:>8.3f} {'YES' if r['passes'] else 'NO':>8}")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration_results.json"
    )
    with open(out_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    print(f"\nCalibration results saved to: {out_path}")
    print("Run mitigation_ic.py next.")
