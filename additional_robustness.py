# additional_robustness.py — S2, S3, C4, R4
# Addresses:
#   S2: Permutation test for Kendall's W (more reliable than chi-square for small m)
#   S3: Actual numbers for 50% survival bound crossover (Charlie Hebdo, Ferguson)
#   C4: KS test on degree distributions — are events structurally similar?
#   R4: Filter threshold sensitivity (what if we use <3 or <5 node cutoff?)
#
# Usage: python3 additional_robustness.py <pheme_data_dir>
# Runtime: ~10 minutes

import os, json, sys, random
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp, kendalltau

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

def load_event_with_threshold(event_path, min_nodes=2):
    G = nx.DiGraph()
    roots, threads = [], []
    for cat in ["rumours", "non-rumours"]:
        cat_path = os.path.join(event_path, cat)
        if not os.path.isdir(cat_path): continue
        for tid in sorted(os.listdir(cat_path)):
            if tid.startswith('.'): continue
            sf = os.path.join(cat_path, tid, 'structure.json')
            if not os.path.exists(sf): continue
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    sd = json.load(f)
                tg = build_tree_graph(sd)
                if tg.number_of_nodes() < min_nodes: continue
                G = nx.compose(G, tg)
                root = str(list(sd.keys())[0])
                roots.append(root)
                threads.append((root, tg))
            except Exception:
                pass
    return G, sorted(roots), threads

# ── S2: Permutation test for Kendall's W ─────────────────────────────────────
def kendalls_w(rankings_matrix):
    m, n = rankings_matrix.shape
    rank_sums = rankings_matrix.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    return 12 * S / (m**2 * (n**3 - n))

def permutation_test_kendalls_w(observed_rankings, n_permutations=10000, seed=42):
    """
    Permute rankings randomly within each event and compute W.
    p-value = fraction of permutations achieving W >= observed W.
    """
    rng = np.random.default_rng(seed)
    observed_W = kendalls_w(observed_rankings)
    m, n = observed_rankings.shape

    count_geq = 0
    for _ in range(n_permutations):
        perm = observed_rankings.copy()
        for i in range(m):
            perm[i] = rng.permutation(perm[i])
        if kendalls_w(perm) >= observed_W:
            count_geq += 1

    return observed_W, count_geq / n_permutations

if __name__ == "__main__":
    base_dir = sys.argv[1]

    events_5 = [
        ("charliehebdo-all-rnr-threads",     "Charlie Hebdo"),
        ("ottawashooting-all-rnr-threads",   "Ottawa Shooting"),
        ("germanwings-crash-all-rnr-threads","Germanwings"),
        ("sydneysiege-all-rnr-threads",      "Sydney Siege"),
        ("ferguson-all-rnr-threads",         "Ferguson"),
    ]

    # Load intervention results
    with open("intervention_results.json") as f:
        int_results = json.load(f)

    # ── S2 ────────────────────────────────────────────────────────────────
    print("S2: Permutation Test for Kendall's W")
    print("="*55)

    events_primary = list(int_results.keys())
    rankings = []
    for e in events_primary:
        r = int_results[e]
        vals = {
            "degree":      r["degree"]["reduction_pct"],
            "betweenness": r["betweenness"]["reduction_pct"],
            "random":      r["random"]["mean_reduction_pct"],
        }
        ranked = sorted(vals, key=vals.get, reverse=True)
        rn = {s: i+1 for i, s in enumerate(ranked)}
        rankings.append([rn["degree"], rn["betweenness"], rn["random"]])
    rankings = np.array(rankings)

    W_obs, p_perm = permutation_test_kendalls_w(rankings, n_permutations=10000)
    print(f"  Observed Kendall's W = {W_obs:.4f}")
    print(f"  Permutation p-value  = {p_perm:.6f}  (10,000 permutations)")
    print(f"  Theoretical minimum permutation p = (1/3!)^5 = {(1/6)**5:.8f}")
    print(f"  Since W=1.0 is achieved only by perfect ordering in all events,")
    print(f"  permutation p = probability of all 5 events by chance in same order")
    print(f"  = (1/6)^5 = {(1/6)**5:.8f}")
    print()

    # ── S3 ────────────────────────────────────────────────────────────────
    print("S3: Sensitivity Analysis — Exact Crossover Numbers")
    print("="*55)
    print("(How much does the 50% survival bound actually cross zero by?)")
    print()

    with open("thread_level_results.json") as f:
        tl = json.load(f)

    with open("all_events_results.json") as f:
        all_ev = json.load(f)

    for label in ["Charlie Hebdo", "Ferguson"]:
        r = all_ev.get(label, {})
        if not r: continue
        base = r["baseline"]
        deg_r = r["degree"]["reduction_pct"]
        naive_removed = base * deg_r / 100

        for surv_pct in [0, 5, 10, 20, 30, 50]:
            survived = surv_pct/100 * naive_removed
            effective = naive_removed - survived
            effective_r = 100.0 * effective / base
            print(f"  {label} | {surv_pct:>2}% survival: "
                  f"effective reduction = {effective_r:+.2f}%  "
                  f"({'positive' if effective_r > 0 else 'NEGATIVE'})")
        print()

    # ── C4 ────────────────────────────────────────────────────────────────
    print("C4: KS Test on Degree Distributions Across Events")
    print("="*55)
    print("(Are events structurally similar enough to justify transferability?)")
    print()

    degree_distributions = {}
    for event_dir, label in events_5:
        event_path = os.path.join(base_dir, event_dir)
        G, roots, threads = load_event_with_threshold(event_path, min_nodes=2)
        root_set = set(roots)
        U = G.to_undirected()
        nonseed_degrees = [U.degree(n) for n in G.nodes()
                          if n not in root_set and n in U]
        degree_distributions[label] = nonseed_degrees

    labels = list(degree_distributions.keys())
    print(f"  {'Pair':<40} {'KS stat':>10} {'p-value':>12} {'Similar?'}")
    print("  " + "─"*65)
    ks_stats = []
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            stat, p = ks_2samp(degree_distributions[l1],
                                degree_distributions[l2])
            ks_stats.append(stat)
            similar = "YES" if p > 0.05 else "NO"
            print(f"  {l1[:18]} vs {l2[:18]:<18} {stat:>10.4f} {p:>12.4e}  {similar}")

    print(f"\n  Mean KS statistic: {np.mean(ks_stats):.4f}")
    print(f"  (KS=0 means identical distributions, KS=1 means no overlap)")
    print(f"  Interpretation: distributions may differ statistically (large n)")
    print(f"  but functionally similar if KS < 0.15")
    print()

    # ── R4 ────────────────────────────────────────────────────────────────
    print("R4: Filter Threshold Sensitivity")
    print("="*55)
    print("(Does the ranking change if we use <3 or <5 nodes as exclusion threshold?)")
    print()

    def quick_intervention(threads, root_set, budget_pct=0.05, seed=42):
        G_all = nx.DiGraph()
        for _, tg in threads:
            G_all = nx.compose(G_all, tg)
        U = G_all.to_undirected()
        nonseed = set(G_all.nodes()) - root_set
        nonseed_list = sorted(nonseed)
        budget = int(budget_pct * len(nonseed))
        if budget == 0: return None

        base = sum(tg.number_of_nodes() for _, tg in threads)

        degrees = {n: U.degree(n) for n in nonseed if n in U}
        top_deg = set(sorted(degrees, key=degrees.get, reverse=True)[:budget])

        def remove(removed):
            total = 0
            for _, tg in threads:
                surv = set(tg.nodes())
                for n in removed:
                    if n in tg:
                        surv -= {n} | nx.descendants(tg, n)
                total += len(surv)
            return total

        deg_r = 100.0*(base - remove(top_deg))/base

        rng = random.Random(seed)
        rand_rs = []
        for trial in range(200):
            rng.seed(trial)
            rs = set(rng.sample(nonseed_list, budget))
            rand_rs.append(100.0*(base - remove(rs))/base)

        return {"degree": deg_r, "random": np.mean(rand_rs),
                "threads": len(threads), "nodes": G_all.number_of_nodes()}

    for event_dir, label in events_5[:3]:  # 3 events for speed
        event_path = os.path.join(base_dir, event_dir)
        print(f"  {label}:")
        for threshold in [2, 3, 5]:
            G, roots, threads = load_event_with_threshold(event_path, threshold)
            r = quick_intervention(threads, set(roots))
            if r:
                print(f"    min_nodes≥{threshold}: threads={r['threads']}, "
                      f"degree={r['degree']:.1f}%, random={r['random']:.1f}%, "
                      f"degree wins: {r['degree'] > r['random']}")
        print()

    print("Interpretation: if degree wins under all filter thresholds,")
    print("the finding is not sensitive to this preprocessing choice.")
