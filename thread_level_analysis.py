# thread_level_analysis.py — D2, D3, S5, R3
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

def remove_nodes(threads, removed):
    total = 0
    for _, tg in threads:
        surv = set(tg.nodes())
        for n in removed:
            if n in tg:
                surv -= desc_plus_self(tg, n)
        total += len(surv)
    return total

def analyse_event(event_path, label, budget_pct=0.05):
    G, roots, threads = load_event(event_path)
    root_set = set(roots)
    nonseed  = set(G.nodes()) - root_set
    nonseed_list = sorted(nonseed)
    budget   = max(1, int(budget_pct * len(nonseed)))
    U = G.to_undirected()
    base = sum(tg.number_of_nodes() for _, tg in threads)

    # D1
    seed_check = (len(roots) == len(threads))

    # Degree selection
    degrees = {n: U.degree(n) for n in nonseed if n in U}
    top_deg = sorted(degrees, key=degrees.get, reverse=True)[:budget]
    top_deg_set = set(top_deg)

    # S5: overlap check
    wasted = set()
    for _, tg in threads:
        sel = [n for n in top_deg if n in tg]
        for i, n1 in enumerate(sel):
            for n2 in sel[i+1:]:
                if n2 in nx.descendants(tg, n1):
                    wasted.add(n2)
                elif n1 in nx.descendants(tg, n2):
                    wasted.add(n1)
    overlap_pct = 100.0 * len(wasted) / budget

    # D2: budget distribution
    nodes_per_thread = np.array([
        sum(1 for n in top_deg if n in tg) for _, tg in threads
    ])
    threads_receiving = int((nodes_per_thread > 0).sum())
    top10 = int(0.1 * len(threads))
    budget_conc = 100.0 * np.sort(nodes_per_thread)[::-1][:top10].sum() / budget

    # D3: per-thread reduction
    rng = random.Random(42)
    thread_deg_r, thread_rand_r = [], []
    for root, tg in threads:
        sz = tg.number_of_nodes()
        if sz < 2: continue
        tn = [n for n in tg.nodes() if n not in root_set]
        if not tn: continue
        tb = max(1, int(budget_pct * len(tn)))
        # degree
        td = sorted(tn, key=lambda n: degrees.get(n,0), reverse=True)[:tb]
        sv = set(tg.nodes())
        for n in td:
            sv -= desc_plus_self(tg, n)
        thread_deg_r.append(100.0*(sz-len(sv))/sz)
        # random
        rng.seed(abs(hash(root)) % (2**31))
        rs = set(rng.sample(tn, min(tb, len(tn))))
        sv2 = set(tg.nodes())
        for n in rs:
            sv2 -= desc_plus_self(tg, n)
        thread_rand_r.append(100.0*(sz-len(sv2))/sz)

    # R3: oracle
    impacts = {}
    for n in nonseed:
        impacts[n] = sum(len(desc_plus_self(tg, n)) for _, tg in threads if n in tg)
    top_oracle = set(sorted(impacts, key=impacts.get, reverse=True)[:budget])
    oracle_r = 100.0*(base - remove_nodes(threads, top_oracle))/base
    deg_r    = 100.0*(base - remove_nodes(threads, top_deg_set))/base

    return {
        "label": label,
        "n_threads": len(threads),
        "budget": budget,
        "seed_check_pass": seed_check,
        "threads_receiving": threads_receiving,
        "threads_receiving_pct": 100.0*threads_receiving/len(threads),
        "budget_in_top10pct": budget_conc,
        "nodes_per_thread_mean": float(nodes_per_thread.mean()),
        "nodes_per_thread_median": float(np.median(nodes_per_thread)),
        "nodes_per_thread_max": int(nodes_per_thread.max()),
        "per_thread_deg_mean": float(np.mean(thread_deg_r)),
        "per_thread_deg_median": float(np.median(thread_deg_r)),
        "per_thread_rand_mean": float(np.mean(thread_rand_r)),
        "degree_wins_thread_level": float(np.mean(thread_deg_r)) > float(np.mean(thread_rand_r)),
        "wasted_nodes": len(wasted),
        "overlap_pct": overlap_pct,
        "oracle_reduction": oracle_r,
        "degree_reduction": deg_r,
        "oracle_gap_pp": oracle_r - deg_r,
        "degree_pct_of_oracle": 100.0*deg_r/oracle_r if oracle_r > 0 else 0,
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

    print("Thread-Level Analysis: D2, D3, S5, R3")
    print("="*65)
    all_r = []
    for event_dir, label in events:
        print(f"\nProcessing {label}...", end=" ", flush=True)
        r = analyse_event(os.path.join(base, event_dir), label)
        all_r.append(r)
        print("done.")
        print(f"  D1 Seed==threads: {'PASS' if r['seed_check_pass'] else 'FAIL'}")
        print(f"  D2 Threads receiving removal: {r['threads_receiving']}/{r['n_threads']} ({r['threads_receiving_pct']:.1f}%)")
        print(f"     Budget in top-10% threads: {r['budget_in_top10pct']:.1f}%")
        print(f"     Nodes/thread: mean={r['nodes_per_thread_mean']:.2f}, median={r['nodes_per_thread_median']:.1f}, max={r['nodes_per_thread_max']}")
        print(f"  D3 Per-thread degree mean: {r['per_thread_deg_mean']:.2f}%  random mean: {r['per_thread_rand_mean']:.2f}%")
        print(f"     Degree wins at thread level: {r['degree_wins_thread_level']}")
        print(f"  S5 Wasted nodes (overlap): {r['wasted_nodes']}/{r['budget']} ({r['overlap_pct']:.1f}%)")
        print(f"  R3 Oracle: {r['oracle_reduction']:.2f}%  Degree: {r['degree_reduction']:.2f}%  Degree achieves {r['degree_pct_of_oracle']:.1f}% of oracle")

    print("\n\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"\n{'Event':<20} {'Threads w/removal':>18} {'Top10%':>8} {'Overlap':>9} {'Oracle%':>9}")
    print("─"*68)
    for r in all_r:
        print(f"{r['label']:<20} {r['threads_receiving']:>5}/{r['n_threads']:<10} "
              f"({r['threads_receiving_pct']:>4.0f}%)  "
              f"{r['budget_in_top10pct']:>6.1f}%  "
              f"{r['overlap_pct']:>7.1f}%  "
              f"{r['degree_pct_of_oracle']:>8.1f}%")

    print(f"\nMean degree achieves {np.mean([r['degree_pct_of_oracle'] for r in all_r]):.1f}% of oracle")
    print(f"Mean overlap: {np.mean([r['overlap_pct'] for r in all_r]):.1f}%")

    with open("thread_level_results.json","w") as f:
        json.dump(all_r, f, indent=2)
    print("\nSaved to: thread_level_results.json")
