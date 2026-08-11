"""
synthetic_baseline.py

Tests whether r(degree, impact) ≈ 0.97–0.99 is a structural near-tautology.

Method:
  Generate random trees matching PHEME structural properties (size distribution,
  branching factor). Run cascade truncation with degree targeting. Compute
  Spearman r(degree, impact) on synthetic trees.

Interpretation:
  - If synthetic r ≈ real r  → tautological. High r is forced by tree geometry,
    not by anything informative about PHEME reply structure.
  - If synthetic r << real r → real trees have structure that makes degree
    genuinely more predictive than in random trees of the same size.

Usage:
  python3 synthetic_baseline.py

Requires: structural_results.json, intervention_results.json (in same directory)
"""

import json
import numpy as np
from scipy import stats
import random
from collections import defaultdict

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_SYNTHETIC_FORESTS = 200   # synthetic forests per event (more = tighter CIs)
N_TRIALS_RANDOM     = 1000  # Monte Carlo trials for random strategy
BUDGET_FRAC         = 0.05  # 5% non-seed budget

EVENTS = [
    "charliehebdo",
    "ottawashooting",
    "germanwings",
    "sydneysiege",
    "ferguson",
]

# Map script slugs → keys in structural_results.json
EVENT_LABEL = {
    "charliehebdo":   "Charlie Hebdo",
    "ottawashooting": "Ottawa Shooting",
    "germanwings":    "Germanwings",
    "sydneysiege":    "Sydney Siege",
    "ferguson":       "Ferguson",
}

# ── load real results ─────────────────────────────────────────────────────────
print("Loading real results...")
with open("structural_results.json") as f:
    structural = json.load(f)

with open("intervention_results.json") as f:
    intervention = json.load(f)

# ── tree utilities ────────────────────────────────────────────────────────────

def get_descendants(node, children):
    """All descendants of node (BFS, not including node itself)."""
    desc = []
    stack = list(children.get(node, []))
    while stack:
        v = stack.pop()
        desc.append(v)
        stack.extend(children.get(v, []))
    return desc


def compute_node_degree(node, children):
    """Out-degree (number of direct children) of node."""
    return len(children.get(node, []))


def cascade_truncation_degree(children, n_nodes, budget_frac=BUDGET_FRAC):
    """
    Degree targeting: remove top-k non-seed nodes by out-degree.
    Seed = node 0. Non-seed = nodes 1..n_nodes-1.
    Returns (impact, degrees, impacts_per_node) where:
      impact            = fraction of non-seed nodes removed
      degrees           = list of degrees for ALL non-seed nodes
      impacts_per_node  = list of individual subtree-size/non-seed fractions
    """
    non_seeds = list(range(1, n_nodes))
    if not non_seeds:
        return 0.0, [], []

    k = max(1, int(len(non_seeds) * budget_frac))

    # Per-node metrics
    degrees = [compute_node_degree(v, children) for v in non_seeds]
    subtree_sizes = [1 + len(get_descendants(v, children)) for v in non_seeds]
    impacts_per_node = [s / len(non_seeds) for s in subtree_sizes]

    # Select targets by degree
    sorted_idx = sorted(range(len(non_seeds)),
                        key=lambda i: degrees[i], reverse=True)
    targets = [non_seeds[sorted_idx[i]] for i in range(k)]

    # Count removed (dedup descendants)
    removed = set()
    for t in targets:
        if t not in removed:
            removed.add(t)
            removed.update(get_descendants(t, children))

    impact = len(removed) / len(non_seeds)
    return impact, degrees, impacts_per_node


def cascade_truncation_random(children, n_nodes, budget_frac=BUDGET_FRAC,
                              n_trials=N_TRIALS_RANDOM):
    """Random targeting averaged over n_trials."""
    non_seeds = list(range(1, n_nodes))
    if not non_seeds:
        return 0.0
    k = max(1, int(len(non_seeds) * budget_frac))

    impacts = []
    for _ in range(n_trials):
        targets = random.sample(non_seeds, min(k, len(non_seeds)))
        removed = set()
        for t in targets:
            if t not in removed:
                removed.add(t)
                removed.update(get_descendants(t, children))
        impacts.append(len(removed) / len(non_seeds))
    return float(np.mean(impacts))


# ── random tree generators ────────────────────────────────────────────────────

def random_tree_uniform(n_nodes):
    """
    Uniform random tree (Prüfer sequence equivalent via random parent attachment).
    Each new node picks a parent uniformly at random from existing nodes.
    This is the simplest null model.
    """
    if n_nodes <= 1:
        return defaultdict(list), n_nodes
    children = defaultdict(list)
    for i in range(1, n_nodes):
        parent = random.randint(0, i - 1)
        children[parent].append(i)
    return children, n_nodes


def random_tree_branching(n_nodes, mean_branching=2.5):
    """
    Random tree grown by sampling branching factor from a Poisson distribution.
    More realistic: mimics shallow reply trees with bursty branching.
    """
    if n_nodes <= 1:
        return defaultdict(list), n_nodes

    children = defaultdict(list)
    queue = [0]
    node_id = 1

    while node_id < n_nodes and queue:
        parent = queue.pop(0)
        n_children = np.random.poisson(mean_branching)
        n_children = min(n_children, n_nodes - node_id)
        for _ in range(n_children):
            children[parent].append(node_id)
            queue.append(node_id)
            node_id += 1
            if node_id >= n_nodes:
                break

    # If queue empty but nodes remain, attach remaining to root
    while node_id < n_nodes:
        children[0].append(node_id)
        node_id += 1

    return children, n_nodes


# ── extract real thread sizes per event ──────────────────────────────────────

def get_real_thread_sizes(event_name, intervention_data):
    """
    Extract thread sizes from intervention_results.json.
    We approximate from the stored per-event data.
    """
    # intervention_results.json stores aggregate stats; we need thread sizes
    # Fall back to reading from structural_results if available
    if event_name in structural:
        ev = structural[event_name]
        # Try to get thread size distribution
        if "thread_sizes" in ev:
            return ev["thread_sizes"]
        # Otherwise reconstruct from n_threads and total nodes
        if "n_threads" in ev and "total_nodes" in ev:
            n = ev["n_threads"]
            total = ev["total_nodes"]
            mean_size = total / n
            # Approximate with log-normal (typical for cascade sizes)
            sigma = 1.2
            mu = np.log(mean_size) - 0.5 * sigma ** 2
            sizes = np.random.lognormal(mu, sigma, n).astype(int)
            sizes = np.clip(sizes, 2, None)  # at least seed + 1
            return sizes.tolist()
    # Hard-coded fallback from known PHEME stats
    fallback = {
        "charliehebdo":   {"n": 2002, "mean": 11.2},
        "ottawashooting": {"n": 857,  "mean": 8.4},
        "germanwings":    {"n": 403,  "mean": 7.1},
        "sydneysiege":    {"n": 1172, "mean": 10.8},
        "ferguson":       {"n": 1010, "mean": 10.1},
    }
    if event_name in fallback:
        fb = fallback[event_name]
        sigma = 1.2
        mu = np.log(fb["mean"]) - 0.5 * sigma ** 2
        sizes = np.random.lognormal(mu, sigma, fb["n"]).astype(int)
        sizes = np.clip(sizes, 2, None)
        return sizes.tolist()
    return [5] * 100  # last-resort fallback


# ── compute r(degree, impact) on a forest ────────────────────────────────────

def compute_r_for_forest(thread_sizes, tree_generator):
    """
    Build synthetic trees matching thread_sizes, run degree targeting,
    collect per-node (degree, impact) pairs, return Spearman r.
    """
    all_degrees = []
    all_impacts = []

    for size in thread_sizes:
        if size < 2:
            continue
        children, n = tree_generator(size)
        _, degrees, impacts = cascade_truncation_degree(children, n)
        all_degrees.extend(degrees)
        all_impacts.extend(impacts)

    if len(all_degrees) < 3:
        return np.nan

    r, _ = stats.spearmanr(all_degrees, all_impacts)
    return r


# ── main analysis ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SYNTHETIC BASELINE: Is r(degree, impact) tautological?")
print("=" * 70)
print(f"\nNull model forests per event : {N_SYNTHETIC_FORESTS}")
print(f"Budget fraction              : {BUDGET_FRAC*100:.0f}% non-seed nodes")
print(f"Tree generators              : uniform-random, Poisson-branching")

results = {}

for event in EVENTS:
    print(f"\n{'─'*60}")
    print(f"Event: {event.upper()}")

    label = EVENT_LABEL[event]

    # Real r from structural_results
    real_r = None
    if label in structural and "spearman_degree_impact" in structural[label]:
        real_r = structural[label]["spearman_degree_impact"]

    if real_r is None:
        print(f"  [WARN] Could not find real r for {event} in structural_results.json")
        print(f"         Keys available: {list(structural.get(label, {}).keys())}")
        real_r = float("nan")
    else:
        print(f"  Real r(degree, impact)     : {real_r:.4f}")

    # Get thread sizes from structural_results using real mean/std
    ev_data = structural[label]
    mean_size = ev_data["mean_thread_size"]
    std_size  = ev_data["std_thread_size"]
    n_threads = ev_data["n_nonseed"]  # approximate; will resample anyway
    mean_branching = ev_data["mean_branching"]

    # Log-normal fit to observed mean/std
    sigma2 = np.log(1 + (std_size / mean_size) ** 2)
    mu     = np.log(mean_size) - 0.5 * sigma2
    thread_sizes = np.random.lognormal(mu, np.sqrt(sigma2),
                                       int(n_threads / mean_size)).astype(int)
    thread_sizes = np.clip(thread_sizes, 2, None).tolist()

    # Simulate N_SYNTHETIC_FORESTS synthetic forests, each with same size dist
    r_uniform   = []
    r_branching = []

    for trial in range(N_SYNTHETIC_FORESTS):
        # Resample sizes with replacement (bootstrap the size distribution)
        sampled_sizes = random.choices(thread_sizes, k=len(thread_sizes))

        r_u = compute_r_for_forest(sampled_sizes,
                                   lambda n: random_tree_uniform(n))
        r_b = compute_r_for_forest(sampled_sizes,
                                   lambda n, mb=mean_branching: random_tree_branching(n, mb))
        if not np.isnan(r_u):
            r_uniform.append(r_u)
        if not np.isnan(r_b):
            r_branching.append(r_b)

    r_u_mean = np.mean(r_uniform)
    r_u_ci   = np.percentile(r_uniform, [2.5, 97.5])
    r_b_mean = np.mean(r_branching)
    r_b_ci   = np.percentile(r_branching, [2.5, 97.5])

    print(f"  Synthetic r — uniform      : {r_u_mean:.4f}  95% CI [{r_u_ci[0]:.4f}, {r_u_ci[1]:.4f}]")
    print(f"  Synthetic r — Poisson-branching: {r_b_mean:.4f}  95% CI [{r_b_ci[0]:.4f}, {r_b_ci[1]:.4f}]")

    if not np.isnan(real_r):
        gap_u = real_r - r_u_mean
        gap_b = real_r - r_b_mean
        print(f"  Gap (real − uniform)       : {gap_u:+.4f}")
        print(f"  Gap (real − Poisson)       : {gap_b:+.4f}")

        if r_u_mean > 0.92:
            verdict_u = "TAUTOLOGICAL (high r in random trees too)"
        elif gap_u > 0.05:
            verdict_u = "INFORMATIVE (real trees meaningfully higher)"
        else:
            verdict_u = "BORDERLINE"
        print(f"  Verdict (uniform model)    : {verdict_u}")

    results[event] = {
        "real_r": real_r,
        "synthetic_uniform_mean": r_u_mean,
        "synthetic_uniform_ci": r_u_ci.tolist(),
        "synthetic_branching_mean": r_b_mean,
        "synthetic_branching_ci": r_b_ci.tolist(),
    }

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

real_rs   = [v["real_r"] for v in results.values() if not np.isnan(v["real_r"])]
synth_us  = [v["synthetic_uniform_mean"] for v in results.values()]
synth_bs  = [v["synthetic_branching_mean"] for v in results.values()]

print(f"\nMean real r across events       : {np.mean(real_rs):.4f}")
print(f"Mean synthetic r (uniform)      : {np.mean(synth_us):.4f}")
print(f"Mean synthetic r (Poisson)      : {np.mean(synth_bs):.4f}")
print(f"Mean gap (real − uniform)       : {np.mean(real_rs) - np.mean(synth_us):+.4f}")

print("\nINTERPRETATION:")
mean_synth = np.mean(synth_us)
mean_real  = np.mean(real_rs)

if mean_synth > 0.92:
    print("  → r(degree, impact) is NEAR-TAUTOLOGICAL.")
    print("    Random trees of the same size produce equally high r.")
    print("    The mechanism section must be reframed: the high correlation")
    print("    follows from tree geometry (removing a node removes all")
    print("    descendants), not from anything special about PHEME structure.")
    print("    Reframe contributions around partial-tree, LOO, oracle-overlap.")
elif mean_real - mean_synth > 0.05:
    print("  → r(degree, impact) is GENUINELY INFORMATIVE.")
    print("    Real reply trees produce substantially higher r than random trees")
    print("    of matching size. PHEME topology (e.g., hub structure, shallow")
    print("    high-degree roots) makes degree more predictive than in null model.")
    print("    The mechanism claim can stand, with this as supporting evidence.")
else:
    print("  → BORDERLINE result. Gap is small but synthetic r < 0.92.")
    print("    Report both numbers in the paper. Acknowledge partial tautology")
    print("    while noting real trees show modestly higher r.")

# ── save results ──────────────────────────────────────────────────────────────
out = {
    "n_synthetic_forests": N_SYNTHETIC_FORESTS,
    "budget_frac": BUDGET_FRAC,
    "events": results,
    "summary": {
        "mean_real_r": float(np.mean(real_rs)) if real_rs else None,
        "mean_synthetic_uniform_r": float(np.mean(synth_us)),
        "mean_synthetic_branching_r": float(np.mean(synth_bs)),
        "gap_real_minus_uniform": float(np.mean(real_rs) - np.mean(synth_us)) if real_rs else None,
    }
}

with open("synthetic_baseline_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nResults saved to synthetic_baseline_results.json")
print("Done.")
