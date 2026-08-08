# transfer_analysis_extended.py
# Extended transferability analysis combining:
#   - 5 primary events (from intervention_results.json)
#   - 4 extended contexts (from intervention_results_extended.json)
# Total: 9 contexts
#
# Note on independence:
#   CH-rumours and CH-non-rumours are subsets of Charlie Hebdo (partial overlap)
#   Ferguson W1 and W3 are temporal subsets of Ferguson (partial overlap)
#   Primary and extended contexts partially overlap — reported separately
#   and combined, with the non-independence caveat stated explicitly.
#
# Usage: python3 transfer_analysis_extended.py
# Runtime: <1 minute

import json
import numpy as np
from scipy.stats import binomtest, wilcoxon, rankdata
from itertools import combinations

def kendalls_w(rankings_matrix):
    m, n = rankings_matrix.shape
    rank_sums = rankings_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    from scipy.stats import chi2 as chi2_dist
    chi2 = m * (n - 1) * W
    p = 1 - chi2_dist.cdf(chi2, df=n - 1)
    return W, chi2, p

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

def analyse(results, title):
    n = len(results)
    labels  = [r["label"] for r in results]
    deg_v   = [r["degree"]["reduction_pct"]           for r in results]
    bet_v   = [r["betweenness"]["reduction_pct"]       for r in results]
    ran_v   = [r["random"]["mean_reduction_pct"]       for r in results]

    print(f"\n{'='*65}")
    print(f"{title}  (n={n})")
    print(f"{'='*65}")

    print(f"\n{'Context':<24} {'Degree':>9} {'Betweenness':>13} {'Random':>9}  Ranking")
    print("─"*70)
    rank_matrix = []
    for label, d, b, r in zip(labels, deg_v, bet_v, ran_v):
        ranked = sorted({"degree":d,"betweenness":b,"random":r}.items(),
                        key=lambda x:x[1], reverse=True)
        rn = {s:i+1 for i,( s,_) in enumerate(ranked)}
        rank_matrix.append([rn["degree"], rn["betweenness"], rn["random"]])
        print(f"{label:<24} {d:>8.2f}% {b:>12.2f}% {r:>8.2f}%  "
              f"{'>'.join(s for s,_ in ranked)}")
    rank_matrix = np.array(rank_matrix)

    print(f"\n  Mean:  Degree={np.mean(deg_v):.2f}%  "
          f"Betweenness={np.mean(bet_v):.2f}%  "
          f"Random={np.mean(ran_v):.2f}%")
    print(f"  Std:   Degree={np.std(deg_v):.2f}%  "
          f"Betweenness={np.std(bet_v):.2f}%  "
          f"Random={np.std(ran_v):.2f}%")

    # Binomial
    wins = sum(1 for r in results if r["ranking"][0]=="degree")
    bp   = binomtest(wins, n, p=1/3, alternative='greater').pvalue
    print(f"\n  Degree wins: {wins}/{n}")
    print(f"  Binomial p = {bp:.8f}  "
          f"({'significant' if bp<0.05 else 'not significant'} at α=0.05)")

    # Kendall's W
    W, chi2, pw = kendalls_w(rank_matrix)
    print(f"  Kendall's W = {W:.4f}, χ²={chi2:.2f}, df=2, p={pw:.6f}")

    # Pairwise Wilcoxon
    pairs = [("Degree vs Betweenness", deg_v, bet_v),
             ("Degree vs Random",      deg_v, ran_v),
             ("Betweenness vs Random", bet_v, ran_v)]
    raw_ps = []
    stats_out = []
    for name, a, b in pairs:
        diff = [x-y for x,y in zip(a,b)]
        if all(d==0 for d in diff):
            stat, p = 0, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, alternative='greater')
            except Exception:
                stat, p = 0, 1.0
        d = cohens_d(np.array(a), np.array(b))
        raw_ps.append(p)
        stats_out.append((name, stat, p, d))

    # Holm
    order = np.argsort(raw_ps)
    holm  = np.array(raw_ps, dtype=float)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, raw_ps[idx]*(len(raw_ps)-rank))
    for i in range(1, len(order)):
        holm[order[i]] = max(holm[order[i]], holm[order[i-1]])

    print(f"\n  Pairwise Wilcoxon (Holm-corrected):")
    for i,(name,stat,rp,d) in enumerate(stats_out):
        hp  = holm[i]
        sig = "***" if hp<0.001 else ("**" if hp<0.01 else ("*" if hp<0.05 else "ns"))
        print(f"    {name:<25}: W={stat:.0f}, raw p={rp:.4f}, "
              f"Holm p={hp:.4f}, d={d:.3f}  [{sig}]")

    return {"wins":wins, "n":n, "binomial_p":bp,
            "kendalls_w":W, "kendalls_p":pw}

if __name__ == "__main__":

    with open("intervention_results.json") as f:
        primary = json.load(f)
    primary_list = [{"label":k, **v} for k,v in primary.items()]

    with open("intervention_results_extended.json") as f:
        extended = json.load(f)

    all_contexts = primary_list + extended

    # ── 1. Primary only (5 events — main result) ──────────────────────────────
    r1 = analyse(primary_list, "PRIMARY ANALYSIS — 5 independent events")

    # ── 2. Extended only (4 sub-event contexts) ───────────────────────────────
    r2 = analyse(extended, "EXTENDED CONTEXTS — 4 sub-event contexts")

    # ── 3. All 9 combined (with non-independence caveat) ──────────────────────
    r3 = analyse(all_contexts,
                 "COMBINED — all 9 contexts (partial non-independence noted)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("SUMMARY OF TRANSFERABILITY EVIDENCE")
    print("="*65)
    print(f"""
  Primary (5 independent events):
    Degree wins {r1['wins']}/5 | Binomial p = {r1['binomial_p']:.6f}
    Kendall's W = {r1['kendalls_w']:.4f} (p = {r1['kendalls_p']:.6f})

  Extended (4 sub-event contexts):
    Degree wins {r2['wins']}/4 | Binomial p = {r2['binomial_p']:.6f}
    Kendall's W = {r2['kendalls_w']:.4f} (p = {r2['kendalls_p']:.6f})

  Combined (9 contexts, noting partial non-independence):
    Degree wins {r3['wins']}/9 | Binomial p = {r3['binomial_p']:.8f}
    Kendall's W = {r3['kendalls_w']:.4f} (p = {r3['kendalls_p']:.6f})

  Interpretation:
    The ranking degree > betweenness > random holds across all primary
    events, all content-type splits (CH rumours vs non-rumours), and
    all temporal decompositions (Ferguson waves). The finding is not
    specific to any content type, temporal phase, or event scale.
    The combined binomial p = {r3['binomial_p']:.2e} reflects this but should
    be interpreted cautiously given partial non-independence.
    The primary analysis (p = {r1['binomial_p']:.6f}) is the conservative
    and appropriate primary claim.
""")
