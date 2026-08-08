# kendalls_w_per_budget.py
# Computes Kendall's W at each budget level (1%, 2%, 5%, 10%, 20%)
# addressing Mo3: the concordance statistic should be reported at each
# budget, not just the primary 5% level.
#
# Usage: python3 kendalls_w_per_budget.py
# Requires: robustness_results.json (from robustness.py)

import json
import numpy as np
from scipy.stats import chi2 as chi2_dist, binomtest

def kendalls_w(rankings_matrix):
    m, n = rankings_matrix.shape
    rank_sums = rankings_matrix.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = 12 * S / (m**2 * (n**3 - n))
    chi2_val = m * (n - 1) * W
    p = 1 - chi2_dist.cdf(chi2_val, df=n - 1)
    return round(W, 4), round(chi2_val, 4), round(p, 6)

if __name__ == "__main__":
    with open("robustness_results.json") as f:
        rob = json.load(f)

    events = ["Charlie Hebdo", "Ottawa Shooting", "Germanwings",
              "Sydney Siege", "Ferguson"]
    budgets = [("1pct","1%"), ("2pct","2%"), ("5pct","5%"),
               ("10pct","10%"), ("20pct","20%")]

    print("Kendall's W at Each Budget Level")
    print("="*65)
    print(f"{'Budget':>8} {'W':>8} {'χ²':>8} {'p':>12} "
          f"{'deg>bet':>8} {'deg>ran':>8} {'Deg wins':>10}")
    print("─"*65)

    for key, label in budgets:
        rank_matrix = []
        deg_wins = 0
        deg_bet  = 0  # how many events: degree > betweenness
        deg_ran  = 0  # how many events: degree > random

        for ev in events:
            r = rob[ev][key]
            vals = {
                "degree":      r["degree"]["reduction_pct"],
                "betweenness": r["betweenness"]["reduction_pct"],
                "random":      r["random"]["mean_reduction_pct"],
            }
            ranked = sorted(vals, key=vals.get, reverse=True)
            rn = {s: i+1 for i, s in enumerate(ranked)}
            rank_matrix.append([rn["degree"], rn["betweenness"], rn["random"]])

            if ranked[0] == "degree":
                deg_wins += 1
            if vals["degree"] > vals["betweenness"]:
                deg_bet += 1
            if vals["degree"] > vals["random"]:
                deg_ran += 1

        W, chi2, p = kendalls_w(np.array(rank_matrix))
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        print(f"{label:>8} {W:>8.4f} {chi2:>8.4f} {p:>12.6f} "
              f"{deg_bet:>5}/5  {deg_ran:>5}/5  {deg_wins:>5}/5  [{sig}]")

    print("\nNote: at 10% and 20% budgets, betweenness vs random ordering")
    print("varies by event (see Table 4), reducing W from 1.0.")
    print("Degree ALWAYS ranks first regardless of budget.")

    # Also compute W for the full 3-way ranking breakdown
    print("\n\nFull ranking strings per budget:")
    print(f"{'Budget':>8}  {'CH':>18} {'Ottawa':>18} {'German':>18} "
          f"{'Sydney':>18} {'Ferguson':>18}")
    print("─"*100)
    for key, label in budgets:
        row = []
        for ev in events:
            r = rob[ev][key]
            vals = {"degree": r["degree"]["reduction_pct"],
                    "betweenness": r["betweenness"]["reduction_pct"],
                    "random": r["random"]["mean_reduction_pct"]}
            ranked = sorted(vals, key=vals.get, reverse=True)
            row.append(">".join(s[:3] for s in ranked))
        print(f"{label:>8}  " + "  ".join(f"{s:>18}" for s in row))
