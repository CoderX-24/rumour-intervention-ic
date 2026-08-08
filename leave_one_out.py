# leave_one_out.py — R1
# Leave-one-out transferability test.
# For each event, train "strategy recommendation" on the remaining 6 events
# (majority vote on which strategy ranks first), then test on the held-out event.
# This directly simulates the operational use case: given prior events,
# which strategy should a platform deploy for the next event?
#
# Also tests: if we rank strategies by mean reduction across training events,
# does the top-ranked strategy also win on the held-out event?
#
# Usage: python3 leave_one_out.py
# Requires: all_events_results.json

import json
import numpy as np
from scipy.stats import binomtest

with open("all_events_results.json") as f:
    all_results = json.load(f)

# Use 7 powered events (exclude Ebola-Essien)
events = [k for k, v in all_results.items() if not v.get("underpowered", False)]
print(f"Events: {events}\n")

strategies = ["degree", "betweenness", "random"]

def get_reduction(result, strategy):
    if strategy == "random":
        return result["random"]["mean_reduction_pct"]
    return result[strategy]["reduction_pct"]

def rank_strategies(event_results):
    vals = {s: get_reduction(event_results, s) for s in strategies}
    return sorted(strategies, key=vals.get, reverse=True)

print("="*65)
print("LEAVE-ONE-OUT TRANSFERABILITY TEST")
print("="*65)
print(f"\n{'Hold-out event':<22} {'Train recommends':<20} {'Hold-out winner':<20} {'Correct?'}")
print("─"*70)

correct = 0
results_loo = []

for held_out in events:
    training_events = [e for e in events if e != held_out]

    # Method 1: majority vote on which strategy ranks first
    vote_counts = {s: 0 for s in strategies}
    for e in training_events:
        winner = rank_strategies(all_results[e])[0]
        vote_counts[winner] += 1
    recommended = max(vote_counts, key=vote_counts.get)

    # Method 2: mean reduction across training events
    mean_reductions = {}
    for s in strategies:
        mean_reductions[s] = np.mean([get_reduction(all_results[e], s)
                                       for e in training_events])
    recommended_by_mean = max(mean_reductions, key=mean_reductions.get)

    # Actual winner on held-out event
    actual_winner = rank_strategies(all_results[held_out])[0]

    correct_vote = recommended == actual_winner
    correct_mean = recommended_by_mean == actual_winner
    if correct_vote:
        correct += 1

    print(f"{held_out:<22} {recommended:<20} {actual_winner:<20} "
          f"{'✓' if correct_vote else '✗'}  (mean method: {'✓' if correct_mean else '✗'})")
    results_loo.append({
        "held_out": held_out,
        "recommended": recommended,
        "actual_winner": actual_winner,
        "correct": correct_vote,
    })

n = len(events)
p_loo = binomtest(correct, n, p=1/3, alternative='greater').pvalue
print(f"\nLOO accuracy: {correct}/{n}")
print(f"Binomial p (H0: random guessing p=1/3): {p_loo:.6f}")
print()
print("Interpretation:")
if correct == n:
    print(f"  ✓ Degree-centrality recommendation is correct for ALL {n} held-out events.")
    print(f"  ✓ This is the strongest evidence for transferability: training on any 6 events")
    print(f"    correctly predicts the best strategy for the 7th.")
else:
    print(f"  Degree recommended correctly in {correct}/{n} events.")
    failures = [r for r in results_loo if not r["correct"]]
    for f in failures:
        print(f"  ✗ {f['held_out']}: recommended {f['recommended']}, "
              f"actual winner {f['actual_winner']}")

# Also show: what would happen if a platform always used degree (policy simulation)
print("\n\nPOLICY SIMULATION: Always deploy degree-centrality targeting")
print("─"*55)
for e in events:
    r = all_results[e]
    deg_r  = r["degree"]["reduction_pct"]
    rand_r = r["random"]["mean_reduction_pct"]
    best_r = max(get_reduction(r, s) for s in strategies)
    regret = best_r - deg_r
    print(f"  {e:<22}: degree={deg_r:.1f}%  "
          f"best_possible={best_r:.1f}%  regret={regret:.1f}pp")

mean_regret = np.mean([max(get_reduction(all_results[e], s) for s in strategies)
                        - all_results[e]["degree"]["reduction_pct"]
                        for e in events])
print(f"\n  Mean regret from always using degree: {mean_regret:.2f} percentage points")
print(f"  (Regret = performance gap vs optimal strategy for that event)")

out = "loo_results.json"
with open(out, "w") as f:
    json.dump(results_loo, f, indent=2)
print(f"\nSaved to: {out}")
