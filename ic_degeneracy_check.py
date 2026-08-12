"""
ic_degeneracy_check.py

Diagnoses WHY calibrate_ic.py found best_p=0.99 for every event (a value
sitting at the edge of the [0.01, 0.99] grid). Per the handoff doc's
PENDING item ("IC degeneracy not honestly addressed"), this either
mechanistically explains the saturation or confirms it's unresolved --
it does not let a possible bug pass as supporting evidence either way.

Two checks:
  1. MONOTONICITY: extend the grid past 0.99 toward 1.0. If KS keeps
     falling all the way to p=1 with no interior minimum, the original
     grid search didn't stop at an arbitrary boundary -- larger p is
     ALWAYS better, i.e. there is no true calibrated optimum inside
     (0,1). This is degeneracy, not a coincidence of the grid spacing.
  2. EQUIVALENCE: at p=1, IC is deterministic full-tree traversal from
     each root. If simulated cascade size at p=1 matches the real
     cascade size closely, that mechanistically explains WHY p*->1:
     the real data IS (approximately) the p=1 IC trace, so IC has
     nothing to calibrate -- it just re-derives the tree it was given.

Run from ~/Desktop/rumour-intervention-ic/ with:
    python3 ic_degeneracy_check.py <pheme_data_dir>

Reuses calibrate_ic.py's own tree-building and simulation functions so
results are directly comparable to calibration_results.json -- this
script must sit in the same directory as calibrate_ic.py.
"""

import os
import sys
import json
import numpy as np
from scipy.stats import ks_2samp

# Reuse the exact simulation code from calibrate_ic.py rather than
# reimplementing it, so results are apples-to-apples.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_ic import load_event, simulate_cascade_sizes  # noqa: E402


EVENTS = [
    ("charliehebdo-all-rnr-threads", "Charlie Hebdo"),
    ("ottawashooting-all-rnr-threads", "Ottawa Shooting"),
    ("germanwings-crash-all-rnr-threads", "Germanwings"),
    ("sydneysiege-all-rnr-threads", "Sydney Siege"),
    ("ferguson-all-rnr-threads", "Ferguson"),
]

# Extend past the original grid's 0.99 ceiling toward the true boundary.
EXTENDED_P_GRID = [0.99, 0.995, 0.999, 0.9999, 1.0]
N_RUNS = 200  # matches calibrate_ic.py's more careful run count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ic_degeneracy_check.py <pheme_data_dir>")
        sys.exit(1)
    base = sys.argv[1]

    print("=" * 70)
    print("IC DEGENERACY CHECK")
    print("Extending calibration grid toward p=1 to test for a true")
    print("interior optimum vs. boundary degeneracy.")
    print("=" * 70)

    all_diagnostics = {}

    for event_dir, label in EVENTS:
        event_path = os.path.join(base, event_dir)
        if not os.path.isdir(event_path):
            print(f"\n{label}: directory not found -- skipping")
            continue

        print(f"\nLoading {label}...", end=" ", flush=True)
        G, roots, thread_sizes, threads = load_event(event_path)
        real_sizes = np.array(thread_sizes)
        real_mean = real_sizes.mean()
        print(f"done ({len(threads)} threads)")

        rows = []
        for p in EXTENDED_P_GRID:
            sim_sizes = simulate_cascade_sizes(threads, p, n_runs=N_RUNS)
            ks_stat, _ = ks_2samp(real_sizes, sim_sizes)
            sim_mean = sim_sizes.mean()
            rows.append(
                {
                    "p": p,
                    "ks_stat": float(ks_stat),
                    "sim_mean": float(sim_mean),
                    "ratio": float(sim_mean / real_mean),
                }
            )
            print(
                f"    p={p:<8} KS={ks_stat:.5f}  sim_mean={sim_mean:.2f}  "
                f"real_mean={real_mean:.2f}  ratio={sim_mean/real_mean:.4f}"
            )

        # ---- Check 1: monotonicity ----
        ks_values = [r["ks_stat"] for r in rows]
        is_monotone_decreasing = all(
            ks_values[i] >= ks_values[i + 1] - 1e-9 for i in range(len(ks_values) - 1)
        )

        # ---- Check 2: equivalence at p=1 ----
        p1_ratio = rows[-1]["ratio"]
        p1_ks = rows[-1]["ks_stat"]
        close_to_real = abs(p1_ratio - 1.0) < 0.05  # within 5% of real mean

        print(f"\n  Monotone decreasing all the way to p=1: "
              f"{'YES -- confirms boundary degeneracy' if is_monotone_decreasing else 'NO -- interior structure exists, investigate further'}")
        print(f"  At p=1: sim/real cascade size ratio = {p1_ratio:.4f}, KS = {p1_ks:.5f}")
        print(f"  {'CONFIRMS' if close_to_real else 'DOES NOT CONFIRM'} "
              f"IC-at-p=1 approximates real cascade sizes "
              f"(mechanistic explanation for p*->1 saturation)")

        all_diagnostics[label] = {
            "extended_grid": rows,
            "monotone_decreasing": bool(is_monotone_decreasing),
            "p1_ratio": float(p1_ratio),
            "p1_ks": float(p1_ks),
            "confirms_equivalence": bool(close_to_real),
        }

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    n_events = len(all_diagnostics)
    n_monotone = sum(1 for d in all_diagnostics.values() if d["monotone_decreasing"])
    n_equivalent = sum(1 for d in all_diagnostics.values() if d["confirms_equivalence"])

    print(f"Monotone-to-p=1 in {n_monotone}/{n_events} events")
    print(f"IC-at-p=1 approximates real cascades in {n_equivalent}/{n_events} events")

    if n_monotone == n_events and n_equivalent == n_events:
        print(
            "\nMECHANISTIC EXPLANATION CONFIRMED: as p->1, IC degenerates "
            "into deterministic full-tree traversal, which is structurally "
            "equivalent to the real observed cascades (since PHEME reply "
            "trees ARE the full traversal from the source tweet). The KS "
            "distance has no interior minimum -- it strictly decreases "
            "toward p=1 -- so grid search reporting p*=0.99 is not an "
            "artifact of the grid's [0.01,0.99] range, it is the true "
            "(boundary) optimum. This is safe to write into the paper as "
            "a mechanistic explanation, NOT flagged as an unresolved bug."
        )
    else:
        print(
            "\nNOT FULLY CONFIRMED across all events -- do not present "
            "p*=0.99 as mechanistically explained in the paper. Flag as "
            "unresolved instead, per the handoff doc's fallback option, "
            "and investigate the events above marked NO / DOES NOT CONFIRM."
        )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ic_degeneracy_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
