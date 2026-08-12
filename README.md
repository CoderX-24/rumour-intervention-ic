# Rumour Cascade Intervention — Reproducibility Guide

This repo tests whether targeting high-degree nodes for early removal
reduces the spread of rumour cascades on Twitter, using the PHEME
dataset (Charlie Hebdo, Ottawa Shooting, Germanwings, Sydney Siege,
Ferguson, Prince Toronto, Putin Missing, Ebola-Essien).

## Setup

```
pip install -r requirements.txt
```

You also need the PHEME dataset locally (not included in this repo —
see licensing at https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619).
Every script that needs it takes the dataset directory as its one
command-line argument, e.g.:

```
python3 build_networks.py /path/to/all-rnr-annotated-threads
```

## Run order

Scripts fall into two groups: ones that read PHEME directly (need the
data-dir argument), and ones that only read another script's JSON
output (no argument, must be run from this directory, after their
dependency has produced its output file).

**Stage 1 — build from PHEME data directly:**
```
python3 build_networks.py <pheme_data_dir>         # -> network_stats.json
python3 compute_all_events.py <pheme_data_dir>      # -> all_events_results.json (8 events)
python3 compute_intervention.py <pheme_data_dir>    # -> intervention_results.json (5-event primary set)
python3 compute_intervention_extended.py <pheme_data_dir>  # -> intervention_results_extended.json
python3 calibrate_ic.py <pheme_data_dir>            # -> calibration_results.json
python3 robustness.py <pheme_data_dir>              # -> robustness_results.json
python3 structural_analysis.py <pheme_data_dir>     # -> structural_results.json
python3 thread_level_analysis.py <pheme_data_dir>   # -> thread_level_results.json
python3 sensitivity_truncation.py <pheme_data_dir>
python3 additional_robustness.py <pheme_data_dir>
```

**Stage 2 — analysis, reads Stage 1 outputs only (run from this directory, no argument):**
```
python3 transfer_analysis.py            # reads intervention_results.json
python3 transfer_analysis_extended.py   # reads intervention_results_extended.json
python3 leave_one_out.py                # reads all_events_results.json  -> loo_results.json
python3 cohens_d_with_ci.py             # reads intervention_results.json
python3 kendalls_w_per_budget.py        # reads robustness_results.json
python3 synthetic_baseline.py           # reads structural_results.json, intervention_results.json
                                         # -> synthetic_baseline_results.json
```

**Stage 3 — verification / diagnostics (run last, after Stage 1 + 2 outputs exist):**
```
python3 stats_verification.py           # independent scipy recomputation of every paper stat
python3 ic_degeneracy_check.py <pheme_data_dir>   # confirms IC p*->1 is a true boundary optimum
```

## Notes

- `Ebola-Essien` is deliberately included in `all_events_results.json`
  but flagged `"underpowered": true` (only 14 threads) and excluded
  from the primary 7-event transferability analysis.
- `intervention_results.json` (n=5: Charlie Hebdo, Ottawa Shooting,
  Germanwings, Sydney Siege, Ferguson) is the *primary* statistical
  set referenced by Kendall's W, Wilcoxon, and Cohen's d in the paper.
  `all_events_results.json` (n=7, powered events) is used for the
  binomial ranking test only.
- All budget percentages (e.g. 5%) are computed over the *non-seed*
  node pool only (`nonseed_nodes = all_nodes - root_set`), confirmed
  directly in `compute_intervention.py`. This was Bug 1 from an
  earlier project stage and has been fixed and verified —
  see `stats_verification.py`'s Bug-1 regression check.
