# Frozen Manuscript Benchmark Protocol

This document records the frozen ScGeo manuscript benchmark protocol for
`configs/benchmark/manuscript_protocol_v1.json`. It is an execution and
reporting contract only. It does not modify estimators, thresholds, simulation
truth, evaluation logic, or plotting behavior.

## Freeze

- Protocol: `manuscript_protocol_v1`
- Date frozen: 2026-07-16
- Git commit: `7d6a3b8b8bb9c0bbb92b496994e09ae15a77d5a8`
- Package version: `0.1.0-dev`
- Profile: `manuscript`
- Calibration seeds: `0, 1, 2, 3, 4`
- Evaluation seeds: `5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19`

Calibration and evaluation seeds are disjoint. Evaluation seeds are held out:
after any evaluation-seed run, thresholds, estimator choices, consensus rules,
truth definitions, scenario definitions, evaluation logic, and plotting filters
must not be changed to improve reported performance. Any such change requires a
new protocol version and a new held-out evaluation.

## Frozen Settings

The manuscript profile uses 4 samples per condition, 800 cells per sample, 300
bootstrap iterations for `robust_shift` and `representation_stability`,
`k=(15, 30, 50)` for local geometry, and `max_exact_cells=3000`. The local
geometry bootstrap cap remains the current implementation behavior:
`min(profile_n_boot, 50)`.

The frozen biological shift threshold is `0.5`. The zero-effect magnitude
epsilon is `1e-8`. Bootstrap magnitude intervals are reported as bootstrap
uncertainty intervals; zero-effect states at or below epsilon are not treated as
formal magnitude coverage failures.

The exact estimator list is:

- `shift_mean`: `scgeo.tl.shift` on `X_truth`
- `robust_mean`: `scgeo.tl.robust_shift(center="mean")` on `X_truth`
- `robust_median`: `scgeo.tl.robust_shift(center="median")` on `X_truth`
- `robust_trimmed_mean`: `scgeo.tl.robust_shift(center="trimmed_mean")` on `X_truth`
- `robust_geometric_median`: `scgeo.tl.robust_shift(center="geometric_median")` on `X_truth`

The full consensus label rules are frozen in the JSON file under
`consensus_rules.representation_stability`.

## Scenarios

The frozen scenario set is:

- `abundance_only`
- `aligned_dynamics`
- `balanced_replicate_heterogeneity`
- `batch_condition_confounding`
- `centroid_shift`
- `covariance_only`
- `discordant_dynamics`
- `local_warp`
- `null_effect`
- `outlier_contamination`
- `representation_corruption`
- `unequal_cell_counts`

`batch_condition_confounding` is non-identifiable by design. Its states must be
marked non-identifiable and excluded from ordinary biological shift TP/FP/FN
metrics.

## Quick-Profile Findings To Carry Forward

The completed quick-profile audit is documented without changing behavior:

- Execution completed for all 60 quick jobs with no failed jobs and no missing
  result tables.
- Balanced replicate heterogeneity showed seed-specific neutral-state false
  calls in one held-out quick seed, while other quick seeds did not show
  systematic neutral false calls. The observed pattern is a genuine replicate
  heterogeneity limitation to monitor in the manuscript profile.
- Representation-level explicit corruption detection was perfect in the quick
  profile.
- State-level representation instability adjudication remained incomplete:
  explicit corrupted representations were found, but state-level consensus did
  not perfectly identify all affected conclusions.
- Representation local-distortion detection still had misses at the
  representation-state level.
- Small-profile bootstrap uncertainty intervals were wide; quick-profile
  bootstrap behavior should not be over-interpreted as manuscript-level
  uncertainty performance.
- No estimator dominated universally across all quick-profile criteria. Mean
  and robust centers differed by scenario and robustness target.

## Validation

Run:

```bash
python scripts/validate_benchmark_protocol.py
```

The validator compares the frozen protocol with the importable package defaults,
checks seed disjointness, verifies that all scenario names are valid, and runs a
small batch-condition-confounding invariant check confirming that
non-identifiable states do not enter ordinary shift TP/FP metrics.
