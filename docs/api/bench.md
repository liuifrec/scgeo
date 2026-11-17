# scgeo.bench

ScGeo 0.1.0 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
**Synthetic benchmarks.** Simulate controlled perturbations and evaluate stored outputs against known truth. The benchmark profiler requires Linux, WSL or macOS.

- [`evaluate_ground_truth`](#scgeobenchevaluate_ground_truth)
- [`framework_ablation`](#scgeobenchframework_ablation)
- [`plot_framework_ablation`](#scgeobenchplot_framework_ablation)
- [`run_simulation_suite`](#scgeobenchrun_simulation_suite)
- [`simulate_perturbation_geometry`](#scgeobenchsimulate_perturbation_geometry)

### `scgeo.bench.evaluate_ground_truth`

Evaluate stored ScGeo outputs against synthetic simulation truth.

```text
scgeo.bench.evaluate_ground_truth(
    adata,
    *,
    robust_shift_key: 'str' = 'robust_shift',
    representation_key: 'str' = 'representation_stability',
    local_geometry_key: 'str' = 'local_geometry_stability',
) -> 'dict[str, pd.DataFrame]'
```

[View implementation](../../scgeo/bench/_simulation.py)

- Output: dictionary of tidy DataFrames for state magnitude errors, rank agreement, shifted-state precision/recall/F1, null false classification, bootstrap uncertainty interval coverage for nonzero effects, representation consensus labels, abundance truth, condition distribution-shape truth, non-identifiability truth, dynamics class accuracy, representation quality outlier detection, representation distortion detection, explicit corruption detection, pair-state representation-local-distortion detection, neighborhood preservation discrimination, replicate/sample diagnostics, coverage, runtime, and threshold sensitivity. Zero-effect magnitude coverage is marked `not_applicable`; null behavior is evaluated through shifted-state false-call rates and prespecified null diagnostics. Non-identifiable batch-confounded states are excluded from ordinary biological shifted-state classification rates. No composite benchmark score is created.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Evaluate stored ScGeo outputs against synthetic simulation truth.

The function returns tidy DataFrames and does not create a composite score.
Baseline magnitude comparisons are run on a temporary copy of ``adata``.
Bootstrap magnitude intervals are treated as uncertainty intervals; states
whose true magnitude is at or below ``_MAGNITUDE_ZERO_EPSILON`` are marked
``not_applicable`` for interval coverage rather than counted as failures.
```

</details>

### `scgeo.bench.framework_ablation`

Build a tidy framework-ablation table from synthetic benchmark outputs.

```text
scgeo.bench.framework_ablation(
    tables: 'Mapping[str, pd.DataFrame]',
    *,
    final_split: 'str' = 'evaluation',
) -> 'pd.DataFrame'
```

[View implementation](../../scgeo/bench/_simulation.py)

- Variants: A original mean shift on one representation; B robust shift on one representation; C robust shift plus representation consensus; D C plus local geometry diagnostics; E D plus dynamics agreement.
- Output: one row per scenario, seed/job, variant, failure mode, and evaluated unit with explicit `truth`, `call`, `status`, and `outcome`; held-out evaluation rows are marked by `final_evaluation`. `condition_distribution_shape_change` is separate from pair-state `representation_local_distortion`; representation-level targets are split into `representation_quality_outlier`, `representation_distortion_detection`, and `explicit_corruption_detection`. Zero-effect bootstrap magnitude coverage rows use `not_applicable` rather than `misses`. No composite score is created.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build a tidy framework-ablation table from synthetic benchmark outputs.

The ablation compares five explicitly named framework variants and records
per-unit outcomes without creating a composite score. Calibration and
held-out evaluation rows are preserved; ``final_split`` only marks which
rows should be used for final reporting.
```

</details>

### `scgeo.bench.plot_framework_ablation`

Plot held-out framework-ablation summaries without mixing unavailable evidence with performance outcomes.

```text
scgeo.bench.plot_framework_ablation(
    ablation_table: 'pd.DataFrame',
    *,
    split: 'str' = 'evaluation',
    normalize: 'bool' = True,
    figsize='auto',
    row_height: 'float' = 0.3,
    min_height: 'float' = 3.2,
    wrap_width: 'int' = 18,
    show_values: 'bool | str' = 'auto',
    title: 'Optional[str]' = None,
    save_path=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/bench/_simulation.py)

- Output: compact matplotlib Figure with a fixed 5 x 5 capability table, a curated applicable-performance heatmap, and a separate support-status heatmap. `figsize='auto'` sizes height from row counts and row height, and panel widths from wrapped label lengths within fixed caps. Unsupported cells are blank; `not_computed`, `not_applicable`, and scientifically irrelevant scenario/failure-mode rows are not plotted on the performance color scale. Plotting metadata is attached to the returned Figure.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a manuscript-ready framework-ablation summary.

``figsize='auto'`` sizes the figure from performance/support row counts,
row height, the documented minimum height, and wrapped label lengths.
The figure separates capability coverage, applicable performance outcomes,
and support statuses so unavailable or not-computed evidence is not plotted
on the same color scale as performance.
```

</details>

### `scgeo.bench.run_simulation_suite`

Run a reproducible synthetic ScGeo benchmark suite.

```text
scgeo.bench.run_simulation_suite(
    *,
    profile: 'str' = 'smoke',
    scenarios=None,
    seeds=None,
    output_dir=None,
    resume: 'bool' = True,
    n_jobs: 'int' = 1,
) -> 'dict[str, Any]'
```

[View implementation](../../scgeo/bench/_simulation.py)

- Profiles: `smoke` is approximately 1,000-2,000 cells/job with 25 bootstrap iterations and `k=(15,)`; `quick` is approximately 3,000-5,000 cells/job with 100 bootstrap iterations and `k=(15, 30)`; `manuscript` is approximately 5,000-10,000 cells/job with 300 bootstrap iterations and `k=(15, 30, 50)`.
- Calibration and held-out evaluation seeds are tracked separately. The suite exports threshold-sensitivity tables, the complete long-form framework-ablation table, seed/job-level ablation summaries, and final held-out summaries that report mean, median, spread, and confidence intervals across held-out jobs when the number of jobs permits. The suite does not modify consensus thresholds automatically.
- Synthetic benchmarks are methodological checks, not experimental validation.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Run a reproducible synthetic ScGeo benchmark suite.

Calibration and evaluation seeds are kept separate in all result tables.
The suite does not alter consensus thresholds automatically; threshold
sensitivity tables are exported for downstream calibration analyses. Final
ablation summaries aggregate outcomes within simulation seed/job before
summarizing across held-out jobs.
```

</details>

### `scgeo.bench.simulate_perturbation_geometry`

Generate a synthetic perturbation-geometry benchmark AnnData object.

```text
scgeo.bench.simulate_perturbation_geometry(
    *,
    scenario: 'str' = 'centroid_shift',
    n_states: 'int' = 5,
    n_samples_per_condition: 'int' = 4,
    cells_per_sample: 'int' = 400,
    latent_dim: 'int' = 8,
    effect_size: 'float' = 1.0,
    affected_states=None,
    outlier_fraction: 'float' = 0.0,
    outlier_scale: 'float' = 10.0,
    sample_heterogeneity: 'float' = 0.15,
    abundance_effect: 'float' = 0.0,
    covariance_effect: 'float' = 0.0,
    warp_strength: 'float' = 0.0,
    velocity_mode=None,
    seed: 'int' = 0,
) -> 'ad.AnnData'
```

[View implementation](../../scgeo/bench/_simulation.py)

- Scenarios: `null_effect`, `centroid_shift`, `abundance_only`, `covariance_only`, `local_warp`, `outlier_contamination`, `unequal_cell_counts`, `balanced_replicate_heterogeneity`, `batch_condition_confounding`, `representation_corruption`, `aligned_dynamics`, `discordant_dynamics`. The legacy input aliases `null` and `replicate_heterogeneity` are accepted but exported as `null_effect` and `balanced_replicate_heterogeneity`, respectively.
- Replicate designs: `balanced_replicate_heterogeneity` draws random sample offsets and exactly zero-centers them within each condition, so the expected outcome is wider uncertainty with a low neutral-state false-call rate. `batch_condition_confounding` applies a systematic sample/batch offset correlated with condition and is marked non-identifiable without additional design information; observed condition deltas are not counted as ordinary biological shifted-state detection successes or false positives.
- Output: synthetic AnnData with `obs['state']`, `obs['condition']`, `obs['sample']`, representation ensemble, optional velocity embeddings, and `adata.uns['simulation_truth']`.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Generate a synthetic perturbation-geometry benchmark AnnData object.

The simulation is synthetic only. It stores known truth under
``adata.uns['simulation_truth']`` and creates multiple latent
representations without requiring Scanpy, scVelo, or GPU libraries.
```

</details>
