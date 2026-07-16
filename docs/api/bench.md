# scgeo.bench

## `scgeo.bench.evaluate_ground_truth`

**Signature**  
`(adata, *, robust_shift_key: 'str' = 'robust_shift', representation_key: 'str' = 'representation_stability', local_geometry_key: 'str' = 'local_geometry_stability') -> 'dict[str, pd.DataFrame]'`

**Docstring**  
Evaluate stored ScGeo outputs against synthetic simulation truth.

Zero-effect magnitude coverage is marked `not_applicable`; bootstrap magnitude
intervals are reported as uncertainty intervals, and null behavior is evaluated
through shifted-state false-call rates and prespecified null diagnostics.
Representation outputs are split into quality outlier, diagnostic distortion,
and explicit corruption targets; state-level local distortion is evaluated at
representation-pair × state resolution.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.framework_ablation`

**Signature**  
`(tables: 'Mapping[str, pd.DataFrame]', *, final_split: 'str' = 'evaluation') -> 'pd.DataFrame'`

**Docstring**  
Build a tidy framework-ablation table from synthetic benchmark outputs.

The long-form table preserves every evaluated unit. Condition distribution-shape
truth is separate from representation-local-distortion truth, and zero-effect
bootstrap magnitude coverage does not generate `misses`.
Diagnostic distorted representations are not scored as false explicit-corruption
calls.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.plot_framework_ablation`

**Signature**  
`(ablation_table: 'pd.DataFrame', *, split: 'str' = 'evaluation', normalize: 'bool' = True, figsize='auto', row_height: 'float' = 0.3, min_height: 'float' = 3.2, wrap_width: 'int' = 18, show_values: 'bool | str' = 'auto', title: 'Optional[str]' = None, save_path=None, show: 'bool' = True)`

**Docstring**  
Plot a manuscript-ready framework-ablation summary.

The default figure contains a fixed 5 x 5 capability table, a curated
applicable-performance heatmap, and a separate support-status heatmap.
`figsize='auto'` sizes height from row counts and row height, and panel widths
from wrapped label lengths within fixed caps. Unsupported cells are blank, and
`not_computed`, `not_applicable`, and scientifically irrelevant rows are not
mixed with performance outcomes. Plotting metadata is attached to the returned
Figure.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.run_simulation_suite`

**Signature**  
`(*, profile: 'str' = 'smoke', scenarios=None, seeds=None, output_dir=None, resume: 'bool' = True, n_jobs: 'int' = 1) -> 'dict[str, Any]'`

**Docstring**  
Run a reproducible synthetic ScGeo benchmark suite.

Final summaries use simulation seed/job as the independent unit and report
mean, median, spread, and confidence intervals across held-out jobs when the
number of held-out jobs permits.
The legacy scenario input `null` is accepted as an alias, but benchmark exports
use `null_effect`.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.simulate_perturbation_geometry`

**Signature**  
`(*, scenario: 'str' = 'centroid_shift', n_states: 'int' = 5, n_samples_per_condition: 'int' = 4, cells_per_sample: 'int' = 400, latent_dim: 'int' = 8, effect_size: 'float' = 1.0, affected_states=None, outlier_fraction: 'float' = 0.0, outlier_scale: 'float' = 10.0, sample_heterogeneity: 'float' = 0.15, abundance_effect: 'float' = 0.0, covariance_effect: 'float' = 0.0, warp_strength: 'float' = 0.0, velocity_mode=None, seed: 'int' = 0) -> 'ad.AnnData'`

**Docstring**  
Generate a synthetic perturbation-geometry benchmark AnnData object.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---
