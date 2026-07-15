# scgeo.bench

## `scgeo.bench.evaluate_ground_truth`

**Signature**  
`(adata, *, robust_shift_key: 'str' = 'robust_shift', representation_key: 'str' = 'representation_stability', local_geometry_key: 'str' = 'local_geometry_stability') -> 'dict[str, pd.DataFrame]'`

**Docstring**  
Evaluate stored ScGeo outputs against synthetic simulation truth.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.framework_ablation`

**Signature**  
`(tables: 'Mapping[str, pd.DataFrame]', *, final_split: 'str' = 'evaluation') -> 'pd.DataFrame'`

**Docstring**  
Build a tidy framework-ablation table from synthetic benchmark outputs.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.plot_framework_ablation`

**Signature**  
`(ablation_table: 'pd.DataFrame', *, split: 'str' = 'evaluation', normalize: 'bool' = True, figsize=None, title: 'Optional[str]' = None, save_path=None, show: 'bool' = True)`

**Docstring**  
Plot a manuscript-ready framework-ablation summary.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.bench.run_simulation_suite`

**Signature**  
`(*, profile: 'str' = 'smoke', scenarios=None, seeds=None, output_dir=None, resume: 'bool' = True, n_jobs: 'int' = 1) -> 'dict[str, Any]'`

**Docstring**  
Run a reproducible synthetic ScGeo benchmark suite.

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
