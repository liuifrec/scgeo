<p align="center">
  <img src="docs/assets/ScGeo_logo.png" width="420" alt="ScGeo logo">
</p>

<h1 align="center">ScGeo</h1>

<p align="center">Geometry-aware analysis of single-cell representations</p>

## 1. What ScGeo is

ScGeo is an installable Python library for quantifying perturbation-associated
geometry in low-dimensional single-cell representations. It organizes robust
state displacement, biological-sample uncertainty, representation stability,
local geometry, dynamics agreement, abundance, and distributional change as
separate evidence layers.

Start with the [framework overview](docs/OVERVIEW.md), then follow the
[quick start](docs/QUICKSTART.md).

## 2. What ScGeo is not

ScGeo is not a batch-correction or integration algorithm, and it does not make
cells into biological replicates. It does not make UMAP a quantitative
coordinate system, turn agreement among nested PCA dimensions into independent
confirmation, or convert descriptive geometry into causal evidence. ScGeo uses
standard geometric and statistical primitives; its contribution is the
auditable framework that connects those primitives across evidence layers.

## 3. Core analysis layers

| Layer | Main question | Typical output |
|---|---|---|
| Robust shift | How far and in what direction did a state move? | displacement, normalized effect, direction stability |
| Sample uncertainty | Is the shift stable across biological samples? | sample bootstrap interval and coverage |
| Representation stability | Does the conclusion persist across prespecified representations? | consensus status and sensitivity diagnostics |
| Local geometry | Are neighborhoods and local shapes preserved? | overlap, Jaccard, and distortion metrics |
| Dynamics agreement | Does displacement align with an available dynamics estimate? | aligned, discordant, neutral, or unavailable evidence |
| Abundance and distribution | Did composition or within-state shape also change? | separate proportions and distribution distances |

See [Analysis layers](docs/ANALYSIS_LAYERS.md) for estimands and interpretation.

## 4. Installation

```bash
git clone https://github.com/liuifrec/scgeo.git
cd scgeo
conda env create -f environment.yml
conda activate scgeo
pip install -e .
```

The frozen package checkpoint used for the major-revision evidence package is
[`9a0ed16`](https://github.com/liuifrec/scgeo/tree/9a0ed16cbaa57f935f9c9bc87d1643a25b51012c).

## 5. Minimal sample-aware example

The biological sample identifier belongs in `adata.obs`; it should identify an
independently sampled unit rather than a cell, condition, or artificial
partition.

```python
import scgeo as sg

sg.tl.robust_shift(
    adata,
    rep="X_pca",
    condition_key="condition",
    group0="control",
    group1="treated",
    by="cell_type",
    sample_key="donor",
    n_boot=500,
)

sg.tl.representation_stability(
    adata,
    reps=["X_pca", "X_scvi"],
    node_key="cell_type",
    condition_key="condition",
    group0="control",
    group1="treated",
    sample_key="donor",
)

report = sg.get.state_report(adata, node_key="cell_type")
sg.pl.state_evidence_panel(report)
```

Required fields and expected outputs are listed in
[Quick start](docs/QUICKSTART.md).

## 6. How to interpret results

Read effect size, uncertainty, representation stability, local geometry, and
coverage together. `stable_effect`, `stable_neutral`,
`representation_unstable`, and `insufficient_coverage` describe evidence under
the stored rules; they are not interchangeable with population-level causal
claims. Bootstrap magnitude intervals summarize resampling uncertainty around a
nonnegative norm and are not automatically null-hypothesis tests.

See [Result interpretation](docs/RESULT_INTERPRETATION.md) for status-specific
language and warnings.

## 7. Representation robustness

Representations must be prespecified as primary or sensitivity views. Related
PCA20, PCA30, and PCA50 views share a basis and should not be counted as three
independent confirmations. UMAP should ordinarily remain display-only.
Condition mixing is a descriptive diagnostic when condition is biological, not
an integration objective. See [Analysis layers](docs/ANALYSIS_LAYERS.md).

## 8. Synthetic validation

The `scgeo.bench` module provides controlled simulations for effect magnitude,
abundance, distributional shape, local distortion, representation stability,
uncertainty, and geometry–dynamics agreement. Calibration and held-out seeds are
kept separate, and frozen thresholds are not automatically tuned by the suite.

The `balanced_replicate_heterogeneity` scenario uses random sample offsets that
are zero-centered within condition and is intended to test uncertainty without a
systematic condition shift. The separate `batch_condition_confounding` scenario
applies a systematic sample/batch offset correlated with condition and is
non-identifiable without additional design information.

Synthetic results validate behavior under their stated perturbations and
failure modes; they do not establish biological truth. The frozen protocol is
documented in [BENCHMARK_PROTOCOL.md](docs/revision/BENCHMARK_PROTOCOL.md).

## 9. Public validation

Public-data workflows live in the companion repository. They cover the public
pancreatic-development workflow; GSE249479, a public human HSPC inflammatory
xenograft dataset with no recoverable biological-replicate identity; and
GSE211713, a public mouse-lung radiation dataset with 20 independent mouse
libraries. The GSE211713 analysis is replicate-aware and cross-sectional, not
longitudinal. Their inference scopes are not interchangeable. See
[Companion notebooks](docs/COMPANION_NOTEBOOKS.md).

## 10. Reproducibility and companion repository

The library code is maintained here. Source notebooks, execution wrappers,
figure assembly, evidence ledgers, and ignored large artifacts are maintained
in [`scgeo-notebooks`](https://github.com/liuifrec/scgeo-notebooks). Source
notebooks are kept output-free; executed review copies are generated locally.

See [Reproducibility](docs/REPRODUCIBILITY.md) and the generated
[API reference](docs/api_reference.md).

## 11. Limitations

- Geometry depends on the chosen representation and its coverage.
- Biological-sample inference requires genuine independent sample identifiers.
- Abundance, displacement, and distributional change answer different questions.
- Dynamics agreement depends on the assumptions of the supplied dynamics model.
- Weak reference support is an unsupported-state warning, not general
  out-of-distribution detection.
- Public validations cannot establish causality beyond their experimental design.

## 12. Citation

If you use ScGeo, please cite:

Liu Y-C, Yoshida K. *ScGeo reveals non-canonical trajectories beyond RNA
velocity in radiation-induced hematopoietic recovery.*

The manuscript and repository documentation remain the authoritative sources
for the exact frozen revision checkpoint and analysis scope.
