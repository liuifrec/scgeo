# Quick start

## Required AnnData fields

For a sample-aware contrast, provide:

- `adata.obs["condition"]`: the prespecified comparison groups;
- `adata.obs["cell_type"]`: the state or node label;
- `adata.obs["donor"]`: an independent biological sample identifier;
- `adata.obsm["X_pca"]`: a quantitative representation.

Replace these example keys with the corresponding columns in your object. Do
not use cells, conditions, or artificial partitions as the sample key.

## Minimal analysis

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

## Primary and sensitivity representations

Prespecify a small primary set that supports the consensus, then label other
embeddings as dimensional or exploratory sensitivity views. Related PCA views
share information. UMAP is generally appropriate for display, not quantitative
geometry.

## Expected outputs

Analysis functions write named results into the AnnData object and preserve
metadata describing conditions, representations, sample key, estimator,
coverage, and rules. `scgeo.get.state_report` assembles one row per state from
available stored analyses. Plotting functions consume that report without
creating a new combined estimand.

Interpret the report with [Result interpretation](RESULT_INTERPRETATION.md).
