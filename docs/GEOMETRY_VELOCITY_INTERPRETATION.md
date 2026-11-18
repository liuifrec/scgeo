# Geometry–velocity interpretation (development API)

[Plotting overview](PLOTTING.md) · [PAGA maps](PAGA_SHIFT_MAP.md) · [Projected velocity fields](VELOCITY_FIELDS.md) · [Result interpretation](RESULT_INTERPRETATION.md)

The comparison has **two different quantitative contexts**. Stored displacement and alignment are calculated in the declared ScGeo analysis representation; UMAP positions and projected RNA-velocity fields describe an existing **display embedding**. Treating a UMAP arrow angle as a substitute for stored 30-PC alignment would be incorrect.

This guide covers three additions in the development checkout, not the published `scgeo==0.1.0` wheel:

- `sg.pl.geometry_velocity_concordance`: a table-driven state overview of effect magnitude versus signed, stored alignment cosine.
- `sg.pl.velocity_condition_compare`: matched condition-specific projected-velocity grids or streamlines plus a support-masked local direction-comparison panel.
- `sg.pl.paga_shift_map(..., focus_states=..., edge_focus='incident'|'within')`: display-only graph filtering to foreground states with available evidence.

## 1. Concordance from frozen state tables

```python
import pandas as pd
import scgeo as sg

shift = pd.read_csv('robust_shift_selected.tsv', sep='\t')
alignment = pd.read_csv('velocity_shift_alignment_selected.tsv', sep='\t')

# Explicitly choose the same representation, filtering scope, center, and
# analysis parameters upstream. Do not concatenate conflicting summaries.
result = sg.pl.geometry_velocity_concordance(
    shift,
    alignment,
    state_key='state',
    alignment_state_key='leiden',
    effect_metric='normalized_delta_norm',
    alignment_metric='alignment_cosine',
    analysis_rep='X_pca'  # document that the stored analysis used 30 PCs,
    return_result=True,
    show=False,
)
result.save('figures/concordance', formats=('png', 'svg', 'pdf'))
focus = result.data['eligible']['_state_id'].tolist()
```

Eligibility is the intersection of stored coverage eligibility, usable alignment, and finite magnitude/cosine. Missing states are exported with their exclusion reason. Default cosine classes are **descriptive** (`≥0.5`, `≤−0.25`, otherwise intermediate), not significance decisions. Provide other thresholds when justified and report them. No sample-level inference is added. Raw displacement confidence intervals are drawn only for `effect_metric='delta_norm'` and their matching stored `magnitude_ci95_*` columns; a normalized magnitude never inherits those raw intervals.

For radiation-related small-sample studies, the biological mouse—not each cell—is the replication unit. Magnitude, cosine, and directional coherence cannot establish a causal fate trajectory, lineage path, or treatment significance by themselves.

## 2. Matched condition-specific projected velocity

This function requires `adata.obsm['X_umap']` and the *matching* `adata.obsm['velocity_umap']`, or another exactly matched embedding/projection pair. ScGeo does not project or refit the vectors in this plotting step.

```python
result = sg.pl.velocity_condition_compare(
    adata,
    condition_key='condition',
    group0='nonirradiated',
    group1='irradiated',
    basis='umap',
    mode='grid',  # or 'stream'
    grid_size=25,
    n_neighbors=30,
    min_neighbors=8,
    min_coherence=0.2,
    scale='auto',  # one multiplier computed for both grid panels
    return_result=True,
    show=False,
)
result.save('figures/velocity_by_condition', formats=('png', 'svg', 'pdf'))
```

The two fields share **grid positions, smoothing radius, neighbor cap, support thresholds, arrow multiplier, color normalization, and view limits**. The radius defaults to a single pooled estimate computed from *both selected groups*; supply an explicit radius when comparing figures across independent analysis runs. The third panel displays cosine between local mean projected velocities only where **both** conditions have supported, nonzero estimates. Single-condition support is marked separately. `paired_grid` retains all attempted positions, support masks and reasons so blank areas remain interpretable. The cosines in this third panel are local **UMAP projected-velocity agreements**, not ScGeo's stored latent-space shift–velocity alignment cosines.

Density or direction differences may arise from sampling, smoothing, and representation choices. Compare per-mouse fields, not only the pooled per-condition field, if biological replication is the target. No statistical uncertainty is inferred from neighbor counts or coherence.

## 3. Reduce graph clutter without editing PAGA

```python
result = sg.pl.paga_shift_map(
    adata,
    node_key='leiden',
    condition_key='condition',
    group0='nonirradiated',
    group1='irradiated',
    graph_node_order=adata.obs['leiden'].cat.categories,
    focus_states=focus,
    edge_focus='incident',  # only edges touching a focused state
    label_top_n=0,          # focus labels only
    return_result=True,
    show=False,
)
result.save('figures/focused_paga', formats=('png', 'svg', 'pdf'))
```

Use `edge_focus='within'` to show only links with both ends focused, or `'all'` for the previous full topology. Edges omitted by focus, thresholds, or unavailable anchors remain in the exported `edge_table` with explicit reasons. These are **undirected PAGA connectivities**, not time-resolved transitions. Focusing alters **display** only and preserves cell counts, shift estimates, graph weights, and state centers.

## Running on local in-house mouse files

The companion CLI `examples/mouse_geometry_velocity_review.py` takes an existing `.h5ad` plus **preselected** frozen per-state TSVs. It writes local images and sources without altering the inputs or uploading study data. Specify your actual column names and condition labels. If the saved `.h5ad` is absent, the tool fails rather than generating replacement coordinates or velocities.

```bash
python examples/mouse_geometry_velocity_review.py \
  --adata /private/path/mouse_30pc.h5ad \
  --shifts /private/path/robust_shift_selected.tsv \
  --alignments /private/path/velocity_shift_alignment_selected.tsv \
  --condition-key condition \
  --node-key leiden \
  --group0 nonirradiated \
  --group1 irradiated \
  --output-dir /private/path/geometry_velocity_review
```

**Availability boundary:** the previously supplied mouse smoke archive includes frozen state tables and figure manifests, but **not** the complete matched per-cell velocity coordinates. It is sufficient to validate the concordance renderer locally; it is insufficient to claim an in-house condition-separated velocity-field test. The latter requires the `.h5ad` on the secured server and must be verified there. Do not publish in-house source tables, cell-level inputs, donor or mouse identifiers without project approval.
