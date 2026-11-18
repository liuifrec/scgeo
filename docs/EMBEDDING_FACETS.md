# Flexible embedding and velocity faceting

`scgeo.pl.embedding_facets` produces matched embedding panels from a standard
[AnnData](https://anndata.readthedocs.io/) object. Split by any one or two
`adata.obs` annotations without fitting another embedding or changing the
original data. Projected velocity is optional.

[Plotting overview](PLOTTING.md) · [Unified geometry–dynamics atlas](GEOMETRY_DYNAMICS_ATLAS.md) · [Velocity support](VELOCITY_FIELDS.md) · [Full plotting API](api/pl.md) · [Citation](CITING.md)

**Availability:** This API is new in the development checkout containing this
page. It is **not** included in the published PyPI `scgeo==0.1.0` distribution.

## One-line UMAP comparison

```python
import scgeo as sg

result = sg.pl.embedding_facets(
    adata,
    basis="umap",
    facet="condition",        # any adata.obs column
    color="cell_type",        # another adata.obs column
    context="all",           # faint whole-dataset UMAP underneath each facet
    return_result=True,
    show=False,
)
result.save("figures/conditions", formats=("pdf", "svg", "png"), dpi=300)
```

Only `adata.obsm["X_umap"]` and the requested observation metadata are
necessary. A single embedding works even if the object has no RNA-velocity
vectors, ScGeo analysis results, PAGA graph, biological samples, or conditions.
Numeric `color` values use **one shared color normalization** across facets;
categorical labels share a palette. A supplied `palette` takes precedence;
otherwise a valid Scanpy `adata.uns["<color>_colors"]` palette is reused when
`color` is an ordered categorical annotation, followed by a deterministic
fallback. Missing labels are not confused with literal `"(missing)"` values.

## Matching local velocity in each facet

A compatible upstream projected velocity array must already exist, with the
same shape as the embedding. For UMAP, this is normally
`adata.obsm["velocity_umap"]`. A PCA velocity is **not** a substitute for a
UMAP-projected velocity.

```python
result = sg.pl.embedding_facets(
    adata,
    facet="condition",
    color="cell_type",
    velocity="stream",       # or "grid"
    shared_limits=True,
    shared_velocity_scale=True,
    velocity_options={
        "grid_size": 26,
        "n_neighbors": 30,
        "min_neighbors": 5,
        "min_coherence": 0.2,
        "stream_density": 1.0,
    },
    return_result=True,
    show=False,
)
result.save("figures/velocity_facets", formats=("pdf", "svg", "png"))
```

The existing ScGeo local-field estimator operates **separately** on each
facet's actual finite cells. It uses a Gaussian kernel with a finite radius,
neighbor-count threshold, and directional-coherence threshold. By default, the
radius is calculated **once from all eligible input cells** and the same grid
coordinates are used for every panel. In grid mode, the arrow multiplier is
also shared and derived from supported grid speeds across all facets, not
optimized separately. Numeric velocity colors (mean speed or coherence) use a
shared normalization. `shared_velocity_scale=False` permits separately
resolved grid-arrow multipliers; doing so reduces visual comparability.

In `velocity_options`, choose `color_by="speed"` for local mean projected-vector
speed or `"coherence"` for directional agreement among input vectors. Neither
value is a sample-level confidence score or a test statistic. Unsupported grid
positions stay blank, and even an empty facet retains an exported support grid
with `reason="no_eligible_vectors"`.

**Interpretation:** Streamlines follow a locally interpolated *display-space*
velocity field. They are not experimentally observed lineages, transition
probabilities, elapsed time, or ScGeo's high-dimensional shift–velocity
alignment. This function never fits kinetics, UMAP, pseudotime, or a PAGA
transition graph.

## Two-way comparisons, ordering, rare categories

```python
result = sg.pl.embedding_facets(
    adata,
    facet_row="batch",
    facet_col="condition",
    color="cell_type",
    row_order=["batch_1", "batch_2"],
    col_order=["control", "treated"],
    col_wrap=4,  # used only for a one-way column facet
    velocity="grid",
    max_points=1200,  # strict drawing limit *per facet*; not an analysis subset
    random_state=17,
    return_result=True,
    show=False,
)
```

- `facet="condition"` is shorthand for `facet_col="condition"`. For more
  than four categories, `col_wrap=4` wraps one-way facet panels across rows.
- When both `facet_row` and `facet_col` are provided, all combinations are
  displayed. Empty combinations are retained with `n = 0` and zero support.
- Categorical level order comes from the ordered pandas categorical when
  available. `facet_order`, `row_order`, and `col_order` can override it.
  They must include all observed labels; optional additional labels create
  explicitly empty panels, rather than silently excluding observations.
- A true missing label gets its own category and does not collapse into a
  literal string like `"NA"` or `"(missing)"`.
- `context="all"` adds the entire existing embedding faintly behind a subset;
  it does **not** include those background cells in that facet's velocity
  estimate. `context="none"` draws only the selected facet.
- `highlight_key="cell_type", highlight="State A"` outlines selected
  observed labels. Highlights, display sampling and view limits never alter
  the source velocity field or any upstream evidence.
- With `shared_limits=False`, each panel uses its own embedding limits and
  local field grid extent; such plots are intentionally less comparable.

Use `result.data["facets"]` to see observation counts, finite coordinates,
velocity support, and exact plotted limits. `result.data["cells"]` retains
*every input observation* with its facet identity and displayed flag. The
`velocity_grid`, `velocity_vectors` and `velocity_paths` tables retain actual
support decisions, projected arrows, and support-audited streamline geometry,
respectively. `PlotResult.save(...)` also writes a checksummed figure/source
manifest. The metadata records the selected facet definitions, the source
basis, palette provenance, and velocity-support settings.

## Reproducible gallery

The gallery is executable without downloading anything:

```bash
python -m pip install -e .
python examples/embedding_facets_gallery.py \
  --dataset synthetic \
  --output-dir results/facets \
  --formats png svg pdf
```

It renders a one-way grid-velocity comparison, a matched streamline comparison,
a two-way facet grid, a shared numeric color scale, and a no-velocity fallback.
**Every synthetic coordinate and vector is constructed**, and the display is
not a fitted UMAP, observed transition, or biological trajectory.

### Public single-cell validations

The same gallery accepts checksum-verified, existing public datasets. No
organism-specific API options or artificial conditions are needed.

```bash
python examples/embedding_facets_gallery.py --dataset pbmc \
  --input-h5ad /data/pbmc3k_processed.h5ad \
  --output-dir results/facets_pbmc

python examples/embedding_facets_gallery.py --dataset pancreas \
  --input-h5ad /data/cellrank_pancreas_preprocessed.h5ad \
  --output-dir results/facets_pancreas
```

Alternatively, explicitly add `--download-public` in place of
`--input-h5ad ...` to fetch the exact checksum-pinned source. **Downloading is
never implicit.** Public inputs are verified against the expected file size
and SHA-256 digest before processing.

- **Scanpy processed PBMC3k:** its observed `louvain` annotations and existing
  `X_umap` are used for eight cell-label facets. There is no supplied velocity;
  the example does not generate it.
- **CellRank E15.5 pancreas:** the four largest supplied `clusters` annotations
  are displayed using original `X_umap` and `velocity_umap`. This is an
  explicitly recorded display subset; the original upstream vectors are never
  refitted. These are cell-state comparisons at one timepoint, not temporal
  transitions or radiation responses.

These independently sourced datasets test how the *software renderer* behaves;
plotting them is not external validation of ScGeo's biological conclusions.

## How this relates to the unified atlas

Use `geometry_dynamics_atlas()` to summarize **distinct evidence layers** in
one figure. Use `embedding_facets()` for **matched comparisons of the same
embedding/dynamics layer across metadata groups**. Both return `PlotResult`
and follow the same export conventions. Neither function makes statistical
claims that are absent from the input.
