# What you can do with ScGeo

**ScGeo helps you quantify how annotated cell states differ between conditions
and assess how those conclusions depend on the representation.** It works
downstream of representation learning, using coordinates and labels already
stored in AnnData.

[Try the quick start](QUICKSTART.md) · [Read the paper](https://doi.org/10.1093/bioadv/vbag266) · [Find a workflow](COMPANION_NOTEBOOKS.md)

## Start from a biological question

You might want to compare a treatment with a control, describe remodeling
across sampled conditions, or ask whether an apparent state shift persists
across analysis choices. ScGeo makes the relevant evidence inspectable:

| Evidence | What it contributes |
| --- | --- |
| Robust displacement | Magnitude and direction of a state-level shift |
| Biological-sample uncertainty | Variability across independently sampled units |
| Representation stability | Agreement and sensitivity across prespecified coordinate systems |
| Local geometry | Neighborhood preservation and shape distortion |
| Dynamics agreement | Relationship between displacement and a separately supplied dynamics estimate |
| Abundance and distribution | Complementary composition and within-state distribution summaries |

The layers stay separate. A population can change in abundance without a
coherent displacement, or show a large displacement that is sensitive to the
representation. Keeping these outcomes visible supports a more precise account
of the biological comparison.

## A practical workflow

1. **Specify the design:** define conditions, annotated states, biological
   samples, and the scope of inference.
2. **Supply quantitative representations:** choose a primary set and identify
   exploratory or dimensionality-sensitivity views in advance.
3. **Estimate the relevant quantities:** begin with displacement, then add
   representation, local-geometry, dynamics, or distributional analyses as needed.
4. **Review the report:** inspect effects, coverage, uncertainty, and sensitivity
   together using `get.state_report` and the plotting functions.
5. **Record and communicate:** retain inputs, versions, settings, and the
   conclusions justified by the design.

ScGeo uses established geometric and statistical tools, including robust
centers, distances, cosine similarity, neighborhood comparisons, and
bootstrapping. Its contribution is the common workflow connecting those tools
to explicit sampling units, representation roles, evidence availability, and
reproducible reporting.

## Fit with the single-cell ecosystem

Use Scanpy or other suitable tools for preprocessing and representation
construction. ScGeo can analyze supplied PCA, corrected, variational, diffusion,
or other quantitative coordinates when their interpretation is justified.
Existing RNA-velocity or fate-inference outputs can provide additional dynamics
context. The [installation guide](INSTALLATION.md) explains optional tools.

Geometry remains conditional on the supplied coordinates and study design.
UMAP is normally a display view; nested PCA representations share information.
Without recoverable biological-sample identity, the analysis remains descriptive.
Coverage and sensitivity results help identify where a conclusion is supported
or unresolved; they do not create causal evidence.

## Learn, reproduce, and publish

- [Quick start](QUICKSTART.md): a small synthetic run and an AnnData example.
- [Plotting guide](PLOTTING.md) and [gallery](PLOTTING_GALLERY.md): development evidence figures, contrast comparisons, biological samples, and source-data exports.
- [PAGA shift maps](PAGA_SHIFT_MAP.md): state connectivity, condition centres, projected velocity, and stored alignment on a shared display.
- [Analysis layers](ANALYSIS_LAYERS.md): quantities and comparison units.
- [Result interpretation](RESULT_INTERPRETATION.md): evidence labels and reporting language.
- [Companion workflows](COMPANION_NOTEBOOKS.md): the paper's applications and validation settings.
- [Reproducibility](REPRODUCIBILITY.md): current versions and frozen checkpoints.
- [Citation and Methods guide](CITING.md): reference-manager downloads and reporting details.
