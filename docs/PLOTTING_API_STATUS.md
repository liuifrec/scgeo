# Plotting API upgrade audit

This audit covers **all 52 names in `scgeo.pl.__all__`**, including compatibility
aliases, the Plotly adapter, bundle export, and the two exported private palette
helpers. It starts from PR #21 head `57349cb`, which contains merged PR #20,
and records the continuation merged in [PR #22](https://github.com/liuifrec/scgeo/pull/22)
at `bd0c2406b41bf29964c630580e54c5c01b8e8299` on 19 November 2026. Existing implementations are extended in
place. Statistical methods in `scgeo.tl` and stored biological results are not
changed by this work.

[Plotting guide](PLOTTING.md) · [API signatures](api/pl.md) ·
[PR #21 baseline verification](validation/embedding_facets_pr21_20261118.json) ·
[Local continuation validation](validation/plotting_api_continuation_20261118.json)

## Completed increments (merged)

1. Reconciled PR #21: four successful workflows, 17 passing jobs, two intentional
   publishing skips; 468 tests passed and one optional integration test skipped
   independently on Python 3.9/3.12 and in the local Python 3.10 baseline. Fixed
   unsupported-facet grid coordinates when panel limits are independent.
2. Added caller-owned axes to facets and the atlas, with explicit colorbar
   placement, Figure/SubFigure support, parent-canvas scope, source-equivalence
   tests, and an executable composition gallery.
3. Upgraded the three stored score panels and top-k renderer. Highlights and
   background now share a scale; histograms use all input finite scores and
   export their exact edges/counts. Existing tuple returns remain the default.

4. Upgraded composition/volcano evidence plots to retain missing values, validate
   probabilities and preserve source scales. Added complete exports and named
   composition axes without changing any statistical method.

5. Extended distribution, overlap, ranking and report views with exact source
   tables and caller-owned axes; added independent global metric scales and
   corrected observation fallback and category legends.
6. Extended mapping/OOD views, retaining novel predictions outside a reference
   palette and true missing labels. Exported threshold/selection provenance and
   actual contour bin counts/means; corrected transposed contour coordinates.

7. Completed consensus structure, standalone legend and recovery-compass source
   exports. Preserved legacy returns and descriptive means; reused PAGA graph
   validation and made display-coordinate semantics explicit.

The development shift-explanation extension adds `shift_curves` and
`shift_embedding` with caller-owned axes and source exports. Its new analysis
function is documented separately in [EXPLAIN_SHIFT.md](EXPLAIN_SHIFT.md);
the earlier plotting milestones below remain historical records.

The [compass evidence update](COMPASS_EVIDENCE.md) adds stored evidence and
individual-sample summaries, corrects contrast-scoped vector aggregation and
missing OOD behavior, and extends consensus selection, concordance labels,
atlas legends and display budgets. Existing return contracts remain available;
the intentional display-mode correctness changes are documented in that guide.

The table describes interface coverage, **not a claim that every scientific or
visual edge case is resolved**. `Yes` means `return_result=True` is available.
All Matplotlib renderers now expose this opt-in contract. The Plotly and bundle contracts are
intentional and should not be replaced with another implementation.

| Export | Axes contract | PlotResult | Status / next work |
|---|---|---|---|
| `PlotResult` | n/a | n/a | Shared result/export/style utility. |
| `export_figure` | n/a | n/a | Shared result/export/style utility. |
| `category_palette` | n/a | n/a | Shared result/export/style utility. |
| `style_context` | n/a | n/a | Shared result/export/style utility. |
| `evidence_comparison` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `geometry_velocity_concordance` | axes | Yes | Validates stored provenance; separate effect/alignment availability, source coverage, display names and deterministic labels. |
| `replicate_effect` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `delta_rank` | ax | Yes | Now exports exact original/displayed sources and accepts caller axes; see summary guide. |
| `score_umap` | ax | Yes | Delegates through kwargs to score_embedding; no duplicate implementation. |
| `score_embedding` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `shift_curves` | ax / axes | Yes | Sample curves, expression/distribution panels for sparse states, explicit coordinate support, coverage and source tables. |
| `shift_embedding` | ax | Yes | Frozen center weights projected for display; curved connectors explicitly schematic. |
| `highlight_topk_cells` | ax | Yes | Now exports full cell selection/display flags and accepts shared normalization. |
| `distribution_test_volcano` | ax | Yes | Now exports source rows, exact coordinates, clipping/label flags; validates probability range. |
| `embedding_density` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `embedding_facets` | axes | Yes | Now composes in caller Figure/SubFigure; parent export and source scope recorded. |
| `density_overlap` | axes | Yes | Now exports exact original/displayed sources and accepts caller axes; see summary guide. |
| `density_overlap_grid` | ax | Yes | Now exports exact original/displayed sources and accepts caller axes; see summary guide. |
| `distribution_test` | axes | Yes | Now exports exact original/displayed sources and accepts caller axes; see summary guide. |
| `distribution_report` | axes | Yes | Now exports exact original/displayed sources and accepts caller axes; see summary guide. |
| `alignment_panel` | axes | Yes | Now delegates to shared score/highlight renderers; one scale and exact all-input histogram sources. |
| `ambiguity_panel` | axes | Yes | Now delegates to shared score/highlight renderers; one scale and exact all-input histogram sources. |
| `consensus_subspace_panel` | axes | Yes | Now delegates to shared score/highlight renderers; one scale and exact all-input histogram sources. |
| `paga_composition_volcano` | ax | Yes | Now exports untouched sources/point transforms; missing evidence stays missing; raw OR uses null one. |
| `paga_composition_bar` | ax | Yes | Now exports selection/order/source values; unavailable effects/probabilities receive explicit labels. |
| `paga_composition_panel` | axes | Yes | Now composes the upgraded child renderers on named axes with complete source provenance. |
| `mapping_confidence_umap` | ax | Yes | Lowest finite scores; full selection/display identity and explicit colorbar axes. |
| `ood_cells` | ax | Yes | Explicit threshold provenance; unavailable flags stay nullable; X-prefixed bases accepted. |
| `mapping_qc_panel` | axes | Yes | Novel reference-palette labels retained; query and scalar display scopes exported. |
| `consensus_structure` | ax | Yes | Top-k or threshold; independent face/outline opacity, complete group/sample denominators and display budgets; four-item return retained. |
| `legend_from_data` | ax | Yes | Full entry exports with explicit truncation; composable legend-only axes. |
| `_get_scanpy_palette` | n/a | n/a | Exported compatibility palette helper; retain its name. |
| `_fallback_palette` | n/a | n/a | Exported compatibility palette helper; retain its name. |
| `paga_shift_map` | ax | Yes | Reference or pooled node anchors, captured-cell or sample-abundance pies, and separate contrast/vector evidence. |
| `ood_landscape` | ax / axes | Yes | Full cells/summary/contour-bin exports; explicit thresholds, support and omission reasons. |
| `velocity_shift_alignment` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `velocity_field` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `velocity_condition_compare` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `composition_drift` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `robustness_matrix` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `gallery_overview` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `geometry_dynamics_atlas` | axes | Yes | Caller canvas support, adaptive layout, shared normalization, highlight-aware labels, explicit legend omissions and sample coverage. |
| `recovery_compass` | ax / axes | Yes | Display or stored evidence, pooled sparse-state anchors, sample/coverage summary axes, shared scales and complete source exports. |
| `state_flow_sankey` | Plotly | Native | Intentional Plotly adapter: complete counts in layout.meta; native HTML/JSON export. |
| `state_flow_alluvial` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `state_flow_chord` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `paga_scgeo` | ax | Yes | Delegates through kwargs to paga_shift_map; no duplicate implementation. |
| `state_evidence_panel` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `representation_stability_heatmap` | axes | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `consensus_state_map` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |
| `perturbation_report` | Bundle | Bundle | Existing multi-figure exporter; strict availability and panel_kwargs already supported. |
| `local_distortion_map` | ax | Yes | Existing axes/result implementation; retain scientific and backward-compatibility regression coverage. |

## Validation milestone and remaining work

The local continuation passed **524 tests**, with one explicitly opt-in public
PBMC integration test skipped. Seven plotting galleries and the synthetic
quickstart completed. Across 34 PlotResult exports, all 102 figure files and
155 source tables matched their recorded hashes; CSV row counts also matched.
Seven new/updated documentation previews are byte-identical to their exports.

Runtime signatures, generated documentation and link checks passed. A wheel
built through the sdist passed the repository's source/metadata/dependency
consistency audit and installation smoke test outside the checkout. No code in
`scgeo.tl`, `scgeo.get` or `scgeo.pp` changed. This is an interface/rendering
validation milestone, not new statistical or biological validation.

PR #22's actual GitHub CI passed: four workflows, 17 successful jobs, and two
intentional publishing skips, including Python 3.9/3.12, operating-system,
conda and public-data matrices. The historical local validation JSON above
retains the pre-push status at the time it was recorded. PR #22 is now merged;
this work did not create a package release. The next milestone is
[deployment readiness and JupyterHub preflight](JUPYTERHUB_PREFLIGHT.md).

Ongoing edge-case regression coverage includes nullable/missing/colliding
labels, unknown evidence, supplied norms, explicit colorbars, deterministic
sampling, early error paths, SubFigures and complete source tables. Study-
specific units, sample structure, original graph order and evidence availability
still need to be supplied by the caller; plots do not infer them.

For each increment: preserve default return shapes and argument meanings,
exercise the relevant synthetic gallery, inspect rendered output, add semantic
regressions, regenerate the API reference, and run the appropriate CI-equivalent
checks. Remote CI evidence covers its exact tested commit; subsequent changes
require their own validation.
