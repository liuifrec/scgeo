# Result interpretation

ScGeo statuses summarize the evidence available under prespecified rules. They
do not replace the experimental design or turn descriptive cell-level analyses
into population-level inference.

## `stable_effect`

The state meets the stored effect, coverage, and primary-representation
agreement rules. Report the effect size, uncertainty unit, representations,
sample counts, and relevant sensitivity results. Prefer “representation-stable
association” unless the design supports stronger language.

## `stable_neutral`

The state meets coverage and stability rules but does not meet the stored
effect rule. This is a stable neutral result under the analyzed resolution and
representations, not proof that every biologically meaningful effect is absent.

## `representation_unstable`

The prespecified representations do not support the same conclusion. Report
which representations differ, whether local geometry is distorted, and whether
the primary consensus changes under leave-one-representation-out analysis. Do
not select a favorable representation after inspecting the outcome.

## `insufficient_coverage`

The state does not meet stored cell, sample, or representation coverage rules.
Keep it visible with the exact reason. Absence of a reported effect is not a
neutral finding when coverage is insufficient.

## Aligned and discordant dynamics

Dynamics evidence compares displacement with an available dynamics estimate.
`aligned` means the stored directions agree under the documented rule;
`discordant` means they do not. Either result is conditional on both models and
representations. Report `unavailable` when the dynamics input or coverage is
missing.

## Warnings and limitations

Recommended language includes:

- “descriptive cell-level stability” when cells, rather than biological
  samples, were resampled;
- “replicate-aware association” when independent biological samples support
  the comparison;
- “cross-sectional difference” when different subjects were sampled at
  different times;
- “representation instability” or “extrapolation sensitivity” for weak or
  corrupted representation support;
- “UMAP shown for display only” when UMAP is plotted but not analyzed.

Avoid causal language without a causal design, independent-confirmation claims
for nested PCA views, general out-of-distribution detection claims, and
biological-replicate inference from pooled cells.
