# ScGeo overview

## Motivation

Single-cell conclusions can change when the same cells are represented by
different embeddings. ScGeo makes that dependence explicit by organizing
perturbation-associated geometry into auditable evidence layers instead of
reporting a single opaque score.

## Estimands

ScGeo distinguishes several quantities:

- state displacement between prespecified conditions;
- uncertainty across independent biological samples when a valid sample key is
  available;
- stability of the displacement conclusion across prespecified representations;
- neighborhood and local-shape preservation;
- agreement between displacement and an available dynamics estimate;
- abundance and within-state distributional change as separate evidence.

These estimands are conditional on the input cells, labels, representations,
and experimental design. A cell-level analysis is descriptive when no valid
biological sample identity is available.

## Standard primitives and framework contribution

Geometric medians, norms, cosine similarity, nearest-neighbor overlap,
distribution distances, bootstrapping, and graph summaries are established
primitives. ScGeo does not claim to invent them. Its framework contribution is
to connect them through explicit sample-aware estimands, representation roles,
coverage rules, reason codes, and reproducible reporting.

## Evidence-layer architecture

The layers are deliberately not collapsed into a composite score. A state may
show large displacement but weak representation stability, stable displacement
without abundance change, or distributional change without a coherent
directional shift. Keeping those outcomes separate makes negative and
inconclusive evidence visible.

Next: [Analysis layers](ANALYSIS_LAYERS.md), [Quick start](QUICKSTART.md), and
[Result interpretation](RESULT_INTERPRETATION.md).
