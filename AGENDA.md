# ScGeo Project Agenda

This file is the **source of truth** for what ScGeo is building and what is already done.
Keep it short, but keep it accurate.

## Current status (as of this repo snapshot)

**✅ Core package is working + tested.**
- scverse-style layout: `scgeo/{tl,pl,get,data,pp,bench}`
- storage contract: `adata.uns["scgeo"][store_key]` with params + outputs
- lightweight utilities for dense/sparse safety

**✅ Phase A — QC-aware mapping (graph-native) is implemented.**
- `sg.tl.map_query_to_ref(...)` (agnostic to how the graph was built: BBKNN / Harmony / scVI / Scanpy neighbors / ...)
- per-cell outputs: predicted label, confidence (entropy/margin), OOD, reject
- tables: `sg.get.table(..., kind="map_query_to_ref", level="global|per_label")`
- plots: `sg.pl.mapping_confidence_umap`, `sg.pl.ood_cells`, `sg.pl.mapping_qc_panel`

**✅ Phase B groundwork — Reference pools + cellxgene-census plumbing.**
- `sg.pp.ReferencePool`, `sg.pp.build_reference_pool(...)` (ANN backend: pynndescent)
- `sg.tl.map_query_to_ref_pool(...)` (pool-mode mapping/QC)
- `scgeo.data._census`: thin wrappers for Census open/query/embedding search
- `sg.pp.build_reference_pool_from_census(...)` + `sg.pp.build_query_to_ref_knn_edges_from_census(...)`

**✅ Demo notebooks are present and runnable.**
- PBMC ingest/BBKNN-based demo: `notebooks/pbmc_ingest_scgeo_demo_fixed.ipynb`
- scVelo velocity basics demo: `notebooks/scvelo_velocitybasics_scgeo_demo_fixed.ipynb`

## What we are doing next (short-term)

1) **Lock down the demos (today/this week)**
- Expand both notebooks with a *ScGeo plot gallery* section that exercises most `sg.pl.*` on realistic data.
- Add lightweight “smoke” checks (optional) for plotting functions.

2) **Phase B: census-backed auto-reference that people can actually use**
- Add a dedicated notebook: `census_reference_pool_demo.ipynb`
- Keep it embedding-only by default (no concatenation):
  - embed search → joinid set → fetch embeddings/labels → build local ReferencePool
  - query mapping with reject thresholds

3) **Phase C: benchmarking harness**
- `sg.bench.run(...)` to compare integration methods via ScGeo metrics (mixing vs conservation).

## Longer-term roadmap (from original timeline)

- Benchmark harness (Week 3)
- ScGeo-guided correction loop (Weeks 4–5)
- Cross-modal merge/deconvolution (Weeks 6–8)

## Phase A — Geometry Core (MVP)
- Δ vectors (μ₁ − μ₀)
- cosine alignment
- distribution distance (Wasserstein)
- overlap / mixing metrics
- ambiguity / disagreement scores

## Phase B — QC-aware Atlas Mapping
- reference → query mapping
- confidence & OOD detection
- replace naive sc.tl.ingest usage

## Phase C — Reference Pool Integration
- auto-download from cellxgene
- build reusable annotation pools

## Phase D — Trajectory / Velocity / Fate
- scVelo velocity–delta alignment
- CellRank fate basin overlap
- scFates branch geometry comparison
- fate ambiguity detection

## Phase E — Benchmarking & Integration
- batch correction benchmarks
- modality alignment benchmarks

## Phase F — Deconvolution & Multimodal
- geometry-guided deconvolution
- scRNA / spatial / bulk integration
