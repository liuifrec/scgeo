<p align="center">
  <img src="docs/assets/ScGeo_logo.png" width="420">
</p>

<h1 align="center">ScGeo</h1>

<p align="center">
Geometry-aware analysis of single-cell representations
</p>

# ScGeo

ScGeo is a geometry-aware framework for single-cell transcriptomics that quantifies embedding structure and links it to dynamical inference.

ScGeo is a scverse-style toolkit for **geometric analysis of single-cell representations**
across conditions, batches, trajectories, and modalities.

It provides tools to measure perturbation-driven state transitions in low-dimensional embeddings through:
- geometric displacement (Δ-shift)
- local neighborhood mixing (mixscore)
- distributional divergence
- velocity–geometry alignment

ScGeo enables systematic analysis of embedding-level dynamics beyond RNA velocity.



## Core questions ScGeo answers

| Question                                  | Geometric tool              |
|-------------------------------------------|-----------------------------|
| How different are two conditions overall? | Wasserstein distance        |
| How much do populations overlap?          | Bhattacharyya / kNN mixing  |
| Are two responses aligned?                | cosine(Δ₁, Δ₂)              |
| Which cells drive the difference?         | consensus subspace          |
| Where are ambiguous cells?                | projection disagreement     |

ScGeo complements Scanpy, scVelo, CellRank, and scFates by making
**representation geometry explicit and measurable**.

## Core functions

- `scgeo.tl.shift` — geometric displacement between conditions
- `scgeo.tl.mixscore` — local neighborhood mixing
- `scgeo.tl.distribution_test` — embedding-level divergence
- `scgeo.tl.velocity_delta_alignment` — geometry–velocity consistency

## Planned scope
- QC-aware atlas mapping & annotation
- cellxgene reference pool integration
- batch correction benchmarking
- trajectory / velocity / fate geometry
- cross-modality (scRNA / spatial / bulk) analysis

# scgeo
## Core Features (v0.2)

- Geometry-aware reference mapping (Census / local)
- Velocity–embedding alignment metrics
- Driver gene identification via geometric shift
- OOD detection in embedding space

## Manuscript

ScGeo is introduced and validated in:

"ScGeo reveals non-canonical trajectories beyond RNA velocity in radiation-induced hematopoietic recovery"

All analysis workflows and figure-generation notebooks are available in:
https://github.com/liuifrec/scgeo-notebooks