# ScGeo

⚠️ **Pre-alpha. APIs may change.**

ScGeo is a scverse-style toolkit for **geometric analysis of single-cell representations**
across conditions, batches, trajectories, and modalities.

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

## Planned scope
- QC-aware atlas mapping & annotation
- cellxgene reference pool integration
- batch correction benchmarking
- trajectory / velocity / fate geometry
- cross-modality (scRNA / spatial / bulk) analysis

# scgeo
