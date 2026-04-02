<p align="center">
  <img src="docs/assets/ScGeo_logo.png" width="420">
</p>

<h1 align="center">ScGeo</h1>

<p align="center">
Geometry-aware analysis of single-cell representations
</p>


ScGeo is a geometry-aware framework for single-cell analysis that treats low-dimensional embeddings as quantitative representations of cellular state space.

It enables:

- measurement of perturbation-driven state transitions (Δ-shift)
- evaluation of integration via local mixing structure
- detection of global redistribution (distributional divergence)
- alignment of embedding geometry with RNA velocity and fate inference

Importantly, ScGeo reveals structured biological dynamics and non-canonical trajectories that are not fully captured by RNA velocity alone.

## Installation

```bash
git clone https://github.com/liuifrec/scgeo.git
cd scgeo
pip install -e .

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
- `scgeo.tl.velocity_shift_alignment` — geometry–velocity consistency

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


## Mapping to manuscript concepts

| Manuscript concept | API |
|-------------------|-----|
| Geometric displacement (Δ) | scgeo.tl.shift |
| Local mixing | scgeo.tl.mixscore |
| Distribution divergence | scgeo.tl.distribution_test |
| Geometry–velocity alignment | scgeo.tl.velocity_shift_alignment |
| OOD detection | scgeo.tl.ood_cells |
| Composition drift | scgeo.pl.composition_drift |
| Recovery trajectory visualization | scgeo.pl.recovery_compass |
## Minimal example

```python
import scgeo as sg

# compute geometry
sg.tl.shift(adata)
sg.tl.mixscore(adata)
sg.tl.distribution_test(adata)

# analyze dynamics
sg.tl.velocity_shift_alignment(adata)

# visualize
sg.pl.recovery_compass(adata)
```

## Manuscript

ScGeo is introduced and validated in:

"ScGeo reveals non-canonical trajectories beyond RNA velocity in radiation-induced hematopoietic recovery"

All analysis workflows and figure-generation notebooks are available in:
https://github.com/liuifrec/scgeo-notebooks


## Citation

If you use ScGeo, please cite:

Liu Y-C, Yoshida K.  
*ScGeo reveals non-canonical trajectories beyond RNA velocity in radiation-induced hematopoietic recovery.*

## Manifest layers (reproducible contracts)

ScGeo tracks public and I/O contracts across three aligned JSON manifests:

- `api_manifest.json`: exported public API (`scgeo.tl` and `scgeo.pl`)
- `scgeo_io_raw.json`: raw write-diff observations for TL functions
- `scgeo_io_manifest.json`: normalized TL I/O contract

Rebuild all manifests in one step:

```bash
PYTHONPATH=. python scripts/rebuild_manifests.py
```

Validate the alignment/importability checks:

```bash
PYTHONPATH=. python scripts/validate_manifests.py
```

Public API docs generated from these manifests are available at:

- `docs/api_reference.md`