# Companion notebooks

The [`scgeo-notebooks`](https://github.com/liuifrec/scgeo-notebooks) repository
is the reproducibility companion to this installable library. It is not bundled
with the package and does not store the large generated data objects used on the
frozen workstation.

## Workflow map

| Workflow | Purpose | Inference scope |
|---|---|---|
| Synthetic benchmark | Controlled perturbations, failure modes, and held-out evaluation | Synthetic ground truth only; no biological claim |
| Public pancreas | Developmental dynamics with scVelo and CellRank context | Descriptive dynamics validation |
| Dataset B, GSE249479 | Inflammatory xenograft treatment geometry and official R Augur comparison | `descriptive_only`; no recoverable cell-level biological-replicate identity |
| Dataset C, GSE211713 | Mouse-lung radiation response with independent mouse libraries | Replicate-aware primary associations; cross-sectional, not longitudinal |

The companion also contains the original manuscript workflow, reviewer-facing
metric audits, manuscript figure assembly, and reproducibility gates.

## Repository roles

- **This repository:** installable `scgeo` library, tests, contracts, and
  synthetic benchmark implementation.
- **Companion repository:** source notebooks, workflow wrappers, public-data
  provenance, evidence ledgers, and manuscript-oriented presentation.
- **Local results:** large H5AD files, trained models, executed notebooks, and
  figures generated during validation; these are intentionally not committed.

Begin with the companion’s
[`docs/START_HERE.md`](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/START_HERE.md).
