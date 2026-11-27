# Explore the paper's workflows

The [`scgeo-notebooks`](https://github.com/liuifrec/scgeo-notebooks) repository
connects the ScGeo paper to its source notebooks, workflow implementations,
dataset provenance, output maps, and evidence records.

**[Start here](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/START_HERE.md) ·
[Dataset map](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/DATASET_MAP.md) ·
[Results overview](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/RESULTS_OVERVIEW.md)**

## Learn the released API with controlled simulations

Start with [How the tutorial data are generated](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/SIMULATION_DESIGN.md), a visual explanation of the exact input models used in the companion's [0.2.0 notebook series](https://github.com/liuifrec/scgeo-notebooks/blob/main/notebooks/tutorials/v0.2.0/README.md). The guide shows the planted geometry, molecular-response functions, sample capture and known support failures, with links to the immutable example sources. Eleven lessons use synthetic inputs; the last is an optional checksum-matched public-data exercise. These teaching fixtures are distinct from the paper's frozen benchmarks and biological studies.

## Find an application or validation setting

| Workflow | What to explore | Inference scope |
| --- | --- | --- |
| [GSE280305 hematopoietic recovery](https://github.com/liuifrec/scgeo-notebooks/blob/main/notebooks/manuscript/README.md) | The original radiation-recovery application, state composition and displacement, and dynamics context | D8 is an early post-irradiation reference, not an untreated baseline |
| [Synthetic benchmark](https://github.com/liuifrec/scgeo-notebooks/tree/main/notebooks/benchmarks) | Controlled effects, sampling, representation corruption, and failure modes | Stored synthetic truth; no biological claim |
| [Public pancreas](https://github.com/liuifrec/scgeo-notebooks/tree/main/notebooks/public_validation/pancreas) | Geometry alongside scVelo/CellRank developmental context | Descriptive dynamics validation |
| [GSE249479](https://github.com/liuifrec/scgeo-notebooks/tree/main/notebooks/public_validation/gse249479) | Human HSPC inflammatory geometry and official R Augur comparison | Descriptive; no recoverable biological-replicate identity |
| [GSE211713](https://github.com/liuifrec/scgeo-notebooks/tree/main/notebooks/public_validation/gse211713) | Mouse-lung radiation response with 20 independent mouse libraries | Replicate-aware cross-sectional association |

The dataset map identifies entry notebooks, implementations, environment files,
and output locations. The results overview explains both supported conclusions
and limitations without recomputing the frozen results.

## How the repositories fit together

| Repository or output | Role |
| --- | --- |
| `scgeo` | Installable library, tests, public API, and benchmark implementation |
| `scgeo-notebooks` | Workflow source, public-data provenance, figure assembly, and evidence records |
| Local generated outputs | Executed notebooks, figures/source tables, data objects, models, and logs |

The companion's earlier lightweight synthetic reporting notebook is a conceptual
NumPy/pandas example. To learn actual ScGeo API calls, use the package's
[quick start](QUICKSTART.md) or the [released 0.2.0 tutorial series](https://github.com/liuifrec/scgeo-notebooks/blob/main/notebooks/tutorials/v0.2.0/README.md). For manuscript reproduction, follow the
[frozen provenance](REPRODUCIBILITY.md) and workflow-specific environments.
For reuse in a new publication, see [Citing ScGeo](CITING.md).
