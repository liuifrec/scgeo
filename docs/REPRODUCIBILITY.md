# Reproducibility and version provenance

[Citation guide](CITING.md) · [Companion workflows](https://github.com/liuifrec/scgeo-notebooks) · [Installation](INSTALLATION.md)

## Choose the right starting point

| Task | Starting point | Record with the analysis |
| --- | --- | --- |
| New analysis using the released API | PyPI ScGeo 0.1.0 | Package/dependency versions, inputs, representations and settings |
| Development against current source | A specified Git commit | Full commit, local changes, resolved environment and tests |
| Reproduction of the published evidence | Frozen package and companion checkpoints below | Recorded workflow environments, numerical inputs/checksums, configurations and execution metadata |

The [published study](https://doi.org/10.1093/bioadv/vbag266) identifies the
scientific framework. Its article DOI and the software version have different
roles: cite the paper and record the exact code and environment used.

## Frozen manuscript checkpoints

- ScGeo package:
  [`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`](https://github.com/liuifrec/scgeo/tree/9a0ed16cbaa57f935f9c9bc87d1643a25b51012c).
- Companion merge checkpoint:
  [`958a0ada568c53e78864ede022c050830cb55a36`](https://github.com/liuifrec/scgeo-notebooks/tree/958a0ada568c53e78864ede022c050830cb55a36).
- Dataset-specific inputs, outputs, and settings: the recorded execution metadata,
  validation reports, and evidence ledgers in the companion workflows.

These checkpoints remain the reproduction anchors for the manuscript evidence.
The later PyPI release and documentation improvements do not replace them.
Begin with the companion's
[reproducibility guide](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/REPRODUCIBILITY_GUIDE.md)
before executing a dataset workflow; some scripts retain documented workstation
paths and interpreter requirements.

## Capture the environment for a new analysis

```bash
python -c "import sys, scgeo; print(sys.version); print(scgeo.__version__); print(scgeo.__file__)"
python -m pip freeze > requirements-resolved.txt
```

For a source checkout, also record:

```bash
git rev-parse HEAD
git status --short
```

Preserve input accessions and checksums, preprocessing and representation
settings, sample/state/condition mappings, random seeds, bootstrap settings,
comparison units, and output tables. A Git commit or a `pip freeze` listing alone
does not capture the complete analytical provenance.

## Package tests and documentation checks

In a development environment installed with `.[dev,scanpy]`:

```bash
python -m pytest -q -m "not integration"
python scripts/validate_manifests.py
python scripts/gen_api_docs.py --check
```

Record any skipped network/integration tests and warnings separately from
failures. Package tests and installation smoke checks establish implementation
behavior; reproducing the paper also requires its data, workflow settings, and
recorded execution environments.

## Benchmarks and generated artifacts

The [frozen synthetic protocol](revision/BENCHMARK_PROTOCOL.md) separates
calibration from held-out evaluation seeds, fixes thresholds for final
evaluation, and treats simulation jobs/seeds as the independent units.

Source notebooks are tracked without embedded outputs or execution counts.
Executed notebooks, manuscript figures and figure-source tables, logs, large
data objects, and models belong in the workflow's ignored result or local data
directories. The small PNG under `docs/assets/quickstart_displacement.png` is a
reproducible synthetic documentation illustration from
[`examples/quickstart.py`](../examples/quickstart.py), not a manuscript figure.

## Release and historical tag names

The published package release is `v0.1.0`, with source commit
`6ab8a8ea4144f66613812b7b79976b1a67ad29d1`. The earlier tag
`v0.2.0-geometry-core` identifies a March development checkpoint whose package
metadata was `0.1.0-dev`; it is not a newer published package release.
Historical tags are retained for provenance. See the
[release record](RELEASING.md) and [PyPI publication record](PYPI_PUBLICATION.md).
