# Reproducibility

## Frozen package checkpoint

The package checkpoint used by the major-revision evidence package is commit
[`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`](https://github.com/liuifrec/scgeo/tree/9a0ed16cbaa57f935f9c9bc87d1643a25b51012c).
Check the active checkout before reproducing a frozen analysis.

```bash
git rev-parse HEAD
git status --short
```

## Test command

From the package repository:

```bash
python -m pytest -q
```

The revision checkpoint is expected to run the complete test suite; warnings
should be recorded separately from failures.

## Benchmark protocol

The frozen synthetic protocol is defined in
[`docs/revision/BENCHMARK_PROTOCOL.md`](revision/BENCHMARK_PROTOCOL.md). It
separates calibration seeds from held-out evaluation seeds, keeps thresholds
fixed for final evaluation, and treats one simulation job/seed as the
independent unit.

## Companion repository

The [`scgeo-notebooks`](https://github.com/liuifrec/scgeo-notebooks) repository
contains source notebooks, execution scripts, environment descriptions, public
validation workflows, manuscript assembly, and reviewer evidence ledgers. Its
documentation pins package commits and numerical input checksums where needed.

## Source and generated artifacts

Source notebooks are tracked with zero embedded outputs and null execution
counts. Clean-kernel execution produces review copies, figures, figure-source
tables, alt text, logs, models, and large data artifacts under ignored result
or workstation data directories. Those generated artifacts are validated by
checksums but are not part of the installable package.

Generated API contracts in this repository are documented in
[`docs/api_reference.md`](api_reference.md); the library API remains the source
of truth for function names.
