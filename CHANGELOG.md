# Changelog

## 0.1.0

First non-development release candidate for distribution. A version in the
repository is not evidence of an upload to PyPI or Bioconda.

- Retains the existing ScGeo analysis functions and published-paper citation.
- Includes the pip/conda installation improvements merged in PR #10: local
  geometry and plotting dependencies are installed by default; Census is
  optional; Scanpy-dependent differential expression and Unix-only benchmark
  profiling are loaded only when needed.
- Adds release-archive checks, library tests, optional-extra installation checks,
  and a native conda build from the actual source distribution.
- Adds approval-gated Trusted Publishing to TestPyPI and verifies the uploaded
  files against the exact tested SHA256 checksums. Production PyPI publication
  and Bioconda submission remain separate, explicitly authorized steps.

Python 3.12 is the recommended starting point. Package metadata retains Python
>=3.9; see the CI results for the precise tested combinations. Core installation
checks cover Linux, Windows, and macOS. Benchmark profiling still requires Linux
(including WSL) or macOS. Optional-extra installation checks are on Linux/Python
3.12 and do not certify every upstream workflow or live Census service.

This release preparation does not change scientific calculations, inference
rules, thresholds, public analysis signatures, manuscript figures, or numerical
results. The manuscript's frozen package checkpoint remains
`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`; version 0.1.0 is not a replacement
checkpoint for reproducing that analysis.
