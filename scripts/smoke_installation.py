"""Smoke-test an installed ScGeo distribution, without data downloads.

Run outside the source checkout. CI passes --source-root to reject accidentally
importing the working tree instead of the installed wheel. This tiny synthetic
check is an installation test, not a rerun of manuscript validation.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from packaging.version import Version
    import scgeo as sg

    installed_from = Path(sg.__file__).resolve()
    if args.source_root is not None:
        assert not installed_from.is_relative_to(args.source_root.resolve()), (
            f"Imported the source checkout, not the installed distribution: {installed_from}"
        )
    assert Version(version("scgeo")) == Version(sg.__version__)
    for name in ("robust_shift", "representation_stability", "local_geometry_stability"):
        assert callable(getattr(sg.tl, name)), name

    # Regression check: core import must work even in an environment where
    # Scanpy cannot be imported. The DE helper must explain its optional need.
    optional_probe = """
import importlib.abc
import sys

class NoScanpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "scanpy":
            raise ModuleNotFoundError("Scanpy blocked by installation test", name="scanpy")
        return None

sys.meta_path.insert(0, NoScanpy())
import scgeo
try:
    scgeo.tl.alignment_driver_genes(None, alignment_key="alignment")
except ImportError as exc:
    assert "python -m pip install scanpy" in str(exc), str(exc)
else:
    raise AssertionError("Missing optional Scanpy did not produce an installation hint")
"""
    subprocess.run([sys.executable, "-I", "-c", optional_probe], check=True)

    # The same three sample patterns occur in each condition with a known shift.
    pattern = np.array([[-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]])
    centers = np.array([[0, 0], [0.2, 0.1], [-0.1, 0.2]])
    blocks, conditions, samples = [], [], []
    for condition, shift in (("control", 0.0), ("treated", 1.0)):
        for sample, center in enumerate(centers):
            blocks.append(pattern + center + np.array([shift, 0.0]))
            conditions.extend([condition] * len(pattern))
            samples.extend([f"{condition}_{sample}"] * len(pattern))
    coordinates = np.vstack(blocks)
    obs = pd.DataFrame(
        {"condition": conditions, "sample": samples},
        index=[f"cell_{i}" for i in range(len(coordinates))],
    )
    adata = AnnData(X=coordinates.copy(), obs=obs)
    adata.obsm["X_pca"] = coordinates
    result = sg.tl.robust_shift(
        adata, rep="X_pca", condition_key="condition", group0="control",
        group1="treated", sample_key="sample", center="median", n_boot=12,
    )
    np.testing.assert_allclose(result["global"]["delta"], [1.0, 0.0], atol=1e-6)
    assert result["params"]["resolved_bootstrap_unit"] == "sample"
    assert result["global"]["n_samples0"] == 3
    assert result["global"]["n_samples1"] == 3
    assert adata.uns["scgeo"]["robust_shift"] is result
    assert "cellxgene_census" not in sys.modules
    assert "tiledbsoma" not in sys.modules
    print(f"PASS: ScGeo {version('scgeo')} imported from {installed_from}")
    print("PASS: sample-aware displacement on tiny synthetic input; no Census access")
    print("PASS: core import and helpful DE error with optional Scanpy blocked")


if __name__ == "__main__":
    main()
