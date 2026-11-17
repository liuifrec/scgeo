"""Smoke-test an installed ScGeo distribution, without data downloads.

Run outside the source checkout. CI passes --source-root to reject accidentally
importing the working tree instead of the installed wheel. This tiny synthetic
check is an installation test, not a rerun of manuscript validation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


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
    assert "scgeo.bench" not in sys.modules
    assert "bench" in dir(sg)
    if find_spec("resource") is not None:
        assert sg.bench.__name__ == "scgeo.bench"
        assert sg.bench is sys.modules["scgeo.bench"]
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

    # Reproduce Windows' missing Unix resource module on every CI platform.
    resource_probe = """
import importlib.abc
import sys

class NoResource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "resource":
            raise ModuleNotFoundError("Resource blocked by installation test", name="resource")
        return None

sys.meta_path.insert(0, NoResource())
import scgeo
assert "scgeo.bench" not in sys.modules
assert callable(scgeo.tl.robust_shift)
try:
    scgeo.bench
except ImportError as exc:
    assert "Unix-only resource" in str(exc), str(exc)
else:
    raise AssertionError("Unavailable benchmark profiler was not reported")
"""
    subprocess.run([sys.executable, "-I", "-c", resource_probe], check=True)

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
    sample_table = sg.get.sample_centers(result)
    assert sample_table["sample"].nunique() == 6
    assert set(sample_table["coordinate"]) == {0, 1}
    # Exercise the installed plotting additions and packaged source exporter.
    import json
    from tempfile import TemporaryDirectory

    import matplotlib.pyplot as plt
    with TemporaryDirectory(prefix="scgeo-plot-smoke-") as temporary:
        with sg.pl.style_context("paper"):
            plot = sg.pl.replicate_effect(sample_table, coordinate=0, return_result=True, show=False)
            exported = plot.save(Path(temporary) / "replicates", formats=("png", "svg"))
            assert all(Path(path).is_file() for path in exported["figures"].values())
            provenance = json.loads(Path(exported["metadata"]).read_text())
            assert provenance["software"]["scgeo"] == sg.__version__
            assert "sample_centers" in exported["tables"]
            plt.close(plot.fig)
            # A supplied connectivity fixture needs no optional Scanpy import.
            adata.obs["state"] = pd.Categorical(
                (["A"] * 8 + ["B"] * 4) * 2, categories=["A", "B"]
            )
            adata.uns["paga"] = {
                "groups": "state",
                "connectivities": np.array([[0.0, 0.6], [0.6, 0.0]]),
            }
            adata.obsm["velocity_pca"] = np.tile([0.02, 0.01], (adata.n_obs, 1))
            paga = sg.pl.paga_shift_map(
                adata, node_key="state", condition_key="condition",
                group0="control", group1="treated", basis="pca",
                graph_node_order=["A", "B"], min_cells=2,
                show_velocity=True, return_result=True, show=False,
                velocity_mode="grid",
                velocity_options={
                    "grid_size": 20, "n_neighbors": 12,
                    "radius": 0.28, "min_neighbors": 2,
                },
            )
            assert float(paga.axes["map"].get_aspect()) == 1.0
            assert len(paga.data["edges"]) == 1
            graph_export = paga.save(Path(temporary) / "paga", formats=("png", "svg"))
            assert {"edge_table", "display_cells", "vectors"} <= set(graph_export["tables"])
            assert paga.data["velocity_grid"].supported.any()
            assert {"velocity_grid", "velocity_cells", "velocity_vectors"} <= set(graph_export["tables"])
            assert all(Path(path).is_file() for path in graph_export["figures"].values())
            plt.close(paga.fig)
            field = sg.pl.velocity_field(
                adata, basis="pca", condition_key="condition", groups=["control"],
                mode="stream", grid_size=20, n_neighbors=12, radius=0.28,
                min_neighbors=2, background="selected",
                return_result=True, show=False,
            )
            assert field.data["grid"].supported.any()
            assert not field.data["paths"].empty
            assert field.data["cells"].eligible.sum() == 12
            field_export = field.save(Path(temporary) / "velocity", formats=("svg",))
            assert {"cells", "grid", "paths", "vectors"} <= set(field_export["tables"])
            assert Path(field_export["figures"]["svg"]).is_file()
            plt.close(field.fig)
            for kind in ("alluvial", "chord"):
                associations = getattr(sg.pl, f"state_flow_{kind}")(
                    adata, columns=["state", "condition"], min_count=5,
                    return_result=True, show=False,
                )
                assert associations.data["links"]["count"].sum() == adata.n_obs
                visible = associations.data["links"].query("displayed")
                assert visible["count"].sum() == 16
                assert abs(visible.fraction.sum() - 16 / 24) < 1e-12
                flow_export = associations.save(Path(temporary) / kind, formats=("svg",))
                assert {"cells", "nodes", "links", "node_layout", "link_layout"} <= set(flow_export["tables"])
                plt.close(associations.fig)
            density = sg.pl.embedding_density(
                adata, "state", basis="pca", gridsize=12,
                contour=True, smooth_k=3, normalize="global",
                return_result=True, show=False,
            )
            assert density.data["grid"].raw_count.sum() == adata.n_obs
            density_export = density.save(Path(temporary) / "density", formats=("svg",))
            assert {"cells", "grid", "groups", "contour_levels"} <= set(density_export["tables"])
            plt.close(density.fig)
    assert "cellxgene_census" not in sys.modules
    assert "tiledbsoma" not in sys.modules
    print(f"PASS: ScGeo {version('scgeo')} imported from {installed_from}")
    print("PASS: sample-aware displacement on tiny synthetic input; no Census access")
    print("PASS: core import and helpful DE error with optional Scanpy blocked")
    print("PASS: core import is independent of the Unix-only benchmark profiler")
    print("PASS: stored sample centers, plotting API, and source-data figure export")
    print("PASS: PAGA local velocity grid and exported graph/field sources")
    print("PASS: supported projected streamlines with selected-condition source export")
    print("PASS: alluvial/chord count conservation and filtered-link source exports")
    print("PASS: density contour counts, shared scale, and grid source export")


if __name__ == "__main__":
    main()
