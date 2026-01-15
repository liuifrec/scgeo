import numpy as np
import pandas as pd
import anndata as ad
import scgeo as sg


def _dummy_adata():
    adata = ad.AnnData(X=np.zeros((10, 1), dtype=np.float32))
    df = pd.DataFrame(
        {
            "node": ["A", "B", "C", "D"],
            "logOR": [1.2, -0.5, 0.1, -2.0],
            "p": [1e-3, 0.2, 0.9, 1e-6],
            "q": [2e-3, 0.25, 0.9, 1e-5],
        }
    )
    adata.uns["scgeo"] = {"paga_composition_stats": {"table": df}}
    return adata


def test_paga_composition_volcano_runs():
    adata = _dummy_adata()
    fig, axes = sg.pl.paga_composition_volcano(adata, show=False)
    assert fig is not None
    assert len(axes) == 1


def test_paga_composition_bar_runs():
    adata = _dummy_adata()
    fig, axes = sg.pl.paga_composition_bar(adata, show=False)
    assert fig is not None
    assert len(axes) == 1


def test_paga_composition_panel_runs():
    adata = _dummy_adata()
    fig, axes = sg.pl.paga_composition_panel(adata, show=False)
    assert fig is not None
    assert len(axes) == 2
