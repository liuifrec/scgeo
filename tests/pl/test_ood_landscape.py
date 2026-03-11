from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scgeo.pl import ood_landscape


def _toy_adata():
    n = 24
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                categories=["A", "B", "C"],
            ),
            "scgeo_ood": [
                0.05, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.18,   # A
                0.20, 0.22, 0.24, 0.26, 0.30, 0.32, 0.34, 0.38,   # B
                0.40, 0.45, 0.50, 0.60, 0.70, 0.82, 0.90, 0.98,   # C
            ],
        },
        index=[f"cell_{i}" for i in range(n)],
    )

    adata = AnnData(X=np.zeros((n, 3), dtype=float), obs=obs)

    x = np.concatenate([
        np.linspace(0.0, 1.0, 8),
        np.linspace(2.0, 3.0, 8),
        np.linspace(4.0, 5.0, 8),
    ])
    y = np.concatenate([
        np.linspace(0.0, 0.4, 8),
        np.linspace(0.5, 1.0, 8),
        np.linspace(1.2, 1.8, 8),
    ])
    adata.obsm["X_umap"] = np.c_[x, y]
    return adata


def test_ood_landscape_smoke_return_data_no_groupby():
    adata = _toy_adata()

    fig, ax, out = ood_landscape(
        adata,
        ood_key="scgeo_ood",
        basis="umap",
        threshold=0.8,
        contour=True,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(out, dict)
    assert "flagged" in out
    assert "threshold_used" in out
    assert "summary" in out
    assert out["summary"] is None
    assert int(out["flagged"].sum()) == 3  # 0.82, 0.90, 0.98


def test_ood_landscape_group_summary():
    adata = _toy_adata()

    fig, axes, out = ood_landscape(
        adata,
        ood_key="scgeo_ood",
        basis="umap",
        threshold=0.5,
        groupby="group",
        top_n_groups=3,
        return_data=True,
        show=False,
    )

    ax_main, ax_bar = axes
    assert fig is not None
    assert ax_main is not None
    assert ax_bar is not None
    assert out["summary"] is not None
    summary = out["summary"].set_index("group")
    assert "flagged_frac" in summary.columns
    assert summary.loc["C", "flagged_frac"] > summary.loc["B", "flagged_frac"]
    assert summary.loc["B", "flagged_frac"] >= summary.loc["A", "flagged_frac"]


def test_ood_landscape_quantile_threshold_when_none():
    adata = _toy_adata()

    _, _, out = ood_landscape(
        adata,
        ood_key="scgeo_ood",
        basis="umap",
        threshold=None,
        return_data=True,
        show=False,
    )

    assert out["threshold_used"] is not None
    assert np.isfinite(out["threshold_used"])


def test_ood_landscape_show_only_flagged():
    adata = _toy_adata()

    fig, ax, out = ood_landscape(
        adata,
        ood_key="scgeo_ood",
        basis="umap",
        threshold=0.7,
        show_only_flagged=True,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert int(out["flagged"].sum()) == 4  # 0.70, 0.82, 0.90, 0.98


def test_ood_landscape_missing_basis_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="Embedding"):
        ood_landscape(
            adata,
            ood_key="scgeo_ood",
            basis="tsne",
            show=False,
        )


def test_ood_landscape_missing_ood_key_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="not found"):
        ood_landscape(
            adata,
            ood_key="missing_ood",
            basis="umap",
            show=False,
        )


def test_ood_landscape_invalid_summary_kind_raises():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="summary_kind"):
        ood_landscape(
            adata,
            ood_key="scgeo_ood",
            basis="umap",
            groupby="group",
            summary_kind="weird_metric",
            show=False,
        )


def test_ood_landscape_accepts_existing_ax_when_no_groupby():
    import matplotlib.pyplot as plt

    adata = _toy_adata()
    fig, ax = plt.subplots(figsize=(4, 4))

    out = ood_landscape(
        adata,
        ood_key="scgeo_ood",
        basis="umap",
        threshold=0.8,
        ax=ax,
        show=False,
    )

    assert out is ax


def test_ood_landscape_existing_ax_with_groupby_raises():
    import matplotlib.pyplot as plt

    adata = _toy_adata()
    fig, ax = plt.subplots(figsize=(4, 4))

    with pytest.raises(ValueError, match="Pass ax only when groupby is None"):
        ood_landscape(
            adata,
            ood_key="scgeo_ood",
            basis="umap",
            groupby="group",
            ax=ax,
            show=False,
        )