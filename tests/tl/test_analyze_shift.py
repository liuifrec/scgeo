from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scgeo.tl import analyze_shift


def _toy_adata():
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
                categories=["A", "B", "C"],
            ),
            "timepoint": ["D08", "D08", "D21", "D21"] * 3,
            "scgeo_ood": [0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9],
        },
        index=[f"cell_{i}" for i in range(12)],
    )
    adata = AnnData(X=np.zeros((12, 2)), obs=obs)
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.2, 0.0],
            [3.0, 0.0], [3.2, 0.0], [3.0, 1.0], [3.2, 1.0],
            [6.0, 1.0], [6.2, 1.0], [5.7, 0.7], [5.9, 0.7],
        ],
        dtype=float,
    )
    adata.obsm["velocity_umap"] = np.array(
        [
            [0.8, 0.0], [1.0, 0.0], [0.9, 0.0], [1.1, 0.0],
            [0.0, 0.8], [0.0, 1.0], [0.0, 0.9], [0.0, 1.1],
            [0.6, 0.6], [0.5, 0.5], [0.6, 0.5], [0.5, 0.6],
        ],
        dtype=float,
    )
    return adata


def test_analyze_shift_smoke():
    adata = _toy_adata()
    analyze_shift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        ood_key="scgeo_ood",
        store_key="shift",
    )

    assert "scgeo" in adata.uns
    assert "shift" in adata.uns["scgeo"]

    block = adata.uns["scgeo"]["shift"]
    assert "shift_summary" in block
    assert "composition" in block
    assert "velocity_alignment" in block
    assert "ood_summary" in block

    assert isinstance(block["shift_summary"], pd.DataFrame)
    assert isinstance(block["composition"], pd.DataFrame)
    assert isinstance(block["velocity_alignment"], pd.DataFrame)
    assert isinstance(block["ood_summary"], pd.DataFrame)


def test_analyze_shift_with_robustness():
    adata = _toy_adata()
    robust = pd.DataFrame(
        {
            "feature": ["f1", "f1", "f2", "f2"],
            "setting": ["raw", "scanorama", "raw", "scanorama"],
            "value": [0.9, 0.8, 0.7, 0.6],
        }
    )

    analyze_shift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        robustness=robust,
    )

    assert "robustness" in adata.uns["scgeo"]["shift"]
    assert isinstance(adata.uns["scgeo"]["shift"]["robustness"], pd.DataFrame)


def test_analyze_shift_overwrite_false_raises():
    adata = _toy_adata()
    analyze_shift(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
    )

    with pytest.raises(ValueError, match="already exists"):
        analyze_shift(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            overwrite=False,
        )


def test_analyze_shift_missing_basis_raises():
    adata = _toy_adata()
    with pytest.raises(KeyError, match="Embedding"):
        analyze_shift(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="tsne",
        )