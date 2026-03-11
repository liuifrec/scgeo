from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scgeo.get import (
    get_available_tables,
    get_composition_table,
    get_ood_summary,
    get_robustness_table,
    get_shift_summary,
    get_velocity_alignment_summary,
)


def _toy_adata():
    adata = AnnData(X=np.zeros((4, 2)))
    adata.uns["scgeo"] = {
        "shift": {
            "shift_summary": pd.DataFrame({"node": ["A"], "shift_norm": [1.0]}),
            "velocity_alignment": pd.DataFrame({"node": ["A"], "alignment_cosine": [0.8]}),
            "composition": pd.DataFrame({"node": ["A"], "delta_frac": [0.1]}),
            "ood_summary": pd.DataFrame({"group": ["A"], "flagged_frac": [0.2]}),
            "robustness": pd.DataFrame({"feature": ["x"], "setting": ["raw"], "value": [0.9]}),
            "params": {"basis": "umap"},
        }
    }
    return adata


def test_get_shift_summary():
    adata = _toy_adata()
    df = get_shift_summary(adata)
    assert isinstance(df, pd.DataFrame)
    assert "shift_norm" in df.columns


def test_get_velocity_alignment_summary():
    adata = _toy_adata()
    df = get_velocity_alignment_summary(adata)
    assert "alignment_cosine" in df.columns


def test_get_composition_table():
    adata = _toy_adata()
    df = get_composition_table(adata)
    assert "delta_frac" in df.columns


def test_get_ood_summary():
    adata = _toy_adata()
    df = get_ood_summary(adata)
    assert "flagged_frac" in df.columns


def test_get_robustness_table():
    adata = _toy_adata()
    df = get_robustness_table(adata)
    assert "value" in df.columns


def test_get_available_tables():
    adata = _toy_adata()
    out = get_available_tables(adata)
    assert "shift_summary" in out
    assert "params" not in out


def test_missing_store_key_raises():
    adata = _toy_adata()
    with pytest.raises(KeyError, match="ghost"):
        get_shift_summary(adata, store_key="ghost")