from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scgeo.pl import state_flow_alluvial


def _toy_adata():
    obs = pd.DataFrame(
        {
            "timepoint": ["D8", "D8", "D11", "D11", "D21", "D21", "D21"],
            "alignment_group": ["aligned", "discordant", "aligned", "other", "discordant", "aligned", "aligned"],
            "macrostate": ["A", "B", "A", "C", "B", "A", "A"],
        },
        index=[f"cell_{i}" for i in range(7)],
    )
    return AnnData(X=np.zeros((7, 3)), obs=obs)


def test_state_flow_alluvial_smoke_return_data():
    adata = _toy_adata()

    fig, ax, stage_boxes, alloc_df = state_flow_alluvial(
        adata,
        columns=["timepoint", "alignment_group", "macrostate"],
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert not stage_boxes.empty
    assert not alloc_df.empty
    assert {"stage", "label", "x", "y0", "y1", "value"}.issubset(stage_boxes.columns)
    assert {"left_stage", "right_stage", "left_label", "right_label", "value"}.issubset(alloc_df.columns)


def test_state_flow_alluvial_requires_two_columns():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="at least 2"):
        state_flow_alluvial(
            adata,
            columns=["timepoint"],
            show=False,
        )


def test_state_flow_alluvial_missing_column_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="not found"):
        state_flow_alluvial(
            adata,
            columns=["timepoint", "missing_col"],
            show=False,
        )


def test_state_flow_alluvial_min_count_filters():
    adata = _toy_adata()

    fig, ax, stage_boxes, alloc_df = state_flow_alluvial(
        adata,
        columns=["timepoint", "alignment_group", "macrostate"],
        min_count=2,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert (alloc_df["value"] > 0).all()


def test_state_flow_alluvial_color_by_validation():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="color_by"):
        state_flow_alluvial(
            adata,
            columns=["timepoint", "alignment_group"],
            color_by="weird",
            show=False,
        )