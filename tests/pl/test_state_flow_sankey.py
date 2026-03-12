from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scgeo.pl import state_flow_sankey


def _toy_adata():
    obs = pd.DataFrame(
        {
            "timepoint": ["D8", "D8", "D11", "D11", "D21", "D21"],
            "alignment_group": ["aligned", "discordant", "aligned", "other", "discordant", "aligned"],
            "macrostate": ["A", "B", "A", "C", "B", "A"],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    return AnnData(X=np.zeros((6, 3)), obs=obs)


def test_state_flow_sankey_smoke_return_data():
    adata = _toy_adata()

    fig, nodes_df, links_df = state_flow_sankey(
        adata,
        columns=["timepoint", "alignment_group", "macrostate"],
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert not nodes_df.empty
    assert not links_df.empty
    assert {"node_key", "label", "stage", "column", "node_id"}.issubset(nodes_df.columns)
    assert {"source", "target", "value", "source_col", "target_col"}.issubset(links_df.columns)


def test_state_flow_sankey_requires_two_columns():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="at least 2"):
        state_flow_sankey(
            adata,
            columns=["timepoint"],
            show=False,
        )


def test_state_flow_sankey_missing_column_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="not found"):
        state_flow_sankey(
            adata,
            columns=["timepoint", "missing_col"],
            show=False,
        )


def test_state_flow_sankey_min_count_filters():
    adata = _toy_adata()

    fig, nodes_df, links_df = state_flow_sankey(
        adata,
        columns=["timepoint", "alignment_group", "macrostate"],
        min_count=2,
        return_data=True,
        show=False,
    )

    assert fig is not None
    # some links should be filtered out
    assert (links_df["value"] >= 2).all()