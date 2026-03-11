from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import matplotlib
matplotlib.use("Agg")

from scgeo.pl import velocity_shift_alignment


def _toy_adata():
    # 12 cells, 3 nodes, 2 timepoints
    obs = pd.DataFrame(
        {
            "node": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
                categories=["A", "B", "C"],
            ),
            "timepoint": ["D08", "D08", "D21", "D21"] * 3,
        },
        index=[f"cell_{i}" for i in range(12)],
    )

    adata = AnnData(X=np.zeros((12, 3), dtype=float), obs=obs)

    # A shifts right; B shifts up; C shifts left/down a bit
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.2, 0.0],   # A
            [3.0, 0.0], [3.2, 0.0], [3.0, 1.0], [3.2, 1.0],   # B
            [6.0, 1.0], [6.2, 1.0], [5.7, 0.7], [5.9, 0.7],   # C
        ],
        dtype=float,
    )

    # Mean velocity aligned with A, aligned with B, anti-aligned-ish with C
    adata.obsm["velocity_umap"] = np.array(
        [
            [0.8, 0.0], [1.0, 0.0], [0.9, 0.0], [1.1, 0.0],   # A
            [0.0, 0.8], [0.0, 1.0], [0.0, 0.9], [0.0, 1.1],   # B
            [0.6, 0.6], [0.5, 0.5], [0.6, 0.5], [0.5, 0.6],   # C
        ],
        dtype=float,
    )
    adata.uns["node_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    return adata


def test_velocity_shift_alignment_smoke_return_data():
    adata = _toy_adata()

    fig, ax, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(align, pd.DataFrame)
    assert {
        "node",
        "dx",
        "dy",
        "vx",
        "vy",
        "alignment_cosine",
        "abs_alignment_cosine",
        "alignment_class",
        "usable",
    }.issubset(align.columns)

    idx = align.set_index("node")
    assert np.isclose(idx.loc["A", "dx"], 1.0, atol=1e-6)
    assert np.isclose(idx.loc["B", "dy"], 1.0, atol=1e-6)
    assert idx.loc["A", "alignment_cosine"] > 0.9
    assert idx.loc["B", "alignment_cosine"] > 0.9
    assert idx.loc["A", "alignment_class"] == "aligned"
    assert idx.loc["B", "alignment_class"] == "aligned"


def test_velocity_shift_alignment_detects_negative_alignment():
    adata = _toy_adata()

    _, _, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        return_data=True,
        show=False,
    )

    idx = align.set_index("node")
    assert idx.loc["C", "alignment_cosine"] < 0.0
    assert idx.loc["C", "alignment_class"] == "discordant"


def test_velocity_shift_alignment_neutral_class_thresholds():
    adata = _toy_adata()

    # Make B nearly orthogonal / near-zero alignment
    adata.obsm["velocity_umap"][4:8, :] = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    _, _, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        alignment_pos_thr=0.3,
        alignment_neg_thr=-0.3,
        return_data=True,
        show=False,
    )

    idx = align.set_index("node")
    assert abs(idx.loc["B", "alignment_cosine"]) < 0.3
    assert idx.loc["B", "alignment_class"] == "neutral"


def test_velocity_shift_alignment_missing_velocity_raises():
    adata = _toy_adata()
    del adata.obsm["velocity_umap"]

    with pytest.raises(KeyError, match="Velocity embedding"):
        velocity_shift_alignment(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            show=False,
        )


def test_velocity_shift_alignment_missing_basis_raises():
    adata = _toy_adata()

    with pytest.raises(KeyError, match="Embedding"):
        velocity_shift_alignment(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="tsne",
            show=False,
        )


def test_velocity_shift_alignment_accepts_existing_ax():
    import matplotlib.pyplot as plt

    adata = _toy_adata()
    fig, ax = plt.subplots(figsize=(4, 4))

    out = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        ax=ax,
        show=False,
    )

    assert out is ax


def test_velocity_shift_alignment_min_cells_masks_nodes():
    adata = _toy_adata()

    # Drop one B cell from D21 so B has only one D21 cell
    keep = ~(
        (adata.obs["node"] == "B")
        & (adata.obs["timepoint"] == "D21")
        & (adata.obs_names == "cell_7")
    )
    adata = adata[keep].copy()

    _, _, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=2,
        return_data=True,
        show=False,
    )

    idx = align.set_index("node")
    assert bool(idx.loc["B", "present0"]) is True
    assert bool(idx.loc["B", "present1"]) is False
    assert bool(idx.loc["B", "usable"]) is False


def test_velocity_shift_alignment_label_mode_validation():
    adata = _toy_adata()

    with pytest.raises(ValueError, match="label_mode"):
        velocity_shift_alignment(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D08",
            group1="D21",
            basis="umap",
            label_mode="weird_mode",
            show=False,
        )


def test_velocity_shift_alignment_show_arrow_toggles():
    adata = _toy_adata()

    fig, ax, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        show_shift_arrow=False,
        show_velocity_arrow=True,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert isinstance(align, pd.DataFrame)


def test_velocity_shift_alignment_alignment_class_values():
    adata = _toy_adata()

    _, _, align = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D08",
        group1="D21",
        basis="umap",
        min_cells=1,
        return_data=True,
        show=False,
    )

    allowed = {"aligned", "neutral", "discordant", "missing"}
    assert set(align["alignment_class"].dropna().unique()).issubset(allowed)