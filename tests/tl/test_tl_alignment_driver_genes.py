import numpy as np
import pandas as pd
from anndata import AnnData

from scgeo.tl._alignment_driver_genes import alignment_driver_genes


def make_test_adata():
    # 12 cells, 4 genes
    X = np.array(
        [
            [10, 1, 0, 0],  # discordant
            [11, 1, 0, 0],
            [12, 1, 0, 0],
            [10, 1, 0, 0],
            [1, 10, 0, 0],  # aligned
            [1, 11, 0, 0],
            [1, 12, 0, 0],
            [1, 10, 0, 0],
            [8, 1, 0, 0],   # discordant, state B
            [8, 1, 0, 0],
            [1, 8, 0, 0],   # aligned, state B
            [1, 8, 0, 0],
        ],
        dtype=float,
    )

    obs = pd.DataFrame(
        {
            "alignment_group": [
                "discordant", "discordant", "discordant", "discordant",
                "aligned", "aligned", "aligned", "aligned",
                "discordant", "discordant",
                "aligned", "aligned",
            ],
            "cluster_label_manual": [
                "stateA", "stateA", "stateA", "stateA",
                "stateA", "stateA", "stateA", "stateA",
                "stateB", "stateB",
                "stateB", "stateB",
            ],
        },
        index=[f"cell{i}" for i in range(12)],
    )

    var = pd.DataFrame(index=["geneA", "geneB", "geneC", "geneD"])
    return AnnData(X=X, obs=obs, var=var)


def test_alignment_driver_genes_global():
    adata = make_test_adata()

    out = alignment_driver_genes(
        adata,
        alignment_key="alignment_group",
        group1="discordant",
        group2="aligned",
        method="wilcoxon",
        min_cells=2,
        key_added="test_alignment_driver_genes",
    )

    assert isinstance(out, pd.DataFrame)
    assert "test_alignment_driver_genes" in adata.uns
    assert "subset" in out.columns
    assert "group1" in out.columns
    assert "group2" in out.columns
    assert set(out["subset"]) == {"all"}

    # geneA should rank as a discordant-associated gene
    top_names = out["names"].head(5).tolist()
    assert "geneA" in top_names


def test_alignment_driver_genes_by_subset():
    adata = make_test_adata()

    out = alignment_driver_genes(
        adata,
        alignment_key="alignment_group",
        group1="discordant",
        group2="aligned",
        subset_key="cluster_label_manual",
        subset_values=["stateA", "stateB"],
        method="wilcoxon",
        min_cells=2,
        key_added="test_alignment_driver_genes_by_subset",
    )

    assert isinstance(out, pd.DataFrame)
    assert "test_alignment_driver_genes_by_subset" in adata.uns
    assert set(out["subset"].unique()) == {"stateA", "stateB"}

    stateA = out[out["subset"] == "stateA"]
    stateB = out[out["subset"] == "stateB"]

    assert "geneA" in stateA["names"].head(5).tolist()
    assert "geneA" in stateB["names"].head(5).tolist()


def test_alignment_driver_genes_missing_key_raises():
    adata = make_test_adata()

    try:
        alignment_driver_genes(
            adata,
            alignment_key="not_here",
            group1="discordant",
            group2="aligned",
        )
    except KeyError as e:
        assert "not_here" in str(e)
    else:
        raise AssertionError("Expected KeyError for missing alignment_key.")