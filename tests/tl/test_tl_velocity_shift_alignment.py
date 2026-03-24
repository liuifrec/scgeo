import numpy as np
import pandas as pd
from anndata import AnnData

from scgeo.tl._velocity_shift_alignment import velocity_shift_alignment


def make_test_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "node": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "timepoint": ["D8", "D8", "D21", "D21", "D8", "D8", "D21", "D21"],
        },
        index=[f"cell{i}" for i in range(8)],
    )

    X = np.zeros((8, 3), dtype=float)
    adata = AnnData(X=X, obs=obs)

    # UMAP coordinates
    # Node A: shift right, velocity right -> aligned
    # Node B: shift right, velocity left -> discordant
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0],  # A D8
            [0.0, 1.0],  # A D8
            [1.0, 0.0],  # A D21
            [1.0, 1.0],  # A D21
            [0.0, 3.0],  # B D8
            [0.0, 4.0],  # B D8
            [1.0, 3.0],  # B D21
            [1.0, 4.0],  # B D21
        ],
        dtype=float,
    )

    # Per-cell velocity in UMAP space
    adata.obsm["velocity_umap"] = np.array(
        [
            [1.0, 0.0],  # A
            [1.0, 0.0],  # A
            [1.0, 0.0],  # A
            [1.0, 0.0],  # A
            [-1.0, 0.0],  # B
            [-1.0, 0.0],  # B
            [-1.0, 0.0],  # B
            [-1.0, 0.0],  # B
        ],
        dtype=float,
    )

    return adata


def test_velocity_shift_alignment_basic():
    adata = make_test_adata()

    out = velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D8",
        group1="D21",
        basis="umap",
        min_cells=2,
        key_added="test_vsa",
        propagate_to_obs=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert set(["node", "dx", "dy", "shift_norm", "vx", "vy", "vel_norm", "alignment_cosine", "alignment_class", "usable"]).issubset(out.columns)
    assert "test_vsa" in adata.uns

    out = out.set_index("node")

    assert np.isclose(out.loc["A", "dx"], 1.0)
    assert np.isclose(out.loc["A", "dy"], 0.0)
    assert np.isclose(out.loc["A", "alignment_cosine"], 1.0)
    assert out.loc["A", "alignment_class"] == "aligned"
    assert bool(out.loc["A", "usable"]) is True

    assert np.isclose(out.loc["B", "dx"], 1.0)
    assert np.isclose(out.loc["B", "dy"], 0.0)
    assert np.isclose(out.loc["B", "alignment_cosine"], -1.0)
    assert out.loc["B", "alignment_class"] == "discordant"
    assert bool(out.loc["B", "usable"]) is True


def test_velocity_shift_alignment_propagate_to_obs():
    adata = make_test_adata()

    velocity_shift_alignment(
        adata,
        node_key="node",
        condition_key="timepoint",
        group0="D8",
        group1="D21",
        basis="umap",
        min_cells=2,
        key_added="test_vsa",
        propagate_to_obs=True,
    )

    expected_cols = [
        "test_vsa_cosine",
        "test_vsa_class",
        "test_vsa_usable",
        "test_vsa_shift_norm",
        "test_vsa_vel_norm",
    ]
    for col in expected_cols:
        assert col in adata.obs.columns

    # Node A cells should all be aligned
    a_mask = adata.obs["node"] == "A"
    assert set(adata.obs.loc[a_mask, "test_vsa_class"].astype(str)) == {"aligned"}

    # Node B cells should all be discordant
    b_mask = adata.obs["node"] == "B"
    assert set(adata.obs.loc[b_mask, "test_vsa_class"].astype(str)) == {"discordant"}


def test_velocity_shift_alignment_missing_basis_raises():
    adata = make_test_adata()
    del adata.obsm["X_umap"]

    try:
        velocity_shift_alignment(
            adata,
            node_key="node",
            condition_key="timepoint",
            group0="D8",
            group1="D21",
            basis="umap",
        )
    except KeyError as e:
        assert "Embedding 'X_umap' not found" in str(e)
    else:
        raise AssertionError("Expected KeyError for missing embedding.")