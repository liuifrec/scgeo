import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _make_local_adata():
    rng = np.random.RandomState(21)
    centers = {
        "s0": np.array([0.0, 0.0]),
        "s1": np.array([4.0, 0.2]),
        "s2": np.array([0.3, 4.0]),
    }
    rows = []
    obs = []
    for state, center in centers.items():
        cloud = rng.normal(scale=[0.45, 0.18], size=(40, 2)) + center
        rows.append(cloud)
        for i in range(cloud.shape[0]):
            obs.append({"state": state, "sample": f"{state}_sample{i % 4}", "batch": f"b{i % 2}"})
    X = np.vstack(rows).astype(np.float32)
    obs = pd.DataFrame(obs, index=[f"c{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=np.zeros((X.shape[0], 1)), obs=obs)

    rot = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    adata.obsm["X_base"] = X
    adata.obsm["X_rot"] = (X @ rot).astype(np.float32)
    adata.obsm["X_translate"] = (X + np.array([10.0, -7.0])).astype(np.float32)
    adata.obsm["X_scaled"] = (3.0 * X).astype(np.float32)
    adata.obsm["X_pad"] = np.column_stack([X, np.zeros((X.shape[0], 2), dtype=np.float32)])
    adata.obsm["X_aniso"] = (X * np.array([3.0, 0.35], dtype=np.float32)).astype(np.float32)
    adata.obsm["X_warp"] = np.column_stack([X[:, 0], X[:, 1] + 0.25 * X[:, 0] ** 2]).astype(np.float32)

    state_distorted = X.copy()
    state_mask = obs["state"].to_numpy() == "s2"
    state_distorted[state_mask] = state_distorted[state_mask] * np.array([4.0, 0.4], dtype=np.float32)
    adata.obsm["X_state_distorted"] = state_distorted

    perm = rng.permutation(X.shape[0])
    adata.obsm["X_shuffle"] = X[perm].astype(np.float32)
    return adata


def _global_metric(out, rep_a, rep_b, k, metric, column="median"):
    df = out["pair_summary"]
    row = df[
        (df["rep_a"] == rep_a)
        & (df["rep_b"] == rep_b)
        & (df["k"] == k)
        & (df["scope"] == "global")
        & (df["metric"] == metric)
    ]
    assert len(row) == 1
    return float(row[column].iloc[0])


def test_local_geometry_invariant_equivalent_transforms():
    import scgeo as sg

    adata = _make_local_adata()
    reps = ["X_base", "X_rot", "X_translate", "X_scaled", "X_pad"]
    out = sg.tl.local_geometry_stability(
        adata,
        reps=reps,
        node_key="state",
        k_values=(10,),
        pair_mode="reference",
        reference_rep="X_base",
        n_boot=0,
        max_exact_cells=200,
        seed=1,
    )

    for rep in reps[1:]:
        assert _global_metric(out, "X_base", rep, 10, "neighbor_overlap") > 0.999
        assert _global_metric(out, "X_base", rep, 10, "neighbor_jaccard") > 0.999
        assert _global_metric(out, "X_base", rep, 10, "global_distortion_median") < 5e-6
        assert _global_metric(out, "X_base", rep, 10, "local_distortion_median") < 5e-6

    ordered = out["ordered_rank_summary"]
    assert (ordered["trustworthiness"].dropna() > 0.999).all()
    assert (ordered["continuity"].dropna() > 0.999).all()


def test_local_geometry_anisotropic_warp_and_shuffle_outlier():
    import scgeo as sg

    adata = _make_local_adata()
    out = sg.tl.local_geometry_stability(
        adata,
        reps=["X_base", "X_aniso", "X_warp", "X_shuffle"],
        node_key="state",
        k_values=(10,),
        n_boot=0,
        max_exact_cells=200,
        seed=2,
    )

    assert _global_metric(out, "X_base", "X_aniso", 10, "global_distortion_median") > 0.2
    assert _global_metric(out, "X_base", "X_warp", 10, "local_distortion_median") > 0.02
    warp_rank = out["ordered_rank_summary"][
        (out["ordered_rank_summary"]["rep_a"] == "X_base")
        & (out["ordered_rank_summary"]["rep_b"] == "X_warp")
    ].iloc[0]
    assert min(warp_rank["trustworthiness"], warp_rank["continuity"]) < 0.999
    assert _global_metric(out, "X_base", "X_shuffle", 10, "neighbor_overlap") < 0.2
    assert _global_metric(out, "X_base", "X_shuffle", 10, "global_distortion_median") > 0.5

    rep_summary = out["representation_summary"].set_index("rep")
    assert bool(rep_summary.loc["X_shuffle", "neighborhood_outlier"])
    assert bool(rep_summary.loc["X_shuffle", "distortion_outlier"])


def test_local_geometry_state_specific_distortion_and_state_graph():
    import scgeo as sg

    adata = _make_local_adata()
    out = sg.tl.local_geometry_stability(
        adata,
        reps=["X_base", "X_rot", "X_state_distorted", "X_shuffle"],
        node_key="state",
        k_values=(10,),
        n_boot=0,
        max_exact_cells=200,
        seed=3,
    )

    state_rows = out["state_pair_summary"]
    pair = state_rows[
        (state_rows["rep_a"] == "X_base")
        & (state_rows["rep_b"] == "X_state_distorted")
        & (state_rows["metric"] == "global_distortion_median")
    ].set_index("state")
    assert pair.loc["s2", "median"] > pair.loc["s0", "median"]
    assert pair.loc["s2", "median"] > pair.loc["s1", "median"]

    graph = out["state_graph_summary"]
    base_rot = graph[(graph["rep_a"] == "X_base") & (graph["rep_b"] == "X_rot")]["spearman_r"].iloc[0]
    base_shuffle = graph[(graph["rep_a"] == "X_base") & (graph["rep_b"] == "X_shuffle")]["spearman_r"].iloc[0]
    assert base_rot > 0.99
    assert base_shuffle < base_rot
    assert out["coverage_summary"]["state_order"] == ["s0", "s1", "s2"]


def test_local_geometry_reference_mode_sample_bootstrap_subset_and_reproducible():
    import scgeo as sg

    adata1 = _make_local_adata()
    adata2 = adata1.copy()
    kwargs = dict(
        reps=["X_base", "X_rot", "X_scaled"],
        node_key="state",
        sample_key="sample",
        k_values=(5, 10),
        pair_mode="reference",
        reference_rep="X_base",
        n_boot=5,
        max_exact_cells=50,
        stratify_key="state",
        store_per_cell=True,
        seed=4,
    )
    out1 = sg.tl.local_geometry_stability(adata1, **kwargs)
    out2 = sg.tl.local_geometry_stability(adata2, **kwargs)

    assert set(out1["pair_summary"]["k"]) == {5, 10}
    assert set(zip(out1["pair_summary"]["rep_a"], out1["pair_summary"]["rep_b"])) == {
        ("X_base", "X_rot"),
        ("X_base", "X_scaled"),
    }
    assert out1["coverage_summary"]["rank_subset_sampled"]
    assert out1["coverage_summary"]["rank_subset_n"] == 50
    assert "per_cell" in out1
    sample_rows = out1["pair_summary"][out1["pair_summary"]["scope"] == "global"]
    assert sample_rows["n_samples"].dropna().eq(12).all()

    compare_cols = ["rep_a", "rep_b", "k", "scope", "state", "metric", "mean", "median", "ci95_low", "ci95_high"]
    pd.testing.assert_frame_equal(out1["pair_summary"][compare_cols], out2["pair_summary"][compare_cols])
    pd.testing.assert_frame_equal(out1["ordered_rank_summary"], out2["ordered_rank_summary"])


def test_local_geometry_missing_rep_invalid_k_and_sparse_obsm():
    import scgeo as sg

    sparse = pytest.importorskip("scipy.sparse")
    adata = _make_local_adata()
    with pytest.raises(KeyError):
        sg.tl.local_geometry_stability(adata, reps=["X_base", "X_missing"], k_values=(5,))
    with pytest.raises(ValueError, match="k < n_obs"):
        sg.tl.local_geometry_stability(adata, reps=["X_base", "X_rot"], k_values=(adata.n_obs,))

    adata_sparse = _make_local_adata()
    adata_sparse.obsm["X_sparse_base"] = sparse.csr_matrix(adata_sparse.obsm["X_base"])
    out = sg.tl.local_geometry_stability(
        adata_sparse,
        reps=["X_sparse_base", "X_rot"],
        k_values=(10,),
        n_boot=0,
        max_exact_cells=200,
        seed=5,
    )
    assert _global_metric(out, "X_sparse_base", "X_rot", 10, "neighbor_overlap") > 0.999


def test_local_geometry_underpowered_state_is_flagged():
    import scgeo as sg

    rng = np.random.RandomState(31)
    X_big = rng.normal(size=(20, 2))
    X_small = rng.normal(loc=4.0, size=(3, 2))
    X = np.vstack([X_big, X_small]).astype(np.float32)
    obs = pd.DataFrame(
        {"state": ["big"] * 20 + ["small"] * 3},
        index=[f"c{i}" for i in range(23)],
    )
    adata = ad.AnnData(X=np.zeros((23, 1)), obs=obs)
    adata.obsm["X_a"] = X
    adata.obsm["X_b"] = X + np.array([1.0, -2.0], dtype=np.float32)

    out = sg.tl.local_geometry_stability(
        adata,
        reps=["X_a", "X_b"],
        node_key="state",
        k_values=(5,),
        n_boot=0,
        max_exact_cells=30,
        seed=6,
    )
    small = out["state_pair_summary"][
        (out["state_pair_summary"]["state"] == "small")
        & (out["state_pair_summary"]["metric"] == "neighbor_overlap")
    ]
    assert small["status"].eq("underpowered_cells").all()
