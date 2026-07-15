import anndata as ad
import numpy as np
import pandas as pd


def _orthogonal_rotation():
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _make_synthetic_adata(*, with_samples=False):
    rng = np.random.RandomState(11)
    specs = [
        ("aligned", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 30),
        ("discordant", np.array([0.0, 4.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), 30),
        ("null", np.array([4.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 30),
        ("distorted", np.array([4.0, 4.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]), 30),
        ("underpowered", np.array([8.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 8),
    ]

    X_rows = []
    V_rows = []
    obs_rows = []
    for node, base, delta, velocity, n_per_condition in specs:
        for condition, offset in [("A", np.zeros(3)), ("B", delta)]:
            X_node = base + offset + rng.normal(scale=0.03, size=(n_per_condition, 3))
            V_node = np.repeat(velocity[None, :], n_per_condition, axis=0)
            start = len(obs_rows)
            X_rows.append(X_node)
            V_rows.append(V_node)
            for i in range(n_per_condition):
                row = {"condition": condition, "state": node}
                if with_samples:
                    sample_idx = min(i // max(1, n_per_condition // 3), 2)
                    row["sample"] = f"{condition}_s{sample_idx}"
                row["cell_id"] = f"{node}_{condition}_{start + i}"
                obs_rows.append(row)

    missing_node = "missing_condition"
    missing_base = np.array([8.0, 4.0, 0.0])
    missing_n = 25
    X_missing = missing_base + rng.normal(scale=0.03, size=(missing_n, 3))
    V_missing = np.repeat(np.array([[1.0, 0.0, 0.0]]), missing_n, axis=0)
    X_rows.append(X_missing)
    V_rows.append(V_missing)
    for i in range(missing_n):
        row = {"condition": "A", "state": missing_node}
        if with_samples:
            sample_idx = min(i // max(1, missing_n // 3), 2)
            row["sample"] = f"A_s{sample_idx}"
        row["cell_id"] = f"{missing_node}_A_{i}"
        obs_rows.append(row)

    X = np.vstack(X_rows).astype(np.float32)
    V = np.vstack(V_rows).astype(np.float32)
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    adata = ad.AnnData(X=np.zeros((X.shape[0], 1)), obs=obs)

    Q = _orthogonal_rotation()
    adata.obsm["X_base"] = X
    adata.obsm["X_rot"] = (X @ Q).astype(np.float32)
    adata.obsm["X_scaled"] = (2.5 * X).astype(np.float32)
    adata.obsm["X_5d"] = np.column_stack([X, np.zeros((X.shape[0], 2), dtype=np.float32)])

    distorted = X.copy()
    mask_distorted_b = (obs["state"].to_numpy() == "distorted") & (obs["condition"].to_numpy() == "B")
    distorted[mask_distorted_b] = distorted[mask_distorted_b] + np.array([4.0, 4.0, 0.0], dtype=np.float32)
    adata.obsm["X_distorted"] = distorted

    adata.obsm["V_base"] = V
    adata.obsm["V_rot"] = (V @ Q).astype(np.float32)
    adata.obsm["V_scaled"] = (2.5 * V).astype(np.float32)
    adata.obsm["V_5d"] = np.column_stack([V, np.zeros((V.shape[0], 2), dtype=np.float32)])
    return adata


def test_representation_stability_invariant_rank_and_velocity_cosine():
    import scgeo as sg

    adata = _make_synthetic_adata()
    reps = ["X_base", "X_rot", "X_scaled", "X_5d"]
    out = sg.tl.representation_stability(
        adata,
        reps=reps,
        node_key="state",
        condition_key="condition",
        group0="A",
        group1="B",
        center="mean",
        n_boot=0,
        velocity_keys={
            "X_base": "V_base",
            "X_rot": "V_rot",
            "X_scaled": "V_scaled",
            "X_5d": "V_5d",
        },
        min_cells=20,
        seed=3,
    )

    per = out["per_rep_state"]
    usable = per[per["usable"] & (per["node"] != "underpowered")]

    rank_wide = usable.pivot(index="node", columns="rep", values="magnitude_rank")
    for rep in reps[1:]:
        np.testing.assert_allclose(rank_wide["X_base"], rank_wide[rep], atol=1e-8)

    mag_wide = usable.pivot(index="node", columns="rep", values="normalized_delta_norm")
    for rep in reps[1:]:
        np.testing.assert_allclose(mag_wide["X_base"], mag_wide[rep], rtol=1e-5, atol=1e-5)

    cosine_wide = usable.pivot(index="node", columns="rep", values="alignment_cosine")
    for rep in reps[1:]:
        np.testing.assert_allclose(cosine_wide.loc[["aligned", "discordant"], "X_base"], cosine_wide.loc[["aligned", "discordant"], rep])
    assert cosine_wide.loc["aligned", "X_base"] > 0.99
    assert cosine_wide.loc["discordant", "X_base"] < -0.99

    rank_corr = out["rank_correlation"]
    assert (rank_corr["spearman_r"].dropna() > 0.999).all()

    labels = out["consensus_state"].set_index("node")["consensus_label"].to_dict()
    assert labels["aligned"] == "stable_aligned"
    assert labels["discordant"] == "stable_discordant"
    assert labels["null"] == "stable_neutral"
    assert labels["underpowered"] == "insufficient_coverage"
    assert labels["missing_condition"] == "insufficient_coverage"


def test_representation_stability_distorted_rep_and_missing_velocity():
    import scgeo as sg

    adata = _make_synthetic_adata()
    reps = ["X_base", "X_rot", "X_scaled", "X_5d", "X_distorted"]
    out = sg.tl.representation_stability(
        adata,
        reps=reps,
        node_key="state",
        condition_key="condition",
        group0="A",
        group1="B",
        center="mean",
        n_boot=0,
        velocity_keys={
            "X_base": "V_base",
            "X_rot": "V_rot",
            "X_scaled": "V_scaled",
            "X_5d": "V_5d",
            "X_distorted": None,
        },
        min_cells=20,
        seed=4,
    )

    per = out["per_rep_state"]
    distorted_row = per[(per["rep"] == "X_distorted") & (per["node"] == "distorted")].iloc[0]
    assert distorted_row["velocity_status"] == "not_requested"
    assert distorted_row["alignment_class"] == "missing"
    assert distorted_row["loo_magnitude_abs_deviation"] > 2.0

    consensus = out["consensus_state"].set_index("node")
    assert consensus.loc["distorted", "consensus_label"] == "representation_unstable"
    assert consensus.loc["distorted", "loo_rep_magnitude_max_relative_deviation"] > 0.5
    assert consensus.loc["underpowered", "consensus_label"] == "insufficient_coverage"
    assert "underpowered_cells" in out["coverage_summary"]["status_counts"]
    assert consensus.loc["missing_condition", "consensus_label"] == "insufficient_coverage"
    assert "missing_condition" in out["coverage_summary"]["status_counts"]

    assert {"rep_a", "rep_b", "spearman_r", "n_states"} <= set(out["rank_correlation"].columns)


def test_representation_stability_sample_bootstrap_and_fixed_seed():
    import scgeo as sg

    adata1 = _make_synthetic_adata(with_samples=True)
    adata2 = adata1.copy()
    kwargs = dict(
        reps=["X_base", "X_rot"],
        node_key="state",
        condition_key="condition",
        group0="A",
        group1="B",
        sample_key="sample",
        center="mean",
        n_boot=6,
        velocity_keys={"X_base": "V_base", "X_rot": "V_rot"},
        min_cells=20,
        seed=5,
    )

    out1 = sg.tl.representation_stability(adata1, **kwargs)
    out2 = sg.tl.representation_stability(adata2, **kwargs)

    assert out1["params"]["resolved_bootstrap_unit"] == "sample"
    row = out1["per_rep_state"][
        (out1["per_rep_state"]["rep"] == "X_base")
        & (out1["per_rep_state"]["node"] == "aligned")
    ].iloc[0]
    assert row["n_samples0"] == 3
    assert row["n_samples1"] == 3
    assert np.isfinite(row["magnitude_ci95_low"])
    assert np.isfinite(row["magnitude_ci95_high"])

    cols = [
        "rep",
        "node",
        "status",
        "normalized_delta_norm",
        "magnitude_ci95_low",
        "magnitude_ci95_high",
        "directional_resultant_length",
        "alignment_cosine",
        "alignment_class",
        "magnitude_rank",
    ]
    pd.testing.assert_frame_equal(out1["per_rep_state"][cols], out2["per_rep_state"][cols])
    pd.testing.assert_frame_equal(out1["consensus_state"], out2["consensus_state"])
