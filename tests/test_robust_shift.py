import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _make_adata(X, condition, *, sample=None, cell_type=None, rep="X_pca", use_X=False):
    obs = {"condition": condition}
    if sample is not None:
        obs["sample"] = sample
    if cell_type is not None:
        obs["cell_type"] = cell_type
    obs = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(condition))])
    if use_X:
        adata = ad.AnnData(X=X, obs=obs)
    else:
        adata = ad.AnnData(X=np.zeros((len(condition), 1)), obs=obs)
        adata.obsm[rep] = X
    return adata


def test_shift_signature_is_unchanged():
    import scgeo as sg

    assert str(inspect.signature(sg.tl.shift)) == (
        "(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', "
        "group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, "
        "sample_key: 'Optional[str]' = None, store_key: 'str' = 'scgeo') -> 'None'"
    )


def test_robust_shift_gaussian_no_outlier():
    import scgeo as sg

    rng = np.random.RandomState(0)
    expected = np.array([1.0, -0.5, 0.25])
    X0 = rng.normal(loc=0.0, scale=0.2, size=(120, 3))
    X1 = rng.normal(loc=expected, scale=0.2, size=(120, 3))
    X = np.vstack([X0, X1]).astype(np.float32)
    adata = _make_adata(X, ["A"] * 120 + ["B"] * 120)

    out = sg.tl.robust_shift(
        adata,
        group0="A",
        group1="B",
        center="geometric_median",
        n_boot=12,
        seed=1,
    )

    global_out = out["global"]
    np.testing.assert_allclose(global_out["delta"], expected, atol=0.12)
    assert global_out["delta_norm"] > 1.0
    assert np.isfinite(global_out["normalized_delta_norm"])
    assert np.all(np.isfinite(global_out["bootstrap_magnitude_ci95"]))
    assert global_out["bootstrap_directional_resultant_length"] > 0.95
    assert global_out["outlier_sensitivity"]["delta_difference_norm"] < 0.08
    assert adata.uns["scgeo"]["robust_shift"] is out


def test_robust_shift_extreme_outliers_reduce_mean_sensitivity():
    import scgeo as sg

    rng = np.random.RandomState(1)
    X0 = rng.normal(loc=0.0, scale=0.15, size=(100, 2))
    X1 = rng.normal(loc=np.array([1.0, 0.0]), scale=0.15, size=(100, 2))
    X1[:8] = np.array([50.0, 0.0])
    X = np.vstack([X0, X1]).astype(np.float32)
    adata = _make_adata(X, ["A"] * 100 + ["B"] * 100)

    out = sg.tl.robust_shift(
        adata,
        group0="A",
        group1="B",
        center="geometric_median",
        n_boot=12,
        seed=2,
    )["global"]

    sensitivity = out["outlier_sensitivity"]
    assert out["delta"][0] == pytest.approx(1.0, abs=0.25)
    assert sensitivity["mean_delta"][0] > 4.0
    assert out["delta_norm"] < 0.4 * sensitivity["mean_delta_norm"]
    assert sensitivity["delta_difference_norm"] > 3.0


def test_robust_shift_unequal_cell_counts_and_by_levels():
    import scgeo as sg

    rng = np.random.RandomState(2)
    n0, n1 = 25, 140
    X0 = rng.normal(loc=0.0, scale=0.5, size=(n0, 4))
    X1 = rng.normal(loc=0.6, scale=0.5, size=(n1, 4))
    X = np.vstack([X0, X1]).astype(np.float32)
    cell_type = (["T"] * 10 + ["B"] * 15) + (["T"] * 70 + ["B"] * 70)
    adata = _make_adata(X, ["A"] * n0 + ["B"] * n1, cell_type=cell_type)

    out = sg.tl.robust_shift(
        adata,
        group0="A",
        group1="B",
        by="cell_type",
        center="trimmed_mean",
        trim_fraction=0.1,
        n_boot=20,
        seed=3,
    )

    assert out["global"]["n_cells0"] == n0
    assert out["global"]["n_cells1"] == n1
    assert out["global"]["center0"].shape == (4,)
    assert np.isfinite(out["global"]["delta_norm"])
    assert set(out["by"]) == {"T", "B"}
    assert out["by"]["T"]["n_cells0"] == 10
    assert out["by"]["B"]["n_cells1"] == 70


def test_robust_shift_sample_level_bootstrap_uses_sample_centers():
    import scgeo as sg

    rows = []
    conditions = []
    samples = []
    sample_specs = [
        ("A", "a0", 10, [0.0, 0.0]),
        ("A", "a1", 50, [10.0, 0.0]),
        ("A", "a2", 10, [0.0, 0.0]),
        ("B", "b0", 10, [1.0, 0.0]),
        ("B", "b1", 50, [11.0, 0.0]),
        ("B", "b2", 10, [1.0, 0.0]),
    ]
    for condition, sample, n_cells, value in sample_specs:
        rows.append(np.repeat(np.asarray([value], dtype=float), n_cells, axis=0))
        conditions.extend([condition] * n_cells)
        samples.extend([sample] * n_cells)
    X = np.vstack(rows).astype(np.float32)
    adata = _make_adata(X, conditions, sample=samples)

    out = sg.tl.robust_shift(
        adata,
        group0="A",
        group1="B",
        sample_key="sample",
        center="mean",
        n_boot=20,
        seed=4,
    )["global"]

    assert out["estimator_params"]["bootstrap_unit"] == "sample"
    assert out["n_samples0"] == 3
    assert out["n_samples1"] == 3
    assert out["n_cells0"] == 70
    assert out["n_cells1"] == 70
    assert out["center0"][0] == pytest.approx(10.0 / 3.0)
    assert out["center1"][0] == pytest.approx(13.0 / 3.0)
    np.testing.assert_allclose(out["delta"], np.array([1.0, 0.0]), atol=1e-6)
    assert np.all(np.isfinite(out["bootstrap_magnitude_ci95"]))


def test_robust_shift_rejects_cell_bootstrap_with_samples():
    import scgeo as sg

    X = np.zeros((8, 2), dtype=np.float32)
    adata = _make_adata(
        X,
        ["A"] * 4 + ["B"] * 4,
        sample=["a0", "a1", "a0", "a1", "b0", "b1", "b0", "b1"],
    )

    with pytest.raises(ValueError, match="not allowed"):
        sg.tl.robust_shift(
            adata,
            group0="A",
            group1="B",
            sample_key="sample",
            bootstrap_unit="cell",
        )


def test_robust_shift_zero_shift_null_case():
    import scgeo as sg

    rng = np.random.RandomState(5)
    base = rng.normal(size=(50, 4)).astype(np.float32)
    X = np.vstack([base, base])
    adata = _make_adata(X, ["A"] * 50 + ["B"] * 50)

    out = sg.tl.robust_shift(
        adata,
        group0="A",
        group1="B",
        center="median",
        n_boot=30,
        seed=6,
    )["global"]

    assert out["delta_norm"] == pytest.approx(0.0)
    assert out["normalized_delta_norm"] == pytest.approx(0.0)
    assert np.isnan(out["direction_stability"])


def test_robust_shift_reproducible_with_fixed_seed():
    import scgeo as sg

    rng = np.random.RandomState(6)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=1.0, size=(60, 3)),
            rng.normal(loc=0.4, scale=1.0, size=(60, 3)),
        ]
    ).astype(np.float32)
    adata1 = _make_adata(X, ["A"] * 60 + ["B"] * 60)
    adata2 = adata1.copy()

    out1 = sg.tl.robust_shift(
        adata1,
        group0="A",
        group1="B",
        center="median",
        n_boot=35,
        seed=7,
    )["global"]
    out2 = sg.tl.robust_shift(
        adata2,
        group0="A",
        group1="B",
        center="median",
        n_boot=35,
        seed=7,
    )["global"]

    np.testing.assert_allclose(out1["delta"], out2["delta"])
    np.testing.assert_allclose(out1["bootstrap_magnitude_ci95"], out2["bootstrap_magnitude_ci95"])
    assert out1["bootstrap_directional_resultant_length"] == pytest.approx(
        out2["bootstrap_directional_resultant_length"]
    )
    assert out1["direction_stability"] == pytest.approx(out2["direction_stability"])


def test_robust_shift_dense_sparse_obsm_and_sparse_x_match():
    import scgeo as sg

    sparse = pytest.importorskip("scipy.sparse")

    rng = np.random.RandomState(8)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.2, size=(20, 3)),
            rng.normal(loc=1.0, scale=0.2, size=(20, 3)),
        ]
    ).astype(np.float32)
    condition = ["A"] * 20 + ["B"] * 20

    dense = _make_adata(X, condition)
    sparse_obsm = _make_adata(sparse.csr_matrix(X), condition)
    sparse_x = _make_adata(sparse.csr_matrix(X), condition, use_X=True)

    kwargs = dict(
        group0="A",
        group1="B",
        center="median",
        n_boot=0,
        normalize_by="none",
    )
    out_dense = sg.tl.robust_shift(dense, **kwargs)["global"]
    out_sparse_obsm = sg.tl.robust_shift(sparse_obsm, **kwargs)["global"]
    out_sparse_x = sg.tl.robust_shift(sparse_x, rep="X", **kwargs)["global"]

    np.testing.assert_allclose(out_dense["delta"], out_sparse_obsm["delta"])
    np.testing.assert_allclose(out_dense["delta"], out_sparse_x["delta"])
