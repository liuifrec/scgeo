"""Scientific contracts for descriptive attribution of stored ScGeo shifts.

The fixtures distinguish the measured displacement from ordinary mean changes,
biological samples from cells, and local response from population composition.
They do not require a causal interpretation or inferential p-values.
"""

from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scgeo as sg

GENES = ["g_up", "g_down", "g_side", "g_background", "g_unrepresented"]
LOADINGS = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.2, 0.1, 1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _make_affine(*, center="mean", outliers=False, zero=False, sample_key="sample"):
    """Four independent samples per arm, with unequal cell counts and known shift."""
    rng = np.random.RandomState(618)
    delta = np.array([1.4, -0.45, 0.35, 0.0, 4.0])
    if zero:
        delta[:4] = 0.0
    rows = []
    metadata = []
    bases = []
    for j, n in enumerate([24, 32, 40, 48]):
        donor_offset = rng.normal(scale=0.12, size=5)
        cloud = rng.normal(scale=0.16, size=(n, 5))
        cloud -= cloud.mean(axis=0)
        bases.append(np.array([4.0, 3.0, 2.0, 3.0, 0.8]) + donor_offset + cloud)
    for condition, prefix in [("control", "C"), ("treated", "T")]:
        for j, base in enumerate(bases):
            X = base.copy()
            if condition == "treated":
                X += delta
                if outliers and j == 0:
                    X[-3:, :2] += np.array([150.0, 50.0])
            rows.append(X)
            metadata.extend(
                {"condition": condition, "sample": f"{prefix}{j}", "cell_type": "S"}
                for _ in range(len(X))
            )
    obs = pd.DataFrame(metadata, index=[f"cell_{i}" for i in range(len(metadata))])
    a = ad.AnnData(np.vstack(rows), obs=obs, var=pd.DataFrame(index=GENES))
    a.varm["PCs"] = LOADINGS.copy()
    a.obsm["X_pca"] = a.X @ LOADINGS + np.array([12.0, -7.0, 3.0])
    _store_shift(a, center=center, sample_key=sample_key)
    return a


def _store_shift(a, *, center="mean", sample_key="sample"):
    return sg.tl.robust_shift(
        a,
        rep="X_pca",
        condition_key="condition",
        group0="control",
        group1="treated",
        by="cell_type",
        sample_key=sample_key,
        center=center,
        n_boot=0,
        normalize_by="none",
    )


def _explain(a, **kwargs):
    return sg.tl.explain_shift(
        a, states=["S"], n_boot=kwargs.pop("n_boot", 0), **kwargs
    )


def _genes(result, state="S"):
    return (
        result["genes"]
        .loc[result["genes"]["state"] == state]
        .set_index("gene")
        .sort_index()
    )


def _summary(result, state="S"):
    return result["summary"].set_index("state").loc[state]


def _state_result(result, state="S"):
    return next(
        value for value in result["state_results"].values() if value["state"] == state
    )


def _make_local_example(*, composition_only=False):
    rows = []
    records = []
    for condition, prefix in [("control", "C"), ("treated", "T")]:
        for j in range(3):
            for local in ["early", "late"]:
                if composition_only:
                    n = 36 if (condition == "control") == (local == "early") else 12
                    response = 0.0
                else:
                    n = 24
                    response = (
                        (1.0 if local == "early" else -1.0)
                        if condition == "treated"
                        else 0.0
                    )
                jitter = np.tile(np.array([[-0.1, -0.05], [0.1, 0.05]]), (n // 2, 1))
                rows.append(
                    jitter + np.array([response, -2.0 if local == "early" else 2.0])
                )
                records.extend(
                    {
                        "condition": condition,
                        "sample": f"{prefix}{j}",
                        "cell_type": "S",
                        "local_state": local,
                    }
                    for _ in range(n)
                )
    a = ad.AnnData(
        np.vstack(rows),
        obs=pd.DataFrame(records, index=[f"local_{i}" for i in range(len(records))]),
        var=pd.DataFrame(index=["response", "identity"]),
    )
    a.varm["PCs"] = np.eye(2)
    a.obsm["X_pca"] = a.X.copy()
    _store_shift(a)
    return a


def test_mean_attribution_has_known_signed_contributions_and_reconstructs_shift():
    a = _make_affine()
    source_before = deepcopy(a.uns["scgeo"]["robust_shift"])
    result = _explain(a)
    genes = _genes(result)
    delta_expression = np.array([1.4, -0.45, 0.35, 0.0, 4.0])
    delta_rep = delta_expression @ LOADINGS
    direction = delta_rep / np.linalg.norm(delta_rep)
    expected = delta_expression * (LOADINGS @ direction)

    np.testing.assert_allclose(
        genes.loc[GENES, "expression_delta"], delta_expression, atol=1e-10
    )
    np.testing.assert_allclose(genes.loc[GENES, "contribution"], expected, atol=1e-7)
    assert genes.loc["g_down", "contribution"] < 0
    assert genes.loc["g_up", "contribution"] > 0
    assert genes.loc["g_unrepresented", "expression_delta"] == pytest.approx(4.0)
    assert genes.loc["g_unrepresented", "contribution"] == pytest.approx(0.0)
    assert not genes.loc["g_unrepresented", "in_representation"]
    summary = _summary(result)
    assert summary["status"] == "ok"
    assert summary["contribution_sum"] == pytest.approx(
        np.linalg.norm(delta_rep), abs=1e-6
    )
    assert summary["reconstruction_error"] < 1e-6
    assert result is a.uns["scgeo"]["explain_shift"]
    np.testing.assert_array_equal(
        a.uns["scgeo"]["robust_shift"]["by"]["S"]["delta"],
        source_before["by"]["S"]["delta"],
    )
    assert a.uns["scgeo"]["robust_shift"]["params"] == source_before["params"]


def test_geometric_median_uses_matched_hierarchical_weights_not_pseudobulk_means():
    a = _make_affine(center="geometric_median", outliers=True)
    result = _explain(a)
    state = _state_result(result)
    genes = _genes(result).loc[GENES]
    source = a.uns["scgeo"]["robust_shift"]["by"]["S"]

    for arm in ["0", "1"]:
        positions = np.asarray(state[f"obs_positions{arm}"], dtype=int)
        weights = np.asarray(state[f"center_weights{arm}"], dtype=float)
        assert weights.shape == (len(positions),)
        assert np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            weights @ a.obsm["X_pca"][positions],
            source[f"center{arm}"],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            weights @ a.X[positions],
            genes[f"expression_center{arm}"],
            atol=1e-7,
        )
    assert genes["contribution"].sum() == pytest.approx(source["delta_norm"], abs=2e-6)
    # A few outliers have a large effect on ordinary sample means, unlike the
    # matched robust centers the explanatory routine is required to consume.
    raw = pd.DataFrame(a.X, columns=GENES).assign(
        condition=a.obs["condition"].to_numpy(),
        sample=a.obs["sample"].to_numpy(),
    )
    sample_means = raw.groupby(["condition", "sample"])[GENES].mean()
    naive = sample_means.loc["treated"].mean() - sample_means.loc["control"].mean()
    assert abs(naive["g_up"] - genes.loc["g_up", "expression_delta"]) > 2.0


def test_center_weights_reproduce_the_exact_legacy_quantized_sample_centers():
    a = _make_affine(center="geometric_median", outliers=True)
    result = _explain(a)
    state = _state_result(result)
    source = a.uns["scgeo"]["robust_shift"]["by"]["S"]
    quantized = a.obsm["X_pca"].astype(np.float32).astype(np.float64)
    for arm in ["0", "1"]:
        positions = np.asarray(state[f"obs_positions{arm}"], dtype=int)
        weights = np.asarray(state[f"center_weights{arm}"], dtype=float)
        sample_labels = a.obs["sample"].to_numpy()[positions]
        for sample, center in zip(
            source[f"sample_ids{arm}"], source[f"sample_centers{arm}"]
        ):
            selected = sample_labels == sample
            within = weights[selected] / weights[selected].sum()
            np.testing.assert_allclose(
                within @ quantized[positions[selected]], center, rtol=0, atol=5e-12
            )
        np.testing.assert_array_equal(
            np.asarray(state[f"center{arm}"], dtype=np.float32), source[f"center{arm}"]
        )


@pytest.mark.parametrize("center", ["mean", "geometric_median"])
def test_affine_rotation_preserves_gene_attribution(center):
    a = _make_affine(center=center)
    first = _explain(a)
    theta = 0.63
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    rotated = a.copy()
    rotated.varm["PCs"] = LOADINGS @ rotation
    rotated.obsm["X_pca"] = a.obsm["X_pca"] @ rotation + np.array([5.0, 2.0, -11.0])
    _store_shift(rotated, center=center)
    second = _explain(rotated)
    np.testing.assert_allclose(
        _genes(first)[["axis_loading", "contribution"]],
        _genes(second)[["axis_loading", "contribution"]],
        atol=2e-5,
    )
    assert _summary(first)["delta_norm"] == pytest.approx(
        _summary(second)["delta_norm"], abs=2e-5
    )


def test_feature_scaling_and_affine_intercept_are_accounted_for():
    a = _make_affine()
    scale = np.array([3.0, 0.5, 2.0, 1.5, 1.0])
    a.obsm["X_pca"] = (a.X / scale) @ LOADINGS + np.array([-17.0, 40.0, 0.75])
    _store_shift(a)
    result = _explain(a, feature_scale=scale)
    state = _state_result(result)
    expected_loadings = (LOADINGS @ state["direction"]) / scale
    np.testing.assert_allclose(
        _genes(result).loc[GENES, "axis_loading"], expected_loadings, atol=1e-7
    )
    assert _genes(result)["contribution"].sum() == pytest.approx(
        _summary(result)["delta_norm"], abs=1e-6
    )
    with pytest.raises(ValueError, match="(?i)affine|mapping|reconstruct"):
        _explain(a)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_explicit_expression_layer_matches_dense_X(as_sparse):
    a = _make_affine(center="geometric_median", outliers=True)
    expected = _explain(a)
    other = a.copy()
    other.layers["lognorm"] = sparse.csr_matrix(a.X) if as_sparse else a.X.copy()
    other.X = np.zeros_like(a.X)
    actual = _explain(other, layer="lognorm")
    pd.testing.assert_frame_equal(
        _genes(expected), _genes(actual), check_exact=False, atol=1e-7, rtol=1e-7
    )


def test_sparse_expression_and_representation_match_dense_results():
    a = _make_affine()
    expected = _explain(a, n_boot=32, seed=14)
    other = a.copy()
    other.X = sparse.csr_matrix(a.X)
    other.obsm["X_pca"] = sparse.csr_matrix(a.obsm["X_pca"])
    actual = _explain(other, n_boot=32, seed=14)
    pd.testing.assert_frame_equal(
        _genes(expected), _genes(actual), check_exact=False, atol=1e-7, rtol=1e-7
    )


def test_cell_duplication_within_sample_does_not_increase_bootstrap_evidence():
    a = _make_affine(center="geometric_median", outliers=True)
    expected = _explain(a, n_boot=64, seed=17)
    selected = np.flatnonzero(a.obs["sample"].to_numpy() == "C0")
    other = a[np.concatenate([np.arange(a.n_obs), np.tile(selected, 6)])].copy()
    other.obs_names = [f"dup_{i}" for i in range(other.n_obs)]
    _store_shift(other, center="geometric_median")
    actual = _explain(other, n_boot=64, seed=17)
    columns = [
        "expression_delta",
        "contribution",
        "ci_low",
        "ci_high",
        "sign_stability",
    ]
    np.testing.assert_allclose(
        _genes(expected)[columns], _genes(actual)[columns], atol=3e-5, equal_nan=True
    )
    assert _summary(actual)["n_samples0"] == 4
    assert _summary(actual)["n_samples1"] == 4
    assert (
        actual["samples"].loc[actual["samples"]["sample"] == "C0", "n_cells"].iloc[0]
        == len(selected) * 7
    )


@pytest.mark.parametrize("mutation", ["coordinates", "counts", "sample_ids"])
def test_stale_stored_shift_is_rejected_even_with_valid_affine_mapping(mutation):
    a = _make_affine()
    if mutation == "coordinates":
        treated = a.obs["condition"].to_numpy() == "treated"
        a.X[treated, 0] += 0.2
        a.obsm["X_pca"] = a.X @ LOADINGS + np.array([12.0, -7.0, 3.0])
    elif mutation == "counts":
        a = a[1:].copy()
    else:
        a.obs.loc[a.obs["sample"] == "C0", "sample"] = "renamed"
    with pytest.raises(ValueError, match="(?i)stale|stored|mismatch|recompute|changed"):
        _explain(a)
    assert "explain_shift" not in a.uns["scgeo"]


@pytest.mark.parametrize("intercept", [0.0, 1e9])
def test_nonlinear_mapping_is_rejected_even_with_large_translation(intercept):
    a = _make_affine()
    a.obsm["X_pca"] = a.X @ LOADINGS + intercept
    a.obsm["X_pca"][:, 0] += 0.15 * np.square(a.X[:, 0])
    _store_shift(a)
    with pytest.raises(ValueError, match="(?i)affine|mapping|reconstruct"):
        _explain(a)


def test_large_intercept_does_not_hide_stale_biological_sample_centers():
    a = _make_affine()
    # Every coordinate is exactly representable even after the legacy
    # float32 conversion. This isolates stale-center validation from loss of
    # precision in the representation itself.
    a.X = np.round(a.X * 10.0) * 128.0
    integer_loadings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )
    a.varm["PCs"] = integer_loadings
    a.obsm["X_pca"] = a.X @ integer_loadings + 1e9
    _store_shift(a)
    # These opposite changes preserve the equal-sample mean contrast, while
    # changing biological heterogeneity and therefore the bootstrap evidence.
    a.X[a.obs["sample"].to_numpy() == "C0", 0] += 128.0
    a.X[a.obs["sample"].to_numpy() == "C1", 0] -= 128.0
    a.obsm["X_pca"] = a.X @ integer_loadings + 1e9
    with pytest.raises(ValueError, match="(?i)stale|stored|sample centers|changed"):
        _explain(a)


def test_loss_of_geometry_from_legacy_precision_is_rejected_before_attribution():
    a = _make_affine()
    a.obsm["X_pca"] = a.X @ LOADINGS + 1e9
    _store_shift(a)
    # The stored estimator saw float32 coordinates, which have lost the small
    # expression contrast entirely. It is invalid to explain a float64-only
    # direction that the source estimator never measured.
    assert a.uns["scgeo"]["robust_shift"]["by"]["S"]["delta_norm"] == 0
    with pytest.raises(ValueError, match="(?i)affine|mapping|precision"):
        _explain(a)


def test_failed_call_preserves_previous_successful_result():
    a = _make_affine()
    previous = _explain(a)
    a.varm["PCs"][0, 0] *= 3.0
    with pytest.raises(ValueError):
        _explain(a)
    assert a.uns["scgeo"]["explain_shift"] is previous


def test_overflowing_effective_loadings_are_rejected_atomically():
    a = _make_affine()
    previous = _explain(a)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="(?i)finite|overflow|scal|mapping"),
    ):
        _explain(a, feature_scale=np.full(a.n_vars, 1e-320))
    assert a.uns["scgeo"]["explain_shift"] is previous


@pytest.mark.parametrize("center", ["median", "trimmed_mean"])
def test_unsupported_coordinatewise_centers_are_not_silently_reinterpreted(center):
    a = _make_affine(center=center)
    with pytest.raises(ValueError, match="(?i)center|geometric_median|supported"):
        _explain(a)


def test_no_axis_is_invented_for_zero_displacement():
    a = _make_affine(zero=True)
    result = _explain(a, n_boot=16)
    assert _summary(result)["status"] == "zero_shift"
    genes = _genes(result)
    np.testing.assert_allclose(genes["contribution"], 0.0, atol=1e-12)
    assert genes["axis_loading"].isna().all()
    assert genes[["ci_low", "ci_high", "sign_stability"]].isna().all().all()
    assert np.isnan(_state_result(result)["direction"]).all()


def test_missing_condition_within_state_returns_coverage_status():
    a = _make_affine()
    a.obs.loc[a.obs["sample"] == "C0", "cell_type"] = "control_only"
    _store_shift(a)
    result = sg.tl.explain_shift(a, states=["control_only"], n_boot=0)
    row = _summary(result, "control_only")
    assert row["status"] == "insufficient_cells"
    assert row["n_samples1"] == 0
    genes = result["genes"]
    if len(genes):
        assert not (genes["contribution"].fillna(0).abs() > 0).any()


def test_insufficient_samples_keeps_description_without_cell_based_intervals():
    a = _make_affine()
    a = a[a.obs["sample"].isin(["C0", "T0"])].copy()
    _store_shift(a)
    result = _explain(a, n_boot=64)
    assert _summary(result)["status"] == "insufficient_samples"
    assert _genes(result)["contribution"].abs().sum() > 0
    assert _genes(result)[["ci_low", "ci_high", "sign_stability"]].isna().all().all()


def test_absent_sample_key_does_not_fall_back_to_cell_bootstrap():
    a = _make_affine(sample_key=None)
    result = _explain(a, n_boot=64)
    assert _summary(result)["status"] == "insufficient_samples"
    assert _genes(result)[["ci_low", "ci_high", "sign_stability"]].isna().all().all()


def test_paired_bootstrap_pairs_by_sample_id_and_requires_explicit_design():
    a = _make_affine()
    a.obs["sample"] = a.obs["sample"].str[1:]
    controls = np.flatnonzero(a.obs["condition"].to_numpy() == "control")
    treated_reversed = np.concatenate(
        [
            np.flatnonzero(
                (a.obs["condition"] == "treated") & (a.obs["sample"] == str(j))
            )
            for j in range(3, -1, -1)
        ]
    )
    a = a[np.concatenate([controls, treated_reversed])].copy()
    _store_shift(a)
    with pytest.raises(ValueError, match="(?i)paired|shared|overlap"):
        _explain(a, n_boot=64)
    result = _explain(a, n_boot=64, paired=True, seed=21)
    genes = _genes(result)
    # Every real pair has the same known mean response. Independent resampling
    # of the two arms would introduce donor-offset variation into these limits.
    np.testing.assert_allclose(genes["ci_low"], genes["contribution"], atol=1e-6)
    np.testing.assert_allclose(genes["ci_high"], genes["contribution"], atol=1e-6)


def test_opposing_local_responses_survive_zero_global_shift():
    a = _make_local_example()
    result = _explain(a, local_key="local_state")
    assert _summary(result)["status"] == "zero_shift"
    local = result["local_summary"].set_index("local_state")
    assert set(local.index) == {"early", "late"}
    np.testing.assert_allclose(local["delta_norm"], 1.0, atol=1e-6)
    assert set(local["status"]) == {"ok"}
    response = (
        result["local_genes"].query("gene == 'response'").set_index("local_state")
    )
    assert response.loc["early", "expression_delta"] == pytest.approx(1.0)
    assert response.loc["late", "expression_delta"] == pytest.approx(-1.0)


def test_composition_only_shift_has_zero_local_responses_and_visible_fractions():
    a = _make_local_example(composition_only=True)
    result = _explain(a, local_key="local_state")
    assert _summary(result)["delta_norm"] == pytest.approx(2.0)
    assert set(result["local_summary"]["status"]) == {"zero_shift"}
    np.testing.assert_allclose(result["local_summary"]["delta_norm"], 0.0, atol=1e-8)
    fractions = result["composition"].query("local_state == 'early'")
    assert np.allclose(
        fractions.loc[fractions["condition"] == "control", "fraction"], 0.75
    )
    assert np.allclose(
        fractions.loc[fractions["condition"] == "treated", "fraction"], 0.25
    )


def test_broad_representation_evidence_does_not_certify_local_contrasts():
    a = _make_local_example(composition_only=True)
    # This fixture supplies already assessed parent-state evidence. No evidence
    # about the independently specified local groups accompanies that result.
    a.uns["scgeo"]["representation_evidence"] = {
        "params": {
            "condition_key": "condition",
            "sample_key": "sample",
            "center": "mean",
            "group0": "control",
            "group1": "treated",
            "node_key": "cell_type",
            "reps": ["X_pca"],
        },
        "consensus_state": pd.DataFrame(
            {"node": ["S"], "consensus_label": ["stable_effect"]}
        ),
    }
    result = _explain(
        a, local_key="local_state", representation_key="representation_evidence"
    )
    assert _summary(result)["representation_status"] == "stable_effect"
    assert set(result["local_summary"]["representation_status"]) == {"not_assessed"}
    assert set(result["local_summary"]["parent_representation_status"]) == {
        "stable_effect"
    }


def test_programs_retain_signed_contributions_and_explicit_overlapping_membership():
    a = _make_affine()
    result = _explain(
        a, modules={"response": ["g_up", "g_down"], "overlap": ["g_up", "g_side"]}
    )
    genes = _genes(result)
    programs = result["programs"].set_index("program")
    for name, members in {
        "response": ["g_up", "g_down"],
        "overlap": ["g_up", "g_side"],
    }.items():
        assert programs.loc[name, "contribution"] == pytest.approx(
            genes.loc[members, "contribution"].sum()
        )
        assert programs.loc[name, "expression_delta"] == pytest.approx(
            genes.loc[members, "expression_delta"].mean()
        )
    membership = result["module_membership"]
    assert (membership["gene"] == "g_up").sum() == 2
    np.testing.assert_allclose(membership.groupby("program")["weight"].sum(), 1.0)


@pytest.mark.parametrize("members", [["unknown"], ["g_up", "g_up"]])
def test_unknown_and_duplicate_program_members_are_rejected(members):
    a = _make_affine()
    with pytest.raises(ValueError, match="(?i)gene|unknown|duplicate|member"):
        _explain(a, modules={"bad": members})


def test_signed_regulons_report_supplied_target_response_with_provenance():
    a = _make_affine()
    regulons = {
        "forward": {"g_up": 1.0, "g_down": -1.0},
        "reverse": {"g_up": -1.0, "g_down": 1.0},
    }
    with pytest.raises(ValueError, match="regulator_source"):
        _explain(a, regulons=regulons)
    result = _explain(
        a, regulons=regulons, regulator_source="synthetic signed regulatory prior"
    )
    regulators = result["regulators"].set_index("regulator")
    assert regulators.loc["forward", "signed_target_delta"] == pytest.approx(
        (1.4 + 0.45) / 2
    )
    assert regulators.loc["reverse", "signed_target_delta"] == pytest.approx(
        -(1.4 + 0.45) / 2
    )
    assert regulators.loc["forward", "target_contribution_sum"] == pytest.approx(
        regulators.loc["reverse", "target_contribution_sum"],
    )
    assert set(regulators["source"]) == {"synthetic signed regulatory prior"}
    assert set(regulators["evidence_type"]) == {"supplied_signed_target_response"}
    membership = result["regulon_membership"]
    np.testing.assert_allclose(
        membership.groupby("regulator")["weight"].apply(lambda x: x.abs().sum()), 1.0
    )
    assert not any(
        name in regulators.columns
        for name in ["pvalue", "p_value", "fdr", "causal_probability"]
    )


@pytest.mark.parametrize("weight", [0.0, np.nan, np.inf])
def test_invalid_regulatory_weights_are_rejected(weight):
    a = _make_affine()
    with pytest.raises(ValueError, match="(?i)weight|finite|nonzero"):
        _explain(a, regulons={"bad": {"g_up": weight}}, regulator_source="synthetic")


def test_large_finite_regulon_weights_preserve_signed_target_response():
    a = _make_affine()
    result = _explain(
        a,
        regulons={"R": {"g_up": 1e308, "g_down": -1e308}},
        regulator_source="synthetic",
    )
    weights = result["regulon_membership"].set_index("gene")["weight"]
    assert weights.loc["g_up"] == pytest.approx(0.5)
    assert weights.loc["g_down"] == pytest.approx(-0.5)
    assert result["regulators"]["signed_target_delta"].iloc[0] == pytest.approx(
        (1.4 + 0.45) / 2
    )


def test_result_survives_h5ad_roundtrip(tmp_path):
    a = _make_local_example()
    result = _explain(
        a, local_key="local_state", modules={"program": ["response"]}, n_boot=16
    )
    path = tmp_path / "explained_shift.h5ad"
    a.write_h5ad(path)
    restored = ad.read_h5ad(path)
    saved = restored.uns["scgeo"]["explain_shift"]
    for key in [
        "summary",
        "genes",
        "programs",
        "samples",
        "cells",
        "local_summary",
        "local_genes",
        "composition",
    ]:
        pd.testing.assert_frame_equal(
            result[key], saved[key], check_dtype=False, check_categorical=False
        )
    rerun = _explain(
        restored, local_key="local_state", modules={"program": ["response"]}, n_boot=16
    )
    pd.testing.assert_frame_equal(
        saved["local_genes"],
        rerun["local_genes"],
        check_dtype=False,
        check_categorical=False,
    )


@pytest.mark.parametrize("column", ["condition", "sample", "cell_type"])
def test_missing_design_labels_do_not_become_phantom_samples_or_states(column):
    a = _make_affine()
    a.obs.loc[a.obs.index[0], column] = None
    with pytest.raises(ValueError, match="(?i)missing|null|label|stale|stored|changed"):
        _explain(a)
