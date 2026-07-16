import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCENARIOS = [
    "null_effect",
    "centroid_shift",
    "abundance_only",
    "covariance_only",
    "local_warp",
    "outlier_contamination",
    "unequal_cell_counts",
    "balanced_replicate_heterogeneity",
    "batch_condition_confounding",
    "representation_corruption",
    "aligned_dynamics",
    "discordant_dynamics",
]


def _small_sim(**kwargs):
    import scgeo as sg

    params = dict(n_states=4, n_samples_per_condition=2, cells_per_sample=35, latent_dim=4)
    params.update(kwargs)
    return sg.bench.simulate_perturbation_geometry(**params)


def _run_small_stack(adata):
    import scgeo as sg

    truth = adata.uns["simulation_truth"]
    consensus_reps = list(truth["equivalent_representations"]) + list(truth.get("corrupted_representations", []))
    local_geometry_reps = list(truth["representations"])
    velocity_keys = truth.get("velocity_keys")
    if velocity_keys is not None:
        velocity_keys = {rep: velocity_keys.get(rep) for rep in consensus_reps}
    sg.tl.robust_shift(
        adata,
        rep="X_truth",
        condition_key=truth["condition_key"],
        group0=truth["group0"],
        group1=truth["group1"],
        by=truth["state_key"],
        sample_key=truth["sample_key"],
        center="geometric_median",
        n_boot=3,
        seed=truth["seed"],
    )
    sg.tl.representation_stability(
        adata,
        reps=consensus_reps,
        node_key=truth["state_key"],
        condition_key=truth["condition_key"],
        group0=truth["group0"],
        group1=truth["group1"],
        sample_key=truth["sample_key"],
        center="mean",
        n_boot=1,
        velocity_keys=velocity_keys,
        min_cells=5,
        seed=truth["seed"],
    )
    sg.tl.local_geometry_stability(
        adata,
        reps=local_geometry_reps,
        node_key=truth["state_key"],
        sample_key=truth["sample_key"],
        k_values=(5,),
        n_boot=0,
        max_exact_cells=250,
        seed=truth["seed"],
    )
    return adata


def _synthetic_full_smoke_ablation_table():
    from scgeo.bench._simulation import _ABLATION_VARIANTS

    rows = []

    def add(scenario, failure_mode, variant, outcome, *, status="assessed", target="state_0"):
        rows.append(
            {
                "profile": "smoke",
                "seed_split": "evaluation",
                "job_id": f"smoke_{scenario}_evaluation_seed1",
                "scenario": scenario,
                "seed": 1,
                "variant": variant,
                "variant_label": _ABLATION_VARIANTS[variant],
                "failure_mode": failure_mode,
                "unit": "state",
                "target": target,
                "truth": outcome in {"detects_correctly", "misses"},
                "call": outcome in {"detects_correctly", "falsely_calls"},
                "status": status,
                "outcome": outcome,
                "evidence": "unit test",
                "final_evaluation": True,
            }
        )

    variants = list(_ABLATION_VARIANTS)
    shift_scenarios = [
        "null_effect",
        "centroid_shift",
        "outlier_contamination",
        "unequal_cell_counts",
        "balanced_replicate_heterogeneity",
    ]
    for scenario in shift_scenarios:
        for variant in variants:
            outcome = "correctly_rejects" if scenario == "null_effect" else "detects_correctly"
            add(scenario, "centroid_shift", variant, outcome)

    for variant in variants[:3]:
        add("representation_corruption", "explicit_corruption_detection", variant, "not_computed", status="not_computed")
        add(
            "representation_corruption",
            "representation_distortion_detection",
            variant,
            "not_computed",
            status="not_computed",
        )
    for variant in variants[3:]:
        add("representation_corruption", "explicit_corruption_detection", variant, "detects_correctly", target="X_corrupted")
        add(
            "representation_corruption",
            "representation_distortion_detection",
            variant,
            "detects_correctly",
            target="X_anisotropic",
        )

    for scenario, outcome in [("aligned_dynamics", "detects_correctly"), ("discordant_dynamics", "detects_correctly")]:
        for variant in variants[:4]:
            add(scenario, "dynamics_alignment", variant, "not_computed", status="not_computed")
        add(scenario, "dynamics_alignment", variants[-1], outcome)

    for variant in variants:
        add("abundance_only", "abundance_change", variant, "not_computed", status="not_computed")
        add(
            "covariance_only",
            "condition_distribution_shape_change",
            variant,
            "not_computed",
            status="not_computed",
        )
        add("local_warp", "representation_local_distortion", variant, "not_applicable", status="not_applicable")
        add("batch_condition_confounding", "centroid_shift", variant, "non_identifiable", status="non_identifiable")
    for variant in variants[2:]:
        add(
            "representation_corruption",
            "representation_instability",
            variant,
            "representation_unstable",
            status="representation_unstable",
        )
    for variant in variants[1:]:
        add(
            "balanced_replicate_heterogeneity",
            "bootstrap_interval_coverage",
            variant,
            "insufficient_coverage",
            status="insufficient_coverage",
        )
    for scenario in ["null_effect", "centroid_shift", "abundance_only", "covariance_only"]:
        add(scenario, "dynamics_alignment", variants[-1], "unavailable", status="unavailable")

    return pd.DataFrame(rows)


def _synthetic_main_ablation_rows(n_rows: int, *, support_rows: int = 0):
    from scgeo.bench._simulation import _ABLATION_VARIANTS, _MAIN_ABLATION_ROW_SPECS

    rows = []
    variants = list(_ABLATION_VARIANTS)
    for scenario, failure_mode, _label in list(_MAIN_ABLATION_ROW_SPECS)[: int(n_rows)]:
        for variant in variants:
            rows.append(
                {
                    "profile": "unit",
                    "seed_split": "evaluation",
                    "job_id": f"job_{scenario}",
                    "scenario": scenario,
                    "seed": 1,
                    "variant": variant,
                    "variant_label": _ABLATION_VARIANTS[variant],
                    "failure_mode": failure_mode,
                    "unit": "state",
                    "target": "state_0",
                    "truth": True,
                    "call": True,
                    "status": "assessed",
                    "outcome": "detects_correctly",
                    "evidence": "unit test",
                    "final_evaluation": True,
                }
            )
    support_defs = [
        ("representation_corruption", "representation_instability", "representation_unstable"),
        ("balanced_replicate_heterogeneity", "bootstrap_interval_coverage", "insufficient_coverage"),
    ]
    for idx in range(int(support_rows)):
        scenario, failure_mode, outcome = support_defs[idx % len(support_defs)]
        variant = variants[min(idx + 2, len(variants) - 1)]
        rows.append(
            {
                "profile": "unit",
                "seed_split": "evaluation",
                "job_id": f"support_{idx}",
                "scenario": scenario,
                "seed": 1,
                "variant": variant,
                "variant_label": _ABLATION_VARIANTS[variant],
                "failure_mode": failure_mode,
                "unit": "state",
                "target": f"state_{idx}",
                "truth": True,
                "call": True,
                "status": outcome,
                "outcome": outcome,
                "evidence": "unit test",
                "final_evaluation": True,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_supplemental_rows(n_rows: int, *, long_labels: bool = False, unsupported: bool = False):
    from scgeo.bench._simulation import _ABLATION_VARIANTS

    rows = []
    variants = list(_ABLATION_VARIANTS)
    for idx in range(int(n_rows)):
        scenario = (
            f"very_long_synthetic_scenario_name_with_multiple_descriptive_segments_{idx:02d}"
            if long_labels
            else f"scenario_{idx:02d}"
        )
        failure_mode = (
            "very_long_failure_mode_label_for_wrapping_and_spacing_checks"
            if long_labels
            else "centroid_shift"
        )
        for variant_idx, variant in enumerate(variants):
            if unsupported and idx % 3 == 0 and variant_idx < 2:
                outcome = "not_computed"
                status = "not_computed"
            else:
                outcome = "detects_correctly"
                status = "assessed"
            rows.append(
                {
                    "profile": "unit",
                    "seed_split": "evaluation",
                    "job_id": f"job_{idx:02d}",
                    "scenario": scenario,
                    "seed": idx,
                    "variant": variant,
                    "variant_label": _ABLATION_VARIANTS[variant],
                    "failure_mode": failure_mode,
                    "unit": "state",
                    "target": f"state_{idx}",
                    "truth": True,
                    "call": outcome == "detects_correctly",
                    "status": status,
                    "outcome": outcome,
                    "evidence": "unit test",
                    "final_evaluation": True,
                }
            )
    return pd.DataFrame(rows)


def test_simulate_every_scenario_truth_schema_and_synthetic_boundary():
    import scgeo as sg

    for scenario in SCENARIOS:
        adata = _small_sim(scenario=scenario, seed=3)
        truth = adata.uns["simulation_truth"]
        assert {"state", "condition", "sample"} <= set(adata.obs.columns)
        assert {"X_truth", "X_rotated", "X_scaled", "X_padded", "X_anisotropic", "X_nonlinear"} <= set(adata.obsm.keys())
        required = {
            "shifted_states",
            "true_effect_magnitude",
            "true_abundance_change",
            "distorted_states",
            "corrupted_representations",
            "explicitly_corrupted_representations",
            "diagnostically_distorted_representations",
            "representation_categories",
            "dynamics_class",
            "params",
            "seed",
        }
        assert required <= set(truth.keys())
        serialized = json.dumps(truth)
        assert "http://" not in serialized
        assert "https://" not in serialized
        assert "/mnt/" not in serialized
        if scenario == "representation_corruption":
            assert "X_corrupted" in adata.obsm
            assert truth["corrupted_representations"] == ["X_corrupted"]
        if scenario in {"aligned_dynamics", "discordant_dynamics"}:
            assert truth["velocity_keys"] is not None
            assert "V_rotated" in adata.obsm

    assert callable(sg.bench.simulate_perturbation_geometry)


def test_simulation_is_deterministic():
    adata1 = _small_sim(scenario="centroid_shift", seed=5)
    adata2 = _small_sim(scenario="centroid_shift", seed=5)
    np.testing.assert_allclose(adata1.obsm["X_truth"], adata2.obsm["X_truth"])
    pd.testing.assert_frame_equal(adata1.obs, adata2.obs)
    assert adata1.uns["simulation_truth"] == adata2.uns["simulation_truth"]


def test_evaluate_ground_truth_tidy_schemas():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="centroid_shift", seed=7))
    out = sg.bench.evaluate_ground_truth(adata)
    expected = {
        "state_metrics",
        "rank_metrics",
        "shift_detection",
        "threshold_sensitivity",
        "bootstrap_coverage",
        "representation_consensus",
        "alignment_accuracy",
        "representation_quality_outlier",
        "representation_distortion_detection",
        "explicit_corruption_detection",
        "corruption_detection",
        "distorted_state_detection",
        "neighborhood_discrimination",
        "abundance_truth",
        "condition_shape_truth",
        "non_identifiable_truth",
        "summary_metrics",
        "coverage_summary",
        "sample_offset_diagnostics",
        "sample_center_diagnostics",
        "shift_estimate_diagnostics",
        "runtime_summary",
    }
    assert expected <= set(out)
    assert {"scenario", "seed", "method", "state", "magnitude_error"} <= set(out["state_metrics"].columns)
    assert {"threshold", "precision", "recall", "f1"} <= set(out["threshold_sensitivity"].columns)
    assert {"state", "consensus_label", "status"} <= set(out["representation_consensus"].columns)
    assert {"state", "true_abundance_change", "true_abundance_changed"} <= set(out["abundance_truth"].columns)
    assert {"state", "coverage_status", "coverage_applicable", "interval_kind"} <= set(out["bootstrap_coverage"].columns)
    assert {"state", "true_non_identifiable", "sample_offset_design", "expected_outcome"} <= set(
        out["non_identifiable_truth"].columns
    )


def test_shift_detection_threshold_is_predeclared_not_truth_scaled():
    import scgeo as sg

    low = sg.bench.evaluate_ground_truth(_small_sim(scenario="centroid_shift", effect_size=0.4, seed=31))
    high = sg.bench.evaluate_ground_truth(_small_sim(scenario="centroid_shift", effect_size=1.6, seed=31))

    assert set(low["state_metrics"]["threshold"]) == {0.5}
    assert set(high["state_metrics"]["threshold"]) == {0.5}
    assert set(low["threshold_sensitivity"]["threshold"]) == {0.25, 0.5, 0.75, 1.0, 1.25}
    assert set(high["threshold_sensitivity"]["threshold"]) == {0.25, 0.5, 0.75, 1.0, 1.25}


def test_framework_ablation_schema_and_outcomes():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=23))
    out = sg.bench.evaluate_ground_truth(adata)
    ablation = sg.bench.framework_ablation(out)
    required = {
        "scenario",
        "seed",
        "seed_split",
        "variant",
        "failure_mode",
        "unit",
        "target",
        "truth",
        "call",
        "status",
        "outcome",
        "final_evaluation",
    }
    assert required <= set(ablation.columns)
    assert not ablation.empty
    assert ablation["final_evaluation"].all()
    assert set(ablation["variant"]) == {
        "A_mean_shift_one_rep",
        "B_robust_shift_one_rep",
        "C_robust_shift_representation_consensus",
        "D_plus_local_geometry",
        "E_plus_dynamics",
    }
    assert "score" not in ablation.columns
    assert "composite_score" not in ablation.columns
    assert "insufficient_or_unstable" not in set(ablation["outcome"])
    assert "explicit_corruption_detection" in set(ablation["failure_mode"])
    corr = ablation[
        (ablation["variant"] == "D_plus_local_geometry")
        & (ablation["failure_mode"] == "explicit_corruption_detection")
        & (ablation["target"] == "X_corrupted")
    ]
    assert not corr.empty
    assert "detects_correctly" in set(corr["outcome"])


def test_zero_effect_bootstrap_coverage_not_applicable_and_no_ablation_miss():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="null_effect", seed=41))
    out = sg.bench.evaluate_ground_truth(adata)
    coverage = out["bootstrap_coverage"]

    assert set(coverage["coverage_status"]) == {"not_applicable"}
    assert not coverage["coverage_applicable"].any()
    assert coverage["covered"].isna().all()

    ablation = sg.bench.framework_ablation(out)
    boot = ablation[
        (ablation["failure_mode"] == "bootstrap_interval_coverage")
        & (ablation["variant"] != "A_mean_shift_one_rep")
    ]
    assert not boot.empty
    assert set(boot["outcome"]) == {"not_applicable"}
    assert "misses" not in set(boot["outcome"])


def test_abundance_only_equivalent_representations_remain_stable_neutral():
    import scgeo as sg

    adata = _run_small_stack(
        _small_sim(
            scenario="abundance_only",
            n_samples_per_condition=3,
            cells_per_sample=120,
            sample_heterogeneity=0.0,
            seed=43,
        )
    )
    out = sg.bench.evaluate_ground_truth(adata)
    consensus = out["representation_consensus"]

    assert not consensus.empty
    assert "representation_unstable" not in set(consensus["consensus_label"])
    assert set(consensus["consensus_label"]) == {"stable_neutral"}


def test_substantive_corrupted_representation_still_triggers_instability():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=45))
    out = sg.bench.evaluate_ground_truth(adata)
    consensus = out["representation_consensus"]

    assert "representation_unstable" in set(consensus["consensus_label"])


def test_covariance_only_truth_not_conflated_with_representation_distortion():
    import scgeo as sg

    adata = _run_small_stack(
        _small_sim(
            scenario="covariance_only",
            n_samples_per_condition=3,
            cells_per_sample=120,
            sample_heterogeneity=0.0,
            seed=47,
        )
    )
    out = sg.bench.evaluate_ground_truth(adata)
    distortion = out["distorted_state_detection"]

    assert distortion["true_condition_distribution_shape_changed"].any()
    assert not distortion["true_representation_local_distortion"].any()

    ablation = sg.bench.framework_ablation(out)
    local_rows = ablation[
        (ablation["failure_mode"] == "representation_local_distortion")
        & (ablation["variant"].isin(["D_plus_local_geometry", "E_plus_dynamics"]))
    ]
    assert not local_rows.empty
    assert not local_rows["truth"].astype(bool).any()
    assert "misses" not in set(local_rows["outcome"])

    shape_rows = ablation[ablation["failure_mode"] == "condition_distribution_shape_change"]
    assert not shape_rows.empty
    assert set(shape_rows["outcome"]) == {"not_computed"}


def test_diagnostic_representations_not_false_explicit_corruption_calls():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=49))
    out = sg.bench.evaluate_ground_truth(adata)
    explicit = out["explicit_corruption_detection"].set_index("rep")

    for rep in ["X_anisotropic", "X_nonlinear"]:
        assert rep in explicit.index
        assert explicit.loc[rep, "representation_category"] == "diagnostically_distorted"
        assert explicit.loc[rep, "status"] == "not_applicable"
        assert not bool(explicit.loc[rep, "true_corrupted"])

    assert bool(explicit.loc["X_corrupted", "true_corrupted"])


def test_equivalent_and_intended_representation_pair_distortion_truth():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=51))
    truth = adata.uns["simulation_truth"]
    affected = set(truth["representation_distortion_affected_states"])
    out = sg.bench.evaluate_ground_truth(adata)
    distortion = out["distorted_state_detection"]

    equivalent_pairs = distortion[distortion["equivalent_pair"]]
    assert not equivalent_pairs.empty
    assert not equivalent_pairs["true_representation_local_distortion"].any()

    intended = distortion[
        (distortion["distorted_representation"].isin(["X_anisotropic", "X_nonlinear"]))
        & (distortion["reference_representation"] != "")
        & (distortion["state"].isin(affected))
    ]
    assert not intended.empty
    assert intended["true_representation_local_distortion"].all()

    unaffected = distortion[
        (distortion["distorted_representation"].isin(["X_anisotropic", "X_nonlinear"]))
        & (distortion["reference_representation"] != "")
        & (~distortion["state"].isin(affected))
    ]
    assert not unaffected.empty
    assert not unaffected["true_representation_local_distortion"].any()


def test_corruption_does_not_make_every_neutral_state_instability_truth():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=53))
    out = sg.bench.evaluate_ground_truth(adata)
    ablation = sg.bench.framework_ablation(out)
    rows = ablation[
        (ablation["failure_mode"] == "representation_instability")
        & (ablation["variant"] == "C_robust_shift_representation_consensus")
    ]
    state_truth = rows.set_index("target")["truth"].astype(bool)
    shifted = set(adata.uns["simulation_truth"]["shifted_states"])

    assert state_truth.loc[list(shifted)].all()
    assert not state_truth.drop(index=list(shifted)).any()


def test_framework_ablation_plot_returns_figure():
    import matplotlib.pyplot as plt
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="aligned_dynamics", seed=29))
    ablation = sg.bench.framework_ablation(sg.bench.evaluate_ground_truth(adata))
    fig = sg.bench.plot_framework_ablation(ablation, show=False)
    assert fig.__class__.__name__ == "Figure"
    assert fig.get_size_inches()[1] <= 7.8
    titles = {ax.get_title() for ax in fig.axes}
    assert {"Capability", "Applicable Performance", "Support Status"} <= titles
    assert fig.scgeo_ablation_matrices["capability"].shape == (5, 5)
    plt.close(fig)


def test_framework_ablation_plot_matrices_are_compact_and_mask_unsupported_cells():
    import matplotlib.pyplot as plt
    import scgeo as sg

    ablation = _synthetic_full_smoke_ablation_table()
    fig = sg.bench.plot_framework_ablation(ablation, show=False)
    matrices = fig.scgeo_ablation_matrices
    capability = matrices["capability"]
    performance = matrices["performance"]
    support = matrices["support"]

    assert capability.shape == (5, 5)
    assert list(capability.index) == ["A", "B", "C", "D", "E"]
    assert list(capability.columns) == [
        "Effect",
        "Uncertainty",
        "Representation stability",
        "Local geometry",
        "Dynamics",
    ]
    assert set(performance.index) == {
        "Null effect | Shift",
        "Centroid shift | Shift",
        "Outliers | Shift",
        "Unequal counts | Shift",
        "Replicate het. | Shift",
        "Explicit corrupt. | Corruption",
        "Rep. distortion | Distortion",
        "Aligned dyn. | Dynamics",
        "Discordant dyn. | Dynamics",
    }
    assert not any("abundance" in label.lower() for label in performance.index)
    assert not any("condition" in label.lower() for label in performance.index)
    assert performance.loc["Explicit corrupt. | Corruption", "A_mean_shift_one_rep"] != performance.loc[
        "Explicit corrupt. | Corruption", "A_mean_shift_one_rep"
    ]
    assert performance.loc["Rep. distortion | Distortion", "C_robust_shift_representation_consensus"] != performance.loc[
        "Rep. distortion | Distortion", "C_robust_shift_representation_consensus"
    ]
    for outcome in ["not_computed", "not_applicable", "unavailable", "misses", "falsely_calls"]:
        assert not any(outcome in label for label in performance.index)
        assert not any(outcome in label for label in support.index)
    assert not any("Null effect | Dynamics" in label for label in support.index)
    assert any("non-ident." in label for label in support.index)
    plt.close(fig)


def test_framework_ablation_short_labels_are_deterministic():
    from scgeo.bench._simulation import _framework_ablation_plot_matrices, _short_ablation_row_label

    assert _short_ablation_row_label("balanced_replicate_heterogeneity", "centroid_shift") == "Replicate het. | Shift"
    assert _short_ablation_row_label("representation_corruption", "explicit_corruption_detection") == (
        "Explicit corrupt. | Corruption"
    )

    matrices = _framework_ablation_plot_matrices(_synthetic_full_smoke_ablation_table())
    assert list(matrices["performance"].index) == [
        "Null effect | Shift",
        "Centroid shift | Shift",
        "Outliers | Shift",
        "Unequal counts | Shift",
        "Replicate het. | Shift",
        "Explicit corrupt. | Corruption",
        "Rep. distortion | Distortion",
        "Aligned dyn. | Dynamics",
        "Discordant dyn. | Dynamics",
    ]


def test_framework_ablation_plot_does_not_filter_long_form_table():
    import matplotlib.pyplot as plt
    import scgeo as sg

    ablation = _synthetic_full_smoke_ablation_table()
    before = ablation.copy(deep=True)
    fig = sg.bench.plot_framework_ablation(ablation, show=False)

    pd.testing.assert_frame_equal(ablation, before)
    assert {"not_computed", "not_applicable", "non_identifiable", "unavailable"} <= set(ablation["outcome"])
    assert "abundance_change" in set(ablation["failure_mode"])
    plt.close(fig)


def test_framework_ablation_plot_legible_with_full_smoke_scenario_set():
    import matplotlib.pyplot as plt
    import scgeo as sg

    fig = sg.bench.plot_framework_ablation(_synthetic_full_smoke_ablation_table(), show=False)
    matrices = fig.scgeo_ablation_matrices

    width, height = fig.get_size_inches()
    assert width <= 11.0
    assert height <= 7.8
    assert matrices["capability"].shape == (5, 5)
    assert matrices["performance"].shape[0] == 9
    assert matrices["support"].shape[0] <= 4
    assert fig.scgeo_ablation_plot_metadata["page_count"] == 1
    plt.close(fig)


def test_framework_ablation_auto_sizing_for_row_counts_and_support_rows():
    import matplotlib.pyplot as plt
    import scgeo as sg

    heights = []
    for n_rows in [1, 2, 9]:
        fig = sg.bench.plot_framework_ablation(_synthetic_main_ablation_rows(n_rows), show=False)
        metadata = fig.scgeo_ablation_plot_metadata
        assert metadata["figure_dimensions"]["height"] >= metadata["auto_sizing"]["min_height"]
        assert len(metadata["displayed_labels"]["performance_rows"]) == n_rows
        heights.append(metadata["figure_dimensions"]["height"])
        plt.close(fig)
    assert heights[0] <= heights[1] <= heights[2]

    no_support = sg.bench.plot_framework_ablation(_synthetic_main_ablation_rows(2, support_rows=0), show=False)
    two_support = sg.bench.plot_framework_ablation(_synthetic_main_ablation_rows(2, support_rows=2), show=False)
    assert no_support.scgeo_ablation_plot_metadata["original_labels"]["support_rows"] == []
    assert len(two_support.scgeo_ablation_plot_metadata["original_labels"]["support_rows"]) == 2
    assert two_support.scgeo_ablation_plot_metadata["panel_dimensions"]["support"]["height"] < (
        two_support.scgeo_ablation_plot_metadata["panel_dimensions"]["performance"]["height"]
    )
    plt.close(no_support)
    plt.close(two_support)


def test_framework_ablation_long_labels_do_not_overlap_after_draw():
    import matplotlib.pyplot as plt
    import scgeo as sg

    ablation = _synthetic_main_ablation_rows(2)
    long_row = _synthetic_supplemental_rows(1, long_labels=True).iloc[0].copy()
    long_row["outcome"] = "non_identifiable"
    long_row["status"] = "non_identifiable"
    ablation = pd.concat([ablation, long_row.to_frame().T], ignore_index=True)

    fig = sg.bench.plot_framework_ablation(ablation, wrap_width=12, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_by_title = {ax.get_title(): ax for ax in fig.axes}
    perf_ax = axes_by_title["Applicable Performance"]
    support_ax = axes_by_title["Support Status"]
    support_bbox = support_ax.get_window_extent(renderer)
    for label in perf_ax.get_yticklabels():
        if label.get_text():
            assert label.get_window_extent(renderer).x1 < support_bbox.x0
    fig_bbox = fig.bbox
    for ax in [perf_ax, support_ax]:
        for label in ax.get_yticklabels():
            if label.get_text():
                bbox = label.get_window_extent(renderer)
                assert bbox.x0 >= fig_bbox.x0
                assert bbox.x1 <= fig_bbox.x1
    assert any("\n" in label for label in fig.scgeo_ablation_plot_metadata["displayed_labels"]["support_rows"])
    plt.close(fig)


def test_framework_ablation_supplemental_row_counts_masking_and_pagination(tmp_path):
    import matplotlib.pyplot as plt
    from scgeo.bench._simulation import _plot_framework_ablation_supplemental

    result_25 = _plot_framework_ablation_supplemental(
        _synthetic_supplemental_rows(25, unsupported=True),
        save_path=tmp_path / "supplemental.png",
        show=False,
    )
    assert result_25["metadata"]["page_count"] == 1
    assert result_25["paths"] == [str(tmp_path / "supplemental.png")]
    assert result_25["metadata"]["masked_cell_count"] > 0
    for fig in result_25["figures"]:
        plt.close(fig)

    result_26_png = _plot_framework_ablation_supplemental(
        _synthetic_supplemental_rows(26, long_labels=True, unsupported=True),
        save_path=tmp_path / "detail.png",
        max_rows_per_page=25,
        wrap_width=14,
        show=False,
    )
    result_26_svg = _plot_framework_ablation_supplemental(
        _synthetic_supplemental_rows(26, long_labels=True, unsupported=True),
        save_path=tmp_path / "detail.svg",
        max_rows_per_page=25,
        wrap_width=14,
        show=False,
    )
    assert result_26_png["metadata"]["page_count"] == 2
    assert result_26_png["paths"] == [
        str(tmp_path / "detail_page01.png"),
        str(tmp_path / "detail_page02.png"),
    ]
    assert result_26_svg["paths"] == [
        str(tmp_path / "detail_page01.svg"),
        str(tmp_path / "detail_page02.svg"),
    ]
    assert all(Path(path).exists() for path in result_26_png["paths"] + result_26_svg["paths"])
    assert any("\n" in label for label in result_26_png["metadata"]["pages"][0]["displayed_labels"]["rows"])
    assert "very_long_synthetic_scenario_name" in result_26_png["metadata"]["pages"][0]["original_labels"]["rows"][0]
    for result in [result_26_png, result_26_svg]:
        for fig in result["figures"]:
            plt.close(fig)


def test_null_effect_survives_default_pandas_csv_round_trip(tmp_path):
    import scgeo as sg

    out = sg.bench.evaluate_ground_truth(_small_sim(scenario="null", seed=57))
    path = tmp_path / "state_metrics.csv"
    out["state_metrics"].to_csv(path, index=False)
    loaded = pd.read_csv(path)

    assert loaded["scenario"].notna().all()
    assert set(loaded["scenario"]) == {"null_effect"}


def test_seed_level_ablation_summary_aggregates_jobs_not_state_rows():
    from scgeo.bench._simulation import _summarize_framework_ablation, _seed_level_framework_ablation

    rows = []
    for job_id, seed, outcomes in [
        ("job_1", 1, ["detects_correctly", "detects_correctly", "misses"]),
        ("job_2", 2, ["misses", "misses"]),
    ]:
        for i, outcome in enumerate(outcomes):
            rows.append(
                {
                    "profile": "unit",
                    "seed_split": "evaluation",
                    "job_id": job_id,
                    "scenario": "centroid_shift",
                    "seed": seed,
                    "variant": "B_robust_shift_one_rep",
                    "variant_label": "B. robust shift on one representation",
                    "failure_mode": "centroid_shift",
                    "unit": "state",
                    "target": f"s{i}",
                    "truth": True,
                    "call": outcome == "detects_correctly",
                    "status": "assessed",
                    "outcome": outcome,
                    "evidence": "unit",
                    "final_evaluation": True,
                }
            )
    ablation = pd.DataFrame(rows)

    seed_summary = _seed_level_framework_ablation(ablation)
    detects = seed_summary[
        (seed_summary["outcome"] == "detects_correctly")
        & (seed_summary["job_id"].isin(["job_1", "job_2"]))
    ]
    assert set(detects["rate"]) == {2 / 3, 0.0}

    summary = _summarize_framework_ablation(ablation)
    row = summary[summary["outcome"] == "detects_correctly"].iloc[0]
    assert row["n_jobs"] == 2
    assert np.isclose(row["mean_rate"], (2 / 3 + 0.0) / 2)
    assert not np.isclose(row["mean_rate"], 2 / 5)
    assert np.isfinite(row["ci95_low"])
    assert np.isfinite(row["ci95_high"])


def test_framework_ablation_plot_requires_requested_split():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="null_effect", seed=37))
    ablation = sg.bench.framework_ablation(sg.bench.evaluate_ground_truth(adata))
    ablation["seed_split"] = "calibration"

    with pytest.raises(ValueError, match="no rows to plot"):
        sg.bench.plot_framework_ablation(ablation, split="evaluation", show=False)


def test_null_abundance_and_covariance_do_not_force_centroid_shift_calls():
    import scgeo as sg

    for scenario in ["null_effect", "abundance_only", "covariance_only"]:
        adata = _small_sim(
            scenario=scenario,
            n_samples_per_condition=3,
            cells_per_sample=120,
            sample_heterogeneity=0.0,
            seed=11,
        )
        out = sg.bench.evaluate_ground_truth(adata)
        robust = out["state_metrics"][out["state_metrics"]["method"] == "robust_geometric_median"]
        assert not robust["predicted_shifted"].any()
        assert robust["estimated_magnitude"].max() < 0.35


def test_balanced_replicate_heterogeneity_does_not_systematically_shift_neutral_states():
    import scgeo as sg

    adata = _small_sim(
        scenario="balanced_replicate_heterogeneity",
        n_samples_per_condition=4,
        cells_per_sample=160,
        sample_heterogeneity=0.6,
        seed=61,
    )
    truth = adata.uns["simulation_truth"]
    assert truth["replicate_design"]["sample_offset_design"] == "condition_centered_random_replicate_offsets"
    assert truth["replicate_design"]["zero_centered_within_condition"]
    offsets = pd.DataFrame(truth["sample_offsets"])
    dim_cols = [col for col in offsets.columns if col.startswith("offset_dim_")]
    means = offsets.groupby("condition")[dim_cols].mean()
    assert np.allclose(means.to_numpy(dtype=float), 0.0, atol=1e-12)

    out = sg.bench.evaluate_ground_truth(adata)
    robust = out["state_metrics"][out["state_metrics"]["method"] == "robust_geometric_median"]
    neutral = robust[~robust["true_shifted"]]
    assert not neutral["predicted_shifted"].any()
    assert neutral["estimated_magnitude"].max() < 0.35


def test_batch_condition_confounding_truth_metadata_marks_non_identifiable_design():
    adata = _small_sim(
        scenario="batch_condition_confounding",
        n_samples_per_condition=4,
        cells_per_sample=120,
        sample_heterogeneity=0.6,
        seed=63,
    )
    truth = adata.uns["simulation_truth"]
    design = truth["replicate_design"]
    assert design["sample_offset_design"] == "condition_correlated_batch_offset"
    assert design["systematic_condition_confounding"]
    assert design["non_identifiable"]
    assert not design["expected_biological_shift_identifiable"]
    assert set(truth["non_identifiable_states"]) == set(truth["states"])
    assert truth["shifted_states"] == []

    offsets = pd.DataFrame(truth["sample_offsets"])
    dim_cols = [col for col in offsets.columns if col.startswith("offset_dim_")]
    condition_means = offsets.groupby("condition")[dim_cols].mean()
    mean_gap = condition_means.loc["treated"].to_numpy(dtype=float) - condition_means.loc["control"].to_numpy(dtype=float)
    assert np.linalg.norm(mean_gap) > 1.0


def test_non_identifiable_confounding_not_counted_as_ordinary_biological_false_positives():
    import scgeo as sg

    adata = _small_sim(
        scenario="batch_condition_confounding",
        n_samples_per_condition=4,
        cells_per_sample=120,
        sample_heterogeneity=0.6,
        seed=65,
    )
    out = sg.bench.evaluate_ground_truth(adata)
    metrics = out["shift_detection"]
    assert (metrics["n_evaluable_states"] == 0).all()
    assert (metrics["n_non_identifiable_states"] == len(adata.uns["simulation_truth"]["states"])).all()
    assert (metrics["fp"] == 0).all()
    assert metrics["null_state_false_classification_rate"].isna().all()

    ablation = sg.bench.framework_ablation(out)
    centroid = ablation[ablation["failure_mode"] == "centroid_shift"]
    assert not centroid.empty
    assert set(centroid["outcome"]) == {"non_identifiable"}
    assert "falsely_calls" not in set(centroid["outcome"])


def test_replicate_heterogeneity_scenarios_and_expected_outcomes_are_documented():
    from scgeo.bench._simulation import _normalize_scenarios

    legacy = _small_sim(scenario="replicate_heterogeneity", seed=67)
    assert legacy.uns["simulation_truth"]["scenario"] == "balanced_replicate_heterogeneity"

    scenarios = _normalize_scenarios(None)
    assert "balanced_replicate_heterogeneity" in scenarios
    assert "batch_condition_confounding" in scenarios
    assert "replicate_heterogeneity" not in scenarios

    root = Path(__file__).resolve().parents[1]
    docs = (root / "docs" / "api_reference.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for text in (docs, readme):
        assert "balanced_replicate_heterogeneity" in text
        assert "batch_condition_confounding" in text
        assert "non-identifiable" in text


def test_outlier_robustness_beats_arithmetic_mean():
    import scgeo as sg

    adata = _small_sim(scenario="outlier_contamination", outlier_fraction=0.12, seed=13)
    out = sg.bench.evaluate_ground_truth(adata)
    shifted = set(adata.uns["simulation_truth"]["shifted_states"])
    state_metrics = out["state_metrics"][out["state_metrics"]["state"].isin(shifted)]
    mean_err = state_metrics[state_metrics["method"] == "shift_mean"]["magnitude_error"].mean()
    robust_err = state_metrics[state_metrics["method"] == "robust_geometric_median"]["magnitude_error"].mean()
    assert robust_err < mean_err


def test_corrupted_representation_detection():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="representation_corruption", seed=17))
    out = sg.bench.evaluate_ground_truth(adata)
    corr = out["corruption_detection"].set_index("rep")
    assert bool(corr.loc["X_corrupted", "true_corrupted"])
    assert bool(corr.loc["X_corrupted", "predicted_corrupted"])


def test_aligned_and_discordant_dynamics_recovered_in_single_representation():
    import scgeo as sg

    for scenario, expected in [("aligned_dynamics", "aligned"), ("discordant_dynamics", "discordant")]:
        adata = _run_small_stack(_small_sim(scenario=scenario, seed=19))
        out = sg.bench.evaluate_ground_truth(adata)
        shifted = set(adata.uns["simulation_truth"]["shifted_states"])
        rows = out["alignment_accuracy"][
            (out["alignment_accuracy"]["source"] == "single_representation")
            & (out["alignment_accuracy"]["rep"] == "X_truth")
            & (out["alignment_accuracy"]["state"].isin(shifted))
        ]
        assert not rows.empty
        assert set(rows["predicted_class"]) == {expected}
        assert rows["correct"].all()


def test_smoke_suite_execution_resume_and_seed_split(tmp_path, capsys):
    import scgeo as sg

    out1 = sg.bench.run_simulation_suite(
        profile="smoke",
        scenarios=["null_effect"],
        seeds={"calibration": [0], "evaluation": [1]},
        output_dir=tmp_path,
        n_jobs=1,
    )
    printed = capsys.readouterr().out
    assert "dry run" in printed
    assert set(out1["jobs"]["status"]) == {"completed"}
    assert out1["calibration_seeds"] == [0]
    assert out1["evaluation_seeds"] == [1]
    assert {"calibration", "evaluation"} <= set(out1["tables"]["state_metrics"]["seed_split"])
    assert "framework_ablation" in out1["tables"]
    assert "framework_ablation_seed_summary" in out1["tables"]
    assert {"calibration", "evaluation"} <= set(out1["tables"]["framework_ablation"]["seed_split"])
    assert set(out1["tables"]["framework_ablation_summary"]["seed_split"]) == {"evaluation"}
    assert {"mean_rate", "median_rate", "ci95_low", "ci95_high", "n_jobs"} <= set(
        out1["tables"]["framework_ablation_summary"].columns
    )
    assert (tmp_path / "smoke_jobs.csv").exists()
    assert (tmp_path / "smoke_framework_ablation.csv").exists()
    assert (tmp_path / "smoke_framework_ablation_seed_summary.csv").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.csv").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.png").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.svg").exists()
    assert (tmp_path / "smoke_framework_ablation_supplemental.png").exists()
    assert (tmp_path / "smoke_framework_ablation_supplemental.svg").exists()
    assert len(out1["ablation_figure_paths"]) == 4
    assert list(tmp_path.glob("*_config.json"))
    assert list(tmp_path.glob("*_state_metrics.csv"))

    out2 = sg.bench.run_simulation_suite(
        profile="smoke",
        scenarios=["null_effect"],
        seeds={"calibration": [0], "evaluation": [1]},
        output_dir=tmp_path,
        resume=True,
        n_jobs=1,
    )
    assert set(out2["jobs"]["status"]) == {"resumed"}


def test_calibration_and_evaluation_seed_overlap_is_rejected():
    import scgeo as sg

    with pytest.raises(ValueError, match="disjoint"):
        sg.bench.run_simulation_suite(
            profile="smoke",
            scenarios=["null_effect"],
            seeds={"calibration": [0], "evaluation": [0]},
        )
