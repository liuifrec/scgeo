import json

import numpy as np
import pandas as pd
import pytest


SCENARIOS = [
    "null",
    "centroid_shift",
    "abundance_only",
    "covariance_only",
    "local_warp",
    "outlier_contamination",
    "unequal_cell_counts",
    "replicate_heterogeneity",
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
    reps = list(truth["equivalent_representations"]) + list(truth.get("corrupted_representations", []))
    velocity_keys = truth.get("velocity_keys")
    if velocity_keys is not None:
        velocity_keys = {rep: velocity_keys.get(rep) for rep in reps}
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
        reps=reps,
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
        reps=reps,
        node_key=truth["state_key"],
        sample_key=truth["sample_key"],
        k_values=(5,),
        n_boot=0,
        max_exact_cells=250,
        seed=truth["seed"],
    )
    return adata


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
        "corruption_detection",
        "distorted_state_detection",
        "neighborhood_discrimination",
        "abundance_truth",
        "summary_metrics",
        "coverage_summary",
        "runtime_summary",
    }
    assert expected <= set(out)
    assert {"scenario", "seed", "method", "state", "magnitude_error"} <= set(out["state_metrics"].columns)
    assert {"threshold", "precision", "recall", "f1"} <= set(out["threshold_sensitivity"].columns)
    assert {"state", "consensus_label", "status"} <= set(out["representation_consensus"].columns)
    assert {"state", "true_abundance_change", "true_abundance_changed"} <= set(out["abundance_truth"].columns)


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
    assert "representation_corruption" in set(ablation["failure_mode"])
    corr = ablation[
        (ablation["variant"] == "D_plus_local_geometry")
        & (ablation["failure_mode"] == "representation_corruption")
        & (ablation["target"] == "X_corrupted")
    ]
    assert not corr.empty
    assert "detects_correctly" in set(corr["outcome"])


def test_framework_ablation_plot_returns_figure():
    import matplotlib.pyplot as plt
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="aligned_dynamics", seed=29))
    ablation = sg.bench.framework_ablation(sg.bench.evaluate_ground_truth(adata))
    fig = sg.bench.plot_framework_ablation(ablation, show=False)
    assert fig.__class__.__name__ == "Figure"
    plt.close(fig)


def test_framework_ablation_plot_requires_requested_split():
    import scgeo as sg

    adata = _run_small_stack(_small_sim(scenario="null", seed=37))
    ablation = sg.bench.framework_ablation(sg.bench.evaluate_ground_truth(adata))
    ablation["seed_split"] = "calibration"

    with pytest.raises(ValueError, match="no rows to plot"):
        sg.bench.plot_framework_ablation(ablation, split="evaluation", show=False)


def test_null_abundance_and_covariance_do_not_force_centroid_shift_calls():
    import scgeo as sg

    for scenario in ["null", "abundance_only", "covariance_only"]:
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
        scenarios=["null"],
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
    assert {"calibration", "evaluation"} <= set(out1["tables"]["framework_ablation"]["seed_split"])
    assert set(out1["tables"]["framework_ablation_summary"]["seed_split"]) == {"evaluation"}
    assert (tmp_path / "smoke_jobs.csv").exists()
    assert (tmp_path / "smoke_framework_ablation.csv").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.csv").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.png").exists()
    assert (tmp_path / "smoke_framework_ablation_summary.svg").exists()
    assert len(out1["ablation_figure_paths"]) == 2
    assert list(tmp_path.glob("*_config.json"))
    assert list(tmp_path.glob("*_state_metrics.csv"))

    out2 = sg.bench.run_simulation_suite(
        profile="smoke",
        scenarios=["null"],
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
            scenarios=["null"],
            seeds={"calibration": [0], "evaluation": [0]},
        )
