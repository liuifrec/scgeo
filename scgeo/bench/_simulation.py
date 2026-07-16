from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
import json
import resource
import textwrap
import time

import anndata as ad
import numpy as np
import pandas as pd


_SCENARIOS = {
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
}
_SCENARIO_ALIASES = {
    "null": "null_effect",
    "no_perturbation": "null_effect",
    "replicate_heterogeneity": "balanced_replicate_heterogeneity",
}
_CONDITION0 = "control"
_CONDITION1 = "treated"
_STATE_KEY = "state"
_CONDITION_KEY = "condition"
_SAMPLE_KEY = "sample"
_PREDECLARED_SHIFT_DETECTION_THRESHOLD = 0.5
_SHIFT_SENSITIVITY_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25)
_ABLATION_VARIANTS = {
    "A_mean_shift_one_rep": "A. original mean shift on one representation",
    "B_robust_shift_one_rep": "B. robust shift on one representation",
    "C_robust_shift_representation_consensus": "C. robust shift plus representation consensus",
    "D_plus_local_geometry": "D. C plus local geometry diagnostics",
    "E_plus_dynamics": "E. D plus dynamics agreement",
}
_ABLATION_CONTEXT_COLUMNS = ("profile", "seed_split", "job_id", "scenario", "seed")
_MAGNITUDE_ZERO_EPSILON = 1e-8
_ABLATION_OUTCOME_ORDER = (
    "detects_correctly",
    "correctly_rejects",
    "misses",
    "falsely_calls",
    "insufficient_coverage",
    "representation_unstable",
    "non_identifiable",
    "unavailable",
    "not_applicable",
    "not_computed",
)
_PERFORMANCE_OUTCOMES = {"detects_correctly", "correctly_rejects", "misses", "falsely_calls"}
_SUPPORT_OUTCOMES = {"insufficient_coverage", "representation_unstable", "non_identifiable", "unavailable"}
_EVIDENCE_LAYER_SHORT_LABELS = {
    "centroid_shift": "shift",
    "representation_instability": "state\ninstab.",
    "representation_local_distortion": "local\ndist.",
    "representation_quality_outlier": "rep\nquality",
    "representation_distortion_detection": "rep\ndist.",
    "explicit_corruption_detection": "corrupt.",
    "dynamics_alignment": "dyn.",
    "bootstrap_interval_coverage": "boot.",
}
_VARIANT_SHORT_LABELS = {
    "A_mean_shift_one_rep": "A",
    "B_robust_shift_one_rep": "B",
    "C_robust_shift_representation_consensus": "C",
    "D_plus_local_geometry": "D",
    "E_plus_dynamics": "E",
}
_CAPABILITY_COLUMNS = (
    "Effect",
    "Uncertainty",
    "Representation stability",
    "Local geometry",
    "Dynamics",
)
_CAPABILITY_DISPLAY_LABELS = {
    "Effect": "Effect",
    "Uncertainty": "Uncertainty",
    "Representation stability": "Rep.\nstability",
    "Local geometry": "Local\ngeometry",
    "Dynamics": "Dynamics",
}
_CAPABILITY_BY_VARIANT = {
    "A_mean_shift_one_rep": ("Effect",),
    "B_robust_shift_one_rep": ("Effect", "Uncertainty"),
    "C_robust_shift_representation_consensus": (
        "Effect",
        "Uncertainty",
        "Representation stability",
    ),
    "D_plus_local_geometry": (
        "Effect",
        "Uncertainty",
        "Representation stability",
        "Local geometry",
    ),
    "E_plus_dynamics": (
        "Effect",
        "Uncertainty",
        "Representation stability",
        "Local geometry",
        "Dynamics",
    ),
}
_MAIN_ABLATION_ROW_SPECS = (
    ("null_effect", "centroid_shift", "Null effect | Shift"),
    ("centroid_shift", "centroid_shift", "Centroid shift | Shift"),
    ("outlier_contamination", "centroid_shift", "Outliers | Shift"),
    ("unequal_cell_counts", "centroid_shift", "Unequal counts | Shift"),
    ("balanced_replicate_heterogeneity", "centroid_shift", "Replicate het. | Shift"),
    ("representation_corruption", "explicit_corruption_detection", "Explicit corrupt. | Corruption"),
    ("representation_corruption", "representation_distortion_detection", "Rep. distortion | Distortion"),
    ("aligned_dynamics", "dynamics_alignment", "Aligned dyn. | Dynamics"),
    ("discordant_dynamics", "dynamics_alignment", "Discordant dyn. | Dynamics"),
)
_MAIN_ABLATION_LABELS = {
    (scenario, failure_mode): label
    for scenario, failure_mode, label in _MAIN_ABLATION_ROW_SPECS
}
_SUPPORT_STATUS_ORDER = (
    "representation_unstable",
    "insufficient_coverage",
    "non_identifiable",
    "unavailable",
)
_DYNAMICS_SCENARIOS = {"aligned_dynamics", "discordant_dynamics"}
_PLOT_AUTO_ROW_HEIGHT = 0.30
_PLOT_AUTO_MIN_HEIGHT = 3.2
_PLOT_AUTO_MIN_PANEL_WIDTHS = {
    "capability": 2.85,
    "performance": 2.70,
    "support": 1.90,
}
_PLOT_AUTO_MAX_PANEL_WIDTHS = {
    "capability": 3.25,
    "performance": 4.80,
    "support": 3.30,
}


def _check_scenario(scenario: str) -> str:
    scenario = _SCENARIO_ALIASES.get(str(scenario), str(scenario))
    if scenario not in _SCENARIOS:
        allowed = ", ".join(sorted(_SCENARIOS))
        raise ValueError(f"scenario must be one of {{{allowed}}}, got {scenario!r}")
    return scenario


def _state_names(n_states: int) -> list[str]:
    if int(n_states) < 1:
        raise ValueError("n_states must be >= 1")
    return [f"state_{i}" for i in range(int(n_states))]


def _affected_states(
    states: list[str],
    affected_states: Optional[Sequence[str]],
    scenario: str,
) -> list[str]:
    if affected_states is not None:
        out = [str(x) for x in affected_states]
        missing = sorted(set(out) - set(states))
        if missing:
            raise ValueError(f"affected_states contains states not present in simulation: {missing}")
        return out
    if scenario in {"null_effect", "batch_condition_confounding"}:
        return []
    n = max(1, len(states) // 2)
    return states[:n]


def _orthogonal_matrix(rng: np.random.RandomState, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def _base_centers(n_states: int, latent_dim: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n_states)
    centers = np.zeros((n_states, latent_dim), dtype=float)
    centers[:, 0] = 4.0 * t
    if latent_dim > 1:
        centers[:, 1] = 2.0 * np.sin(np.pi * (t + 1.0) / 2.0)
    if latent_dim > 2:
        centers[:, 2] = 1.5 * np.cos(np.pi * (t + 1.0) / 2.0)
    if latent_dim > 3:
        centers[:, 3] = np.where(t >= 0, 1.0, -1.0) * (0.5 + np.abs(t))
    return centers


def _state_directions(n_states: int, latent_dim: int) -> np.ndarray:
    dirs = np.zeros((n_states, latent_dim), dtype=float)
    for i in range(n_states):
        dirs[i, i % latent_dim] = 1.0
        if latent_dim > 1:
            dirs[i, (i + 1) % latent_dim] = 0.35
        norm = np.linalg.norm(dirs[i])
        dirs[i] = dirs[i] / max(norm, 1e-12)
    return dirs


def _state_probabilities(
    states: list[str],
    condition: str,
    affected: set[str],
    abundance_effect: float,
    scenario: str,
) -> np.ndarray:
    n_states = len(states)
    base = np.asarray([1.0 + 0.15 * np.sin(i) for i in range(n_states)], dtype=float)
    if condition == _CONDITION1 and (scenario in {"abundance_only", "unequal_cell_counts"} or abundance_effect != 0.0):
        effect = abundance_effect if abundance_effect != 0.0 else 0.8
        for i, state in enumerate(states):
            if state in affected:
                base[i] *= max(0.05, 1.0 + effect)
            else:
                base[i] *= max(0.05, 1.0 - effect / max(1, n_states - len(affected)))
    probs = base / base.sum()
    return probs


def _scenario_values(
    *,
    scenario: str,
    effect_size: float,
    outlier_fraction: float,
    sample_heterogeneity: float,
    covariance_effect: float,
    warp_strength: float,
) -> tuple[float, float, float, float, float]:
    if scenario == "null_effect":
        effect_size = 0.0
    if scenario in {"abundance_only", "covariance_only", "local_warp", "batch_condition_confounding"}:
        effect_size = 0.0
    if scenario == "outlier_contamination" and outlier_fraction == 0.0:
        outlier_fraction = 0.08
    if scenario in {"balanced_replicate_heterogeneity", "batch_condition_confounding"}:
        sample_heterogeneity = max(float(sample_heterogeneity), 0.6)
    if scenario == "covariance_only" and covariance_effect == 0.0:
        covariance_effect = 0.9
    if scenario == "local_warp" and warp_strength == 0.0:
        warp_strength = 0.7
    return (
        float(effect_size),
        float(outlier_fraction),
        float(sample_heterogeneity),
        float(covariance_effect),
        float(warp_strength),
    )


def _sample_offset_plan(
    rng: np.random.RandomState,
    *,
    scenario: str,
    n_samples_per_condition: int,
    latent_dim: int,
    sample_heterogeneity: float,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    n_samples = int(n_samples_per_condition)
    dim = int(latent_dim)
    sigma = float(sample_heterogeneity)
    offsets: dict[tuple[str, int], np.ndarray] = {}
    residual_scale = sigma
    systematic_by_condition = {
        _CONDITION0: np.zeros(dim, dtype=float),
        _CONDITION1: np.zeros(dim, dtype=float),
    }
    design = {
        "sample_offset_design": "independent_unpaired_random",
        "sample_offset_formula": "offset[c,s] = eps[c,s], eps[c,s] ~ Normal(0, sigma^2 I)",
        "sample_heterogeneity_sigma": sigma,
        "samples_paired": False,
        "zero_centered_within_condition": False,
        "systematic_condition_confounding": False,
        "non_identifiable": False,
        "expected_biological_shift_identifiable": True,
        "expected_outcome": "ordinary_shift_detection",
    }

    if scenario == "balanced_replicate_heterogeneity":
        design.update(
            {
                "sample_offset_design": "condition_centered_random_replicate_offsets",
                "sample_offset_formula": (
                    "raw_offset[c,s] ~ Normal(0, sigma^2 I); "
                    "offset[c,s] = raw_offset[c,s] - mean_s raw_offset[c,s] within each condition"
                ),
                "zero_centered_within_condition": True,
                "expected_outcome": "wider_uncertainty_with_low_neutral_false_call_rate",
            }
        )
        for condition in (_CONDITION0, _CONDITION1):
            raw = rng.normal(scale=sigma, size=(n_samples, dim))
            centered = raw - raw.mean(axis=0, keepdims=True)
            for sample_idx in range(n_samples):
                offsets[(condition, sample_idx)] = centered[sample_idx].astype(float)
        return offsets, design

    if scenario == "batch_condition_confounding":
        direction = np.zeros(dim, dtype=float)
        direction[0] = 1.0
        if dim > 1:
            direction[1] = 0.35
        if dim > 2:
            direction[2] = -0.2
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        confound = sigma * np.sqrt(max(1, dim)) * direction
        systematic_by_condition = {
            _CONDITION0: -0.5 * confound,
            _CONDITION1: 0.5 * confound,
        }
        residual_scale = 0.10 * sigma
        design.update(
            {
                "sample_offset_design": "condition_correlated_batch_offset",
                "sample_offset_formula": (
                    "offset[control,s] = -0.5 * b + eps[c,s]; "
                    "offset[treated,s] = 0.5 * b + eps[c,s]; "
                    "eps is centered within condition and b is a systematic batch offset"
                ),
                "zero_centered_within_condition": False,
                "systematic_condition_confounding": True,
                "non_identifiable": True,
                "expected_biological_shift_identifiable": False,
                "expected_outcome": "non_identifiable_without_additional_design_information",
                "systematic_offset_norm": float(np.linalg.norm(confound)),
                "systematic_condition_offsets": {
                    condition: vector.astype(float).tolist()
                    for condition, vector in systematic_by_condition.items()
                },
            }
        )
        for condition in (_CONDITION0, _CONDITION1):
            residual = rng.normal(scale=residual_scale, size=(n_samples, dim))
            residual = residual - residual.mean(axis=0, keepdims=True)
            planned = residual + systematic_by_condition[condition][None, :]
            for sample_idx in range(n_samples):
                offsets[(condition, sample_idx)] = planned[sample_idx].astype(float)
        return offsets, design

    for condition in (_CONDITION0, _CONDITION1):
        planned = rng.normal(scale=residual_scale, size=(n_samples, dim))
        for sample_idx in range(n_samples):
            offsets[(condition, sample_idx)] = planned[sample_idx].astype(float)
    return offsets, design


def _generate_block(
    rng: np.random.RandomState,
    *,
    n: int,
    state_idx: int,
    state: str,
    condition: str,
    sample_shift: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    affected: set[str],
    shifted: set[str],
    distorted: set[str],
    scenario: str,
    effect_size: float,
    covariance_effect: float,
    warp_strength: float,
    outlier_fraction: float,
    outlier_scale: float,
    sample_effect_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    latent_dim = centers.shape[1]
    mode_axis = np.zeros(latent_dim, dtype=float)
    mode_axis[(state_idx + 2) % latent_dim] = 0.35 + 0.05 * (state_idx % 3)
    modes = rng.choice([-1.0, 1.0], size=n)
    center = centers[state_idx]
    scales = 0.13 + 0.035 * ((state_idx + np.arange(latent_dim)) % 4)
    if condition == _CONDITION1 and state in distorted and covariance_effect > 0.0:
        scales[: min(3, latent_dim)] *= 1.0 + covariance_effect
    X = (
        center
        + modes[:, None] * mode_axis[None, :]
        + rng.normal(scale=scales, size=(n, latent_dim))
        + sample_shift[None, :]
    )
    delta = np.zeros(latent_dim, dtype=float)
    if condition == _CONDITION1 and state in shifted:
        delta = effect_size * sample_effect_scale * directions[state_idx]
        X = X + delta[None, :]
    if condition == _CONDITION1 and state in distorted and warp_strength > 0.0:
        local_x = X[:, 0] - center[0]
        if latent_dim > 1:
            X[:, 1] += warp_strength * (local_x**2 - np.mean(local_x**2))
        if latent_dim > 2:
            X[:, 2] += 0.35 * warp_strength * np.sin(local_x)
    if condition == _CONDITION1 and scenario == "outlier_contamination" and state in affected:
        n_out = int(np.floor(n * outlier_fraction))
        if n_out > 0:
            idx = rng.choice(n, size=n_out, replace=False)
            X[idx] += outlier_scale * directions[state_idx][None, :]
    return X, delta


def _nonlinear_transform(X: np.ndarray, strength: float = 0.25) -> np.ndarray:
    out = X.copy()
    if out.shape[1] > 1:
        out[:, 1] = out[:, 1] + strength * np.sin(out[:, 0])
    if out.shape[1] > 2:
        out[:, 2] = out[:, 2] + 0.15 * strength * out[:, 0] ** 2
    return out


def _nonlinear_velocity(X: np.ndarray, V: np.ndarray, strength: float = 0.25) -> np.ndarray:
    out = V.copy()
    if out.shape[1] > 1:
        out[:, 1] = out[:, 1] + strength * np.cos(X[:, 0]) * V[:, 0]
    if out.shape[1] > 2:
        out[:, 2] = out[:, 2] + 0.3 * strength * X[:, 0] * V[:, 0]
    return out


def simulate_perturbation_geometry(
    *,
    scenario: str = "centroid_shift",
    n_states: int = 5,
    n_samples_per_condition: int = 4,
    cells_per_sample: int = 400,
    latent_dim: int = 8,
    effect_size: float = 1.0,
    affected_states=None,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 10.0,
    sample_heterogeneity: float = 0.15,
    abundance_effect: float = 0.0,
    covariance_effect: float = 0.0,
    warp_strength: float = 0.0,
    velocity_mode=None,
    seed: int = 0,
) -> ad.AnnData:
    """
    Generate a synthetic perturbation-geometry benchmark AnnData object.

    The simulation is synthetic only. It stores known truth under
    ``adata.uns['simulation_truth']`` and creates multiple latent
    representations without requiring Scanpy, scVelo, or GPU libraries.
    """
    scenario = _check_scenario(scenario)
    if int(n_samples_per_condition) < 1:
        raise ValueError("n_samples_per_condition must be >= 1")
    if int(cells_per_sample) < 1:
        raise ValueError("cells_per_sample must be >= 1")
    if int(latent_dim) < 2:
        raise ValueError("latent_dim must be >= 2")

    rng = np.random.RandomState(seed)
    states = _state_names(int(n_states))
    affected = set(_affected_states(states, affected_states, scenario))
    effect_size, outlier_fraction, sample_heterogeneity, covariance_effect, warp_strength = _scenario_values(
        scenario=scenario,
        effect_size=effect_size,
        outlier_fraction=outlier_fraction,
        sample_heterogeneity=sample_heterogeneity,
        covariance_effect=covariance_effect,
        warp_strength=warp_strength,
    )
    shifted = set()
    if scenario in {
        "centroid_shift",
        "outlier_contamination",
        "unequal_cell_counts",
        "balanced_replicate_heterogeneity",
        "representation_corruption",
        "aligned_dynamics",
        "discordant_dynamics",
    }:
        shifted = set(affected)
    distorted = set(affected) if scenario in {"covariance_only", "local_warp"} or covariance_effect > 0.0 or warp_strength > 0.0 else set()

    centers = _base_centers(len(states), int(latent_dim))
    directions = _state_directions(len(states), int(latent_dim))
    rows: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    obs_rows: list[dict[str, Any]] = []
    sample_offsets: list[dict[str, Any]] = []
    sample_offset_by_key, replicate_design = _sample_offset_plan(
        rng,
        scenario=scenario,
        n_samples_per_condition=int(n_samples_per_condition),
        latent_dim=int(latent_dim),
        sample_heterogeneity=sample_heterogeneity,
    )
    true_delta_by_state = {state: np.zeros(int(latent_dim), dtype=float) for state in states}
    state_counts = {(condition, state): 0 for condition in (_CONDITION0, _CONDITION1) for state in states}
    dynamics_mode = velocity_mode
    if dynamics_mode is None and scenario == "aligned_dynamics":
        dynamics_mode = "aligned"
    if dynamics_mode is None and scenario == "discordant_dynamics":
        dynamics_mode = "discordant"

    for condition in (_CONDITION0, _CONDITION1):
        probs = _state_probabilities(states, condition, affected, abundance_effect, scenario)
        for sample_idx in range(int(n_samples_per_condition)):
            sample_name = f"{condition}_sample_{sample_idx}"
            n_sample = int(cells_per_sample)
            if scenario == "unequal_cell_counts" and condition == _CONDITION1:
                n_sample = max(1, int(round(cells_per_sample * (1.0 + max(0.4, abs(abundance_effect))))))
            counts = rng.multinomial(n_sample, probs)
            sample_shift = sample_offset_by_key[(condition, sample_idx)]
            sample_offsets.append(
                {
                    "sample": sample_name,
                    "condition": condition,
                    "sample_index": int(sample_idx),
                    "offset_norm": float(np.linalg.norm(sample_shift)),
                    **{f"offset_dim_{i}": float(sample_shift[i]) for i in range(int(latent_dim))},
                }
            )
            sample_effect_scale = 1.0
            if scenario == "balanced_replicate_heterogeneity":
                sample_effect_scale = max(0.0, rng.normal(loc=1.0, scale=0.45))
            for state_idx, (state, n_cells) in enumerate(zip(states, counts)):
                if n_cells == 0:
                    continue
                X_state, delta = _generate_block(
                    rng,
                    n=int(n_cells),
                    state_idx=state_idx,
                    state=state,
                    condition=condition,
                    sample_shift=sample_shift,
                    centers=centers,
                    directions=directions,
                    affected=affected,
                    shifted=shifted,
                    distorted=distorted,
                    scenario=scenario,
                    effect_size=effect_size,
                    covariance_effect=covariance_effect,
                    warp_strength=warp_strength,
                    outlier_fraction=outlier_fraction,
                    outlier_scale=float(outlier_scale),
                    sample_effect_scale=sample_effect_scale,
                )
                rows.append(X_state)
                if condition == _CONDITION1 and state in shifted:
                    true_delta_by_state[state] = effect_size * directions[state_idx]
                if dynamics_mode == "aligned" and state in shifted:
                    V_state = np.repeat(true_delta_by_state[state][None, :], int(n_cells), axis=0)
                elif dynamics_mode == "discordant" and state in shifted:
                    V_state = np.repeat((-true_delta_by_state[state])[None, :], int(n_cells), axis=0)
                else:
                    V_state = np.zeros((int(n_cells), int(latent_dim)), dtype=float)
                if dynamics_mode is not None:
                    V_state = V_state + rng.normal(scale=0.02, size=V_state.shape)
                velocities.append(V_state)
                state_counts[(condition, state)] += int(n_cells)
                for _ in range(int(n_cells)):
                    obs_rows.append({"state": state, "condition": condition, "sample": sample_name})

    X = np.vstack(rows).astype(np.float32)
    V = np.vstack(velocities).astype(np.float32)
    obs = pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32), obs=obs)

    q = _orthogonal_matrix(rng, int(latent_dim))
    scale = 2.5
    anisotropic_scale = np.linspace(0.45, 2.0, int(latent_dim))
    nonlinear_strength = 0.25 + 0.25 * max(0.0, warp_strength)

    adata.obsm["X_truth"] = X
    adata.obsm["X_rotated"] = (X @ q).astype(np.float32)
    adata.obsm["X_scaled"] = (scale * X).astype(np.float32)
    adata.obsm["X_padded"] = np.column_stack([X, np.zeros((X.shape[0], 2), dtype=np.float32)]).astype(np.float32)
    adata.obsm["X_anisotropic"] = (X * anisotropic_scale[None, :]).astype(np.float32)
    adata.obsm["X_nonlinear"] = _nonlinear_transform(X, nonlinear_strength).astype(np.float32)
    corrupted_reps: list[str] = []
    if scenario == "representation_corruption":
        perm = rng.permutation(X.shape[0])
        adata.obsm["X_corrupted"] = (X[perm] + rng.normal(scale=0.1, size=X.shape)).astype(np.float32)
        corrupted_reps.append("X_corrupted")

    velocity_keys = None
    dynamics_class = {state: "neutral" for state in states}
    if dynamics_mode is not None:
        velocity_keys = {
            "X_truth": "V_truth",
            "X_rotated": "V_rotated",
            "X_scaled": "V_scaled",
            "X_padded": "V_padded",
            "X_anisotropic": "V_anisotropic",
            "X_nonlinear": "V_nonlinear",
        }
        adata.obsm["V_truth"] = V
        adata.obsm["V_rotated"] = (V @ q).astype(np.float32)
        adata.obsm["V_scaled"] = (scale * V).astype(np.float32)
        adata.obsm["V_padded"] = np.column_stack([V, np.zeros((V.shape[0], 2), dtype=np.float32)]).astype(np.float32)
        adata.obsm["V_anisotropic"] = (V * anisotropic_scale[None, :]).astype(np.float32)
        adata.obsm["V_nonlinear"] = _nonlinear_velocity(X, V, nonlinear_strength).astype(np.float32)
        if "X_corrupted" in adata.obsm:
            velocity_keys["X_corrupted"] = None
        for state in shifted:
            dynamics_class[state] = "aligned" if dynamics_mode == "aligned" else "discordant"

    true_magnitude = {state: float(np.linalg.norm(delta)) for state, delta in true_delta_by_state.items()}
    counts_control = np.asarray([state_counts[(_CONDITION0, state)] for state in states], dtype=float)
    counts_treated = np.asarray([state_counts[(_CONDITION1, state)] for state in states], dtype=float)
    freq_control = counts_control / max(1.0, counts_control.sum())
    freq_treated = counts_treated / max(1.0, counts_treated.sum())
    abundance_changes = {
        state: float(freq_treated[i] - freq_control[i])
        for i, state in enumerate(states)
    }
    equivalent_reps = ["X_truth", "X_rotated", "X_scaled", "X_padded"]
    diagnostic_distorted_reps = ["X_anisotropic", "X_nonlinear"]
    reps = equivalent_reps + diagnostic_distorted_reps + corrupted_reps
    consensus_reps = equivalent_reps + corrupted_reps
    non_identifiable_states = states if bool(replicate_design.get("non_identifiable", False)) else []
    adata.uns["simulation_truth"] = {
        "scenario": scenario,
        "seed": int(seed),
        "params": {
            "scenario": scenario,
            "n_states": int(n_states),
            "n_samples_per_condition": int(n_samples_per_condition),
            "cells_per_sample": int(cells_per_sample),
            "latent_dim": int(latent_dim),
            "effect_size": float(effect_size),
            "affected_states": sorted(affected),
            "outlier_fraction": float(outlier_fraction),
            "outlier_scale": float(outlier_scale),
            "sample_heterogeneity": float(sample_heterogeneity),
            "abundance_effect": float(abundance_effect),
            "covariance_effect": float(covariance_effect),
            "warp_strength": float(warp_strength),
            "velocity_mode": dynamics_mode,
            "seed": int(seed),
        },
        "state_key": _STATE_KEY,
        "condition_key": _CONDITION_KEY,
        "sample_key": _SAMPLE_KEY,
        "group0": _CONDITION0,
        "group1": _CONDITION1,
        "states": states,
        "affected_states": sorted(affected),
        "shifted_states": sorted(shifted),
        "non_identifiable_states": list(non_identifiable_states),
        "non_identifiable_reason": (
            "condition-correlated sample/batch offsets make biological condition shifts non-identifiable"
            if non_identifiable_states
            else ""
        ),
        "true_effect_magnitude": true_magnitude,
        "true_delta": {state: true_delta_by_state[state].astype(float).tolist() for state in states},
        "true_abundance_change": abundance_changes,
        "distorted_states": sorted(distorted),
        "condition_shape_changed_states": sorted(distorted),
        "corrupted_representations": corrupted_reps,
        "explicitly_corrupted_representations": corrupted_reps,
        "dynamics_class": dynamics_class,
        "representations": reps,
        "equivalent_representations": equivalent_reps,
        "consensus_representations": consensus_reps,
        "diagnostic_distorted_representations": diagnostic_distorted_reps,
        "diagnostically_distorted_representations": diagnostic_distorted_reps,
        "distorted_representations": diagnostic_distorted_reps,
        "representation_categories": {
            **{rep: "equivalent" for rep in equivalent_reps},
            **{rep: "diagnostically_distorted" for rep in diagnostic_distorted_reps},
            **{rep: "explicitly_corrupted" for rep in corrupted_reps},
        },
        "representation_distortion_affected_states": (
            sorted(affected) if scenario == "representation_corruption" else []
        ),
        "replicate_design": {
            **replicate_design,
            "non_identifiable_states": list(non_identifiable_states),
        },
        "sample_offsets": sample_offsets,
        "velocity_keys": velocity_keys,
    }
    return adata


def _require_truth(adata) -> dict[str, Any]:
    truth = adata.uns.get("simulation_truth")
    if not isinstance(truth, dict):
        raise KeyError("adata.uns['simulation_truth'] is required for ground-truth evaluation")
    return truth


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _detection_threshold() -> float:
    return float(_PREDECLARED_SHIFT_DETECTION_THRESHOLD)


def _robust_store_to_rows(
    store: dict[str, Any],
    *,
    method: str,
    truth: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    true_mag = truth["true_effect_magnitude"]
    shifted = set(truth["shifted_states"])
    non_identifiable = set(truth.get("non_identifiable_states", []))
    rows = []
    for state in truth["states"]:
        by = store.get("by", {}) if isinstance(store, dict) else {}
        rec = by.get(state, {}) if isinstance(by, dict) else {}
        est = float(rec.get("delta_norm", np.nan))
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "method": method,
                "state": state,
                "estimated_magnitude": est,
                "true_magnitude": float(true_mag[state]),
                "magnitude_error": abs(est - float(true_mag[state])) if np.isfinite(est) else np.nan,
                "true_shifted": state in shifted,
                "predicted_shifted": bool(np.isfinite(est) and est >= threshold),
                "non_identifiable": state in non_identifiable,
                "biological_shift_evaluable": state not in non_identifiable,
                "threshold": threshold,
            }
        )
    return rows


def _shift_store_to_rows(
    store: dict[str, Any],
    *,
    truth: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    true_mag = truth["true_effect_magnitude"]
    shifted = set(truth["shifted_states"])
    non_identifiable = set(truth.get("non_identifiable_states", []))
    out = store.get("shift", store)
    by = out.get("by", {}) if isinstance(out, dict) else {}
    rows = []
    for state in truth["states"]:
        rec = by.get(state, {}) if isinstance(by, dict) else {}
        est = float(rec.get("delta_norm", np.nan))
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "method": "shift_mean",
                "state": state,
                "estimated_magnitude": est,
                "true_magnitude": float(true_mag[state]),
                "magnitude_error": abs(est - float(true_mag[state])) if np.isfinite(est) else np.nan,
                "true_shifted": state in shifted,
                "predicted_shifted": bool(np.isfinite(est) and est >= threshold),
                "non_identifiable": state in non_identifiable,
                "biological_shift_evaluable": state not in non_identifiable,
                "threshold": threshold,
            }
        )
    return rows


def _run_baseline_estimators(adata, truth: dict[str, Any], threshold: float) -> pd.DataFrame:
    import scgeo as sg

    work = adata.copy()
    rows: list[dict[str, Any]] = []
    sg.tl.shift(
        work,
        rep="X_truth",
        condition_key=truth["condition_key"],
        group0=truth["group0"],
        group1=truth["group1"],
        by=truth["state_key"],
        sample_key=truth["sample_key"],
        store_key="_bench_shift_mean",
    )
    rows.extend(_shift_store_to_rows(work.uns["_bench_shift_mean"], truth=truth, threshold=threshold))

    for center in ("mean", "median", "trimmed_mean", "geometric_median"):
        key = f"_bench_robust_{center}"
        sg.tl.robust_shift(
            work,
            rep="X_truth",
            condition_key=truth["condition_key"],
            group0=truth["group0"],
            group1=truth["group1"],
            by=truth["state_key"],
            sample_key=truth["sample_key"],
            center=center,
            trim_fraction=0.1,
            n_boot=0,
            seed=int(truth["seed"]),
            store_key=key,
        )
        rows.extend(_robust_store_to_rows(work.uns["scgeo"][key], method=f"robust_{center}", truth=truth, threshold=threshold))
    return pd.DataFrame(rows)


def _rank_metrics(state_metrics: pd.DataFrame, truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, df in state_metrics.groupby("method", sort=False):
        pair = df[["estimated_magnitude", "true_magnitude"]].dropna()
        if pair.shape[0] >= 2 and pair["estimated_magnitude"].nunique() > 1 and pair["true_magnitude"].nunique() > 1:
            rho = pair["estimated_magnitude"].corr(pair["true_magnitude"], method="spearman")
        else:
            rho = np.nan
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "method": method,
                "state_rank_spearman": float(rho) if rho is not None else np.nan,
                "n_states": int(pair.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _shift_classification_metrics(state_metrics: pd.DataFrame, truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, df in state_metrics.groupby("method", sort=False):
        if "biological_shift_evaluable" in df:
            evaluable = df[df["biological_shift_evaluable"].astype(bool)].copy()
        else:
            evaluable = df.copy()
        cls = _classification_metrics(
            evaluable["true_shifted"].to_numpy(),
            evaluable["predicted_shifted"].to_numpy(),
        )
        null = evaluable[~evaluable["true_shifted"]]
        false_rate = float(null["predicted_shifted"].mean()) if not null.empty else np.nan
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "shifted_state_detection",
                "method": method,
                **cls,
                "null_state_false_classification_rate": false_rate,
                "n_evaluable_states": int(evaluable.shape[0]),
                "n_non_identifiable_states": int(df.shape[0] - evaluable.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _threshold_sensitivity(state_metrics: pd.DataFrame, truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, df in state_metrics.groupby("method", sort=False):
        if "biological_shift_evaluable" in df:
            df = df[df["biological_shift_evaluable"].astype(bool)].copy()
        est = df["estimated_magnitude"].to_numpy(dtype=float)
        true_shift = df["true_shifted"].to_numpy(dtype=bool)
        for thr in _SHIFT_SENSITIVITY_THRESHOLDS:
            cls = _classification_metrics(true_shift, np.isfinite(est) & (est >= thr))
            rows.append(
                {
                    "scenario": truth["scenario"],
                    "seed": int(truth["seed"]),
                    "method": method,
                    "threshold": float(thr),
                    **cls,
                    "n_evaluable_states": int(df.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_coverage(adata, truth: dict[str, Any], robust_shift_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(robust_shift_key)
    rows = []
    if not isinstance(store, dict) or not isinstance(store.get("by"), dict):
        return pd.DataFrame(rows)
    for state in truth["states"]:
        rec = store["by"].get(state, {})
        ci = rec.get("bootstrap_magnitude_ci95", [np.nan, np.nan]) if isinstance(rec, dict) else [np.nan, np.nan]
        low = float(ci[0]) if ci is not None and len(ci) == 2 else np.nan
        high = float(ci[1]) if ci is not None and len(ci) == 2 else np.nan
        true_mag = float(truth["true_effect_magnitude"][state])
        applicable = bool(true_mag > _MAGNITUDE_ZERO_EPSILON)
        interval_available = bool(np.isfinite(low) and np.isfinite(high))
        if not applicable:
            status = "not_applicable"
            covered = np.nan
        elif interval_available:
            status = "assessed"
            covered = bool(low <= true_mag <= high)
        else:
            status = "unavailable"
            covered = np.nan
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "method": robust_shift_key,
                "true_magnitude": true_mag,
                "zero_effect_epsilon": float(_MAGNITUDE_ZERO_EPSILON),
                "ci95_low": low,
                "ci95_high": high,
                "interval_kind": "bootstrap_uncertainty_interval",
                "coverage_applicable": applicable,
                "coverage_status": status,
                "covered": covered,
            }
        )
    return pd.DataFrame(rows)


def _class_from_consensus(label: Any) -> str:
    label = str(label)
    if label == "stable_aligned":
        return "aligned"
    if label == "stable_discordant":
        return "discordant"
    if label == "stable_neutral":
        return "neutral"
    return "unavailable"


def _consensus_status(label: Any) -> str:
    if pd.isna(label):
        return "unavailable"
    label = str(label)
    if label == "insufficient_coverage":
        return "insufficient_coverage"
    if label == "representation_unstable":
        return "representation_unstable"
    if label.startswith("stable_"):
        return "assessed"
    return "unavailable"


def _representation_consensus(adata, truth: dict[str, Any], representation_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(representation_key)
    consensus = store.get("consensus_state") if isinstance(store, dict) else None
    rows = []
    if not isinstance(consensus, pd.DataFrame) or consensus.empty:
        return pd.DataFrame(rows)
    velocity_requested = truth.get("velocity_keys") is not None
    for rec in consensus.to_dict(orient="records"):
        label = rec.get("consensus_label")
        state = str(rec.get("node"))
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "consensus_label": label,
                "status": _consensus_status(label),
                "n_usable_representations": int(rec.get("n_usable_representations", 0)),
                "usable_fraction": float(rec.get("usable_fraction", np.nan)),
                "normalized_magnitude_median": float(rec.get("normalized_magnitude_median", np.nan)),
                "normalized_magnitude_max": float(rec.get("normalized_magnitude_max", np.nan)),
                "delta_norm_median": float(rec.get("delta_norm_median", np.nan)),
                "magnitude_rank_mean": float(rec.get("magnitude_rank_mean", np.nan)),
                "magnitude_rank_std": float(rec.get("magnitude_rank_std", np.nan)),
                "pairwise_spearman_median": float(rec.get("pairwise_spearman_median", np.nan)),
                "loo_rep_magnitude_max_relative_deviation": float(
                    rec.get("loo_rep_magnitude_max_relative_deviation", np.nan)
                ),
                "aligned_fraction": float(rec.get("aligned_fraction", np.nan)),
                "discordant_fraction": float(rec.get("discordant_fraction", np.nan)),
                "neutral_fraction": float(rec.get("neutral_fraction", np.nan)),
                "alignment_cosine_median": float(rec.get("alignment_cosine_median", np.nan)),
                "velocity_requested": bool(velocity_requested),
            }
        )
    return pd.DataFrame(rows)


def _alignment_accuracy(adata, truth: dict[str, Any], representation_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(representation_key)
    rows = []
    if not isinstance(store, dict):
        return pd.DataFrame(rows)
    velocity_requested = truth.get("velocity_keys") is not None
    consensus = store.get("consensus_state")
    if isinstance(consensus, pd.DataFrame) and not consensus.empty:
        for rec in consensus.to_dict(orient="records"):
            state = str(rec.get("node"))
            true_class = truth["dynamics_class"].get(state, "neutral")
            pred = _class_from_consensus(rec.get("consensus_label")) if velocity_requested else "unavailable"
            rows.append(
                {
                    "scenario": truth["scenario"],
                    "seed": int(truth["seed"]),
                    "source": "consensus",
                    "rep": None,
                    "state": state,
                    "true_class": true_class,
                    "predicted_class": pred,
                    "correct": bool(pred == true_class),
                    "velocity_requested": bool(velocity_requested),
                }
            )
    per = store.get("per_rep_state")
    if isinstance(per, pd.DataFrame) and not per.empty:
        for rec in per.to_dict(orient="records"):
            state = str(rec.get("node"))
            pred = str(rec.get("alignment_class", "missing"))
            if pred == "missing":
                pred = "unavailable"
            true_class = truth["dynamics_class"].get(state, "neutral")
            rows.append(
                {
                    "scenario": truth["scenario"],
                    "seed": int(truth["seed"]),
                    "source": "single_representation",
                    "rep": rec.get("rep"),
                    "state": state,
                    "true_class": true_class,
                    "predicted_class": pred,
                    "correct": bool(pred == true_class),
                    "velocity_requested": bool(velocity_requested),
                }
            )
    return pd.DataFrame(rows)


def _representation_category(truth: dict[str, Any], rep: str) -> str:
    categories = truth.get("representation_categories", {})
    if isinstance(categories, Mapping) and rep in categories:
        return str(categories[rep])
    if rep in set(truth.get("equivalent_representations", [])):
        return "equivalent"
    if rep in set(truth.get("diagnostically_distorted_representations", truth.get("diagnostic_distorted_representations", []))):
        return "diagnostically_distorted"
    if rep in set(truth.get("explicitly_corrupted_representations", truth.get("corrupted_representations", []))):
        return "explicitly_corrupted"
    return "unknown"


def _representation_quality_tables(adata, truth: dict[str, Any], local_geometry_key: str) -> dict[str, pd.DataFrame]:
    store = adata.uns.get("scgeo", {}).get(local_geometry_key)
    rep_summary = store.get("representation_summary") if isinstance(store, dict) else None
    quality_rows = []
    distortion_rows = []
    corruption_rows = []
    if not isinstance(rep_summary, pd.DataFrame) or rep_summary.empty:
        empty = pd.DataFrame()
        return {
            "representation_quality_outlier": empty,
            "representation_distortion_detection": empty,
            "explicit_corruption_detection": empty,
            "corruption_detection": empty,
        }
    for rec in rep_summary.to_dict(orient="records"):
        rep = str(rec.get("rep"))
        category = _representation_category(truth, rep)
        insufficient = bool(rec.get("insufficient_coverage", False))
        quality_call = any(
            bool(rec.get(col, False))
            for col in ("neighborhood_outlier", "distortion_outlier", "state_graph_outlier")
        )
        base = {
            "scenario": truth["scenario"],
            "seed": int(truth["seed"]),
            "rep": rep,
            "representation_category": category,
            "neighborhood_outlier": bool(rec.get("neighborhood_outlier", False)),
            "distortion_outlier": bool(rec.get("distortion_outlier", False)),
            "state_graph_outlier": bool(rec.get("state_graph_outlier", False)),
            "insufficient_coverage": insufficient,
        }
        status = "insufficient_coverage" if insufficient else "assessed"
        quality_rows.append(
            {
                **base,
                "true_quality_outlier": category in {"diagnostically_distorted", "explicitly_corrupted"},
                "predicted_quality_outlier": quality_call,
                "status": status,
            }
        )
        distortion_status = "not_applicable" if category == "explicitly_corrupted" else status
        distortion_rows.append(
            {
                **base,
                "true_representation_distorted": category == "diagnostically_distorted",
                "predicted_representation_distorted": (
                    bool(rec.get("distortion_outlier", False))
                    or bool(rec.get("neighborhood_outlier", False))
                    or bool(rec.get("state_graph_outlier", False))
                ),
                "status": distortion_status,
            }
        )
        corruption_status = "not_applicable" if category == "diagnostically_distorted" else status
        corruption_rows.append(
            {
                **base,
                "true_corrupted": category == "explicitly_corrupted",
                "predicted_corrupted": quality_call,
                "status": corruption_status,
            }
        )
    corruption = pd.DataFrame(corruption_rows)
    return {
        "representation_quality_outlier": pd.DataFrame(quality_rows),
        "representation_distortion_detection": pd.DataFrame(distortion_rows),
        "explicit_corruption_detection": corruption,
        "corruption_detection": corruption,
    }


def _distorted_state_detection(adata, truth: dict[str, Any], local_geometry_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(local_geometry_key)
    state_pair = store.get("state_pair_summary") if isinstance(store, dict) else None
    rows = []
    if not isinstance(state_pair, pd.DataFrame) or state_pair.empty:
        return pd.DataFrame(rows)
    df = state_pair[state_pair["metric"] == "local_distortion_median"].copy()
    if df.empty:
        return pd.DataFrame(rows)
    df["median"] = pd.to_numeric(df["median"], errors="coerce")
    vals = df["median"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        threshold = med + 3.0 * mad
        if threshold <= med:
            threshold = float(np.percentile(vals, 75.0))
    else:
        threshold = np.nan
    condition_shape_changed = set(truth.get("condition_shape_changed_states", truth.get("distorted_states", [])))
    equivalent_reps = set(truth.get("equivalent_representations", []))
    diagnostic_reps = set(
        truth.get(
            "diagnostically_distorted_representations",
            truth.get("diagnostic_distorted_representations", truth.get("distorted_representations", [])),
        )
    )
    affected_states = set(truth.get("representation_distortion_affected_states", []))
    for rec in df.to_dict(orient="records"):
        rep_a = str(rec.get("rep_a"))
        rep_b = str(rec.get("rep_b"))
        state = str(rec.get("state"))
        reps = {rep_a, rep_b}
        distorted = sorted(reps & diagnostic_reps)
        reference = sorted(reps & equivalent_reps)
        is_equivalent_pair = rep_a in equivalent_reps and rep_b in equivalent_reps
        is_target_pair = bool(distorted and reference)
        truth_positive = bool(distorted and reference and state in affected_states)
        value = float(rec.get("median", np.nan))
        predicted = bool(np.isfinite(value) and np.isfinite(threshold) and value > threshold)
        if is_equivalent_pair:
            status = "assessed"
        elif is_target_pair and affected_states:
            status = "assessed"
        else:
            status = "not_applicable"
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "rep_a": rep_a,
                "rep_b": rep_b,
                "state": state,
                "target": f"{rep_a}|{rep_b}|{state}",
                "distorted_representation": distorted[0] if distorted else "",
                "reference_representation": reference[0] if reference else "",
                "intended_affected_state": state in affected_states,
                "equivalent_pair": is_equivalent_pair,
                "local_distortion_score": value,
                "threshold": threshold,
                "true_condition_distribution_shape_changed": state in condition_shape_changed,
                "true_representation_local_distortion": truth_positive,
                "predicted_representation_local_distortion": predicted,
                "representation_distortion_applicable": status == "assessed",
                "compared_distorted_or_corrupted_representations": ",".join(distorted),
                "status": status,
                "true_distorted": truth_positive,
                "predicted_distorted": predicted,
            }
        )
    return pd.DataFrame(rows)


def _condition_shape_truth(truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    changed = set(truth.get("condition_shape_changed_states", truth.get("distorted_states", [])))
    for state in truth.get("states", []):
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "true_condition_distribution_shape_changed": state in changed,
            }
        )
    return pd.DataFrame(rows)


def _non_identifiable_truth(truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    non_identifiable = set(truth.get("non_identifiable_states", []))
    design = truth.get("replicate_design", {})
    if not isinstance(design, Mapping):
        design = {}
    for state in truth.get("states", []):
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "true_non_identifiable": state in non_identifiable,
                "non_identifiable_reason": truth.get("non_identifiable_reason", ""),
                "sample_offset_design": design.get("sample_offset_design", ""),
                "expected_outcome": design.get("expected_outcome", ""),
            }
        )
    return pd.DataFrame(rows)


def _neighborhood_discrimination(adata, truth: dict[str, Any], local_geometry_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(local_geometry_key)
    pair_summary = store.get("pair_summary") if isinstance(store, dict) else None
    rows = []
    if not isinstance(pair_summary, pd.DataFrame) or pair_summary.empty:
        return pd.DataFrame(rows)
    equiv = set(truth.get("equivalent_representations", []))
    corrupted = set(truth.get("corrupted_representations", []))
    df = pair_summary[(pair_summary["scope"] == "global") & (pair_summary["metric"].isin(["neighbor_overlap", "neighbor_jaccard"]))]
    for rec in df.to_dict(orient="records"):
        rep_a = str(rec.get("rep_a"))
        rep_b = str(rec.get("rep_b"))
        pair_class = "equivalent" if rep_a in equiv and rep_b in equiv else "non_equivalent"
        if rep_a in corrupted or rep_b in corrupted:
            pair_class = "corrupted"
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "rep_a": rep_a,
                "rep_b": rep_b,
                "k": int(rec.get("k", -1)),
                "metric": rec.get("metric"),
                "value": float(rec.get("median", np.nan)),
                "pair_class": pair_class,
            }
        )
    return pd.DataFrame(rows)


def _abundance_truth(truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    abundance_change = truth.get("true_abundance_change", {})
    scenario = str(truth.get("scenario", ""))
    has_design_abundance_effect = scenario in {"abundance_only", "unequal_cell_counts"}
    for state in truth.get("states", []):
        value = float(abundance_change.get(state, np.nan))
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "true_abundance_change": value,
                "true_abundance_changed": bool(has_design_abundance_effect and np.isfinite(value) and abs(value) > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _coverage_summary(adata, truth: dict[str, Any], representation_key: str, local_geometry_key: str) -> pd.DataFrame:
    rows = []
    scgeo = adata.uns.get("scgeo", {})
    for key, module in ((representation_key, "representation_stability"), (local_geometry_key, "local_geometry_stability")):
        store = scgeo.get(key)
        summary = store.get("coverage_summary") if isinstance(store, dict) else None
        if isinstance(summary, Mapping):
            rows.append(
                {
                    "scenario": truth["scenario"],
                    "seed": int(truth["seed"]),
                    "module": module,
                    "store_key": key,
                    "coverage_summary": json.dumps(summary, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _sample_offset_diagnostics(truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rec in truth.get("sample_offsets", []):
        if not isinstance(rec, Mapping):
            continue
        rows.append({"scenario": truth["scenario"], "seed": int(truth["seed"]), **dict(rec)})
    return pd.DataFrame(rows)


def _sample_center_diagnostics(adata, truth: dict[str, Any]) -> pd.DataFrame:
    rep = "X_truth"
    if rep not in adata.obsm:
        return pd.DataFrame()
    X = np.asarray(adata.obsm[rep], dtype=float)
    state_key = truth["state_key"]
    condition_key = truth["condition_key"]
    sample_key = truth["sample_key"]
    if not {state_key, condition_key, sample_key} <= set(adata.obs.columns):
        return pd.DataFrame()
    rows = []
    obs = adata.obs
    grouped = obs.groupby([condition_key, sample_key, state_key], sort=False).indices
    for (condition, sample, state), idx in grouped.items():
        center = X[np.asarray(idx, dtype=int)].mean(axis=0)
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "condition": str(condition),
                "sample": str(sample),
                "state": str(state),
                "n_cells": int(len(idx)),
                "center_norm": float(np.linalg.norm(center)),
                **{f"center_dim_{i}": float(center[i]) for i in range(center.shape[0])},
            }
        )
    return pd.DataFrame(rows)


def _shift_estimate_diagnostics(adata, truth: dict[str, Any], robust_shift_key: str) -> pd.DataFrame:
    store = adata.uns.get("scgeo", {}).get(robust_shift_key)
    rows = []
    if not isinstance(store, dict) or not isinstance(store.get("by"), dict):
        return pd.DataFrame(rows)
    for state in truth["states"]:
        rec = store["by"].get(state, {})
        if not isinstance(rec, Mapping):
            rec = {}
        ci = rec.get("bootstrap_magnitude_ci95", [np.nan, np.nan])
        if ci is None or len(ci) != 2:
            ci = [np.nan, np.nan]
        rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "state": state,
                "true_magnitude": float(truth["true_effect_magnitude"][state]),
                "delta_norm": float(rec.get("delta_norm", np.nan)),
                "normalized_delta_norm": float(rec.get("normalized_delta_norm", np.nan)),
                "bootstrap_ci95_low": float(ci[0]),
                "bootstrap_ci95_high": float(ci[1]),
                "bootstrap_directional_resultant_length": float(
                    rec.get("bootstrap_directional_resultant_length", np.nan)
                ),
                "direction_stability": float(rec.get("direction_stability", np.nan)),
                "sign_stability": float(rec.get("sign_stability", np.nan)),
                "n_cells0": int(rec.get("n_cells0", 0)),
                "n_cells1": int(rec.get("n_cells1", 0)),
                "n_samples0": rec.get("n_samples0", np.nan),
                "n_samples1": rec.get("n_samples1", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _runtime_summary(adata, truth: dict[str, Any]) -> pd.DataFrame:
    runtime = adata.uns.get("benchmark_runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    return pd.DataFrame(
        [
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "runtime_seconds": float(runtime.get("runtime_seconds", np.nan)),
                "peak_memory_mb": float(runtime.get("peak_memory_mb", np.nan)),
            }
        ]
    )


def evaluate_ground_truth(
    adata,
    *,
    robust_shift_key: str = "robust_shift",
    representation_key: str = "representation_stability",
    local_geometry_key: str = "local_geometry_stability",
) -> dict[str, pd.DataFrame]:
    """
    Evaluate stored ScGeo outputs against synthetic simulation truth.

    The function returns tidy DataFrames and does not create a composite score.
    Baseline magnitude comparisons are run on a temporary copy of ``adata``.
    Bootstrap magnitude intervals are treated as uncertainty intervals; states
    whose true magnitude is at or below ``_MAGNITUDE_ZERO_EPSILON`` are marked
    ``not_applicable`` for interval coverage rather than counted as failures.
    """
    truth = _require_truth(adata)
    threshold = _detection_threshold()
    state_metrics = _run_baseline_estimators(adata, truth, threshold)
    rank_metrics = _rank_metrics(state_metrics, truth)
    shift_detection = _shift_classification_metrics(state_metrics, truth)
    threshold_sensitivity = _threshold_sensitivity(state_metrics, truth)
    bootstrap_coverage = _bootstrap_coverage(adata, truth, robust_shift_key)
    representation_consensus = _representation_consensus(adata, truth, representation_key)
    alignment_accuracy = _alignment_accuracy(adata, truth, representation_key)
    representation_quality = _representation_quality_tables(adata, truth, local_geometry_key)
    quality_outlier_detection = representation_quality["representation_quality_outlier"]
    representation_distortion_detection = representation_quality["representation_distortion_detection"]
    explicit_corruption_detection = representation_quality["explicit_corruption_detection"]
    corruption_detection = representation_quality["corruption_detection"]
    distorted_detection = _distorted_state_detection(adata, truth, local_geometry_key)
    neighborhood_discrimination = _neighborhood_discrimination(adata, truth, local_geometry_key)
    abundance_truth = _abundance_truth(truth)
    condition_shape_truth = _condition_shape_truth(truth)
    non_identifiable_truth = _non_identifiable_truth(truth)

    summary_rows = []
    if not bootstrap_coverage.empty:
        applicable = bootstrap_coverage[bootstrap_coverage["coverage_status"] == "assessed"].copy()
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "bootstrap_interval_coverage",
                "method": robust_shift_key,
                "value": float(applicable["covered"].astype(bool).mean()) if not applicable.empty else np.nan,
                "n_applicable": int(applicable.shape[0]),
                "zero_effect_epsilon": float(_MAGNITUDE_ZERO_EPSILON),
            }
        )
    if not alignment_accuracy.empty:
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "alignment_class_accuracy",
                "method": "representation_stability",
                "value": float(alignment_accuracy["correct"].mean()),
            }
        )
    if not quality_outlier_detection.empty:
        assessed = quality_outlier_detection[quality_outlier_detection["status"] == "assessed"]
        cls = _classification_metrics(
            assessed["true_quality_outlier"].to_numpy(),
            assessed["predicted_quality_outlier"].to_numpy(),
        ) if not assessed.empty else {}
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "representation_quality_outlier",
                "method": "local_geometry_stability",
                **cls,
            }
        )
    if not representation_distortion_detection.empty:
        assessed = representation_distortion_detection[representation_distortion_detection["status"] == "assessed"]
        cls = _classification_metrics(
            assessed["true_representation_distorted"].to_numpy(),
            assessed["predicted_representation_distorted"].to_numpy(),
        ) if not assessed.empty else {}
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "representation_distortion_detection",
                "method": "local_geometry_stability",
                **cls,
            }
        )
    if not explicit_corruption_detection.empty:
        assessed = explicit_corruption_detection[explicit_corruption_detection["status"] == "assessed"]
        cls = _classification_metrics(
            assessed["true_corrupted"].to_numpy(),
            assessed["predicted_corrupted"].to_numpy(),
        ) if not assessed.empty else {}
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "explicit_corruption_detection",
                "method": "local_geometry_stability",
                **cls,
            }
        )
    if not distorted_detection.empty:
        assessed = distorted_detection[distorted_detection["status"] == "assessed"]
        cls = _classification_metrics(
            assessed["true_representation_local_distortion"].to_numpy(),
            assessed["predicted_representation_local_distortion"].to_numpy(),
        ) if not assessed.empty else {}
        summary_rows.append(
            {
                "scenario": truth["scenario"],
                "seed": int(truth["seed"]),
                "metric": "representation_local_distortion_detection",
                "method": "local_geometry_stability",
                **cls,
            }
        )

    return {
        "state_metrics": state_metrics,
        "rank_metrics": rank_metrics,
        "shift_detection": shift_detection,
        "threshold_sensitivity": threshold_sensitivity,
        "bootstrap_coverage": bootstrap_coverage,
        "representation_consensus": representation_consensus,
        "alignment_accuracy": alignment_accuracy,
        "representation_quality_outlier": quality_outlier_detection,
        "representation_distortion_detection": representation_distortion_detection,
        "explicit_corruption_detection": explicit_corruption_detection,
        "corruption_detection": corruption_detection,
        "distorted_state_detection": distorted_detection,
        "neighborhood_discrimination": neighborhood_discrimination,
        "abundance_truth": abundance_truth,
        "condition_shape_truth": condition_shape_truth,
        "non_identifiable_truth": non_identifiable_truth,
        "summary_metrics": pd.DataFrame(summary_rows),
        "coverage_summary": _coverage_summary(adata, truth, representation_key, local_geometry_key),
        "sample_offset_diagnostics": _sample_offset_diagnostics(truth),
        "sample_center_diagnostics": _sample_center_diagnostics(adata, truth),
        "shift_estimate_diagnostics": _shift_estimate_diagnostics(adata, truth, robust_shift_key),
        "runtime_summary": _runtime_summary(adata, truth),
    }


def _normalise_benchmark_table(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    out = df.copy()
    for col in _ABLATION_CONTEXT_COLUMNS:
        if col not in out:
            out[col] = np.nan
    for col in ("profile", "seed_split", "job_id", "scenario"):
        out[col] = out[col].astype(object)
    if not out.empty:
        out["profile"] = out["profile"].fillna("single")
        out["seed_split"] = out["seed_split"].fillna("evaluation")
        if "scenario" in out:
            out["scenario"] = out["scenario"].fillna("unknown")
        if "seed" in out:
            out["seed"] = out["seed"].fillna(0).astype(int)
        missing_job = out["job_id"].isna()
        if missing_job.any():
            out.loc[missing_job, "job_id"] = (
                out.loc[missing_job, "profile"].astype(str)
                + "_"
                + out.loc[missing_job, "scenario"].astype(str)
                + "_seed"
                + out.loc[missing_job, "seed"].astype(str)
            )
    return out


def _benchmark_table(tables: Mapping[str, Any], name: str) -> pd.DataFrame:
    return _normalise_benchmark_table(tables.get(name, pd.DataFrame()))


def _context(rec: Mapping[str, Any]) -> dict[str, Any]:
    out = {col: rec.get(col, np.nan) for col in _ABLATION_CONTEXT_COLUMNS}
    if pd.isna(out["profile"]):
        out["profile"] = "single"
    if pd.isna(out["seed_split"]):
        out["seed_split"] = "evaluation"
    if pd.isna(out["scenario"]):
        out["scenario"] = "unknown"
    if pd.isna(out["seed"]):
        out["seed"] = 0
    out["seed"] = int(out["seed"])
    if pd.isna(out["job_id"]):
        out["job_id"] = f"{out['profile']}_{out['scenario']}_seed{out['seed']}"
    return out


def _context_key(rec: Mapping[str, Any], *extra: Any) -> tuple[Any, ...]:
    ctx = _context(rec)
    return tuple(ctx[col] for col in _ABLATION_CONTEXT_COLUMNS) + tuple(extra)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _binary_outcome(
    truth: Any,
    call: Any,
    status: str,
    *,
    unstable_is_call: bool = False,
) -> str:
    status = str(status)
    if status == "not_computed":
        return "not_computed"
    if status == "not_applicable":
        return "not_applicable"
    if status == "non_identifiable":
        return "non_identifiable"
    if status in {"unavailable", "numerical_degeneracy"}:
        return status
    if status == "insufficient_coverage":
        return "insufficient_coverage"
    if status == "representation_unstable" and not unstable_is_call:
        return "representation_unstable"
    truth_bool = _as_bool(truth)
    call_bool = _as_bool(call)
    if truth_bool and call_bool:
        return "detects_correctly"
    if truth_bool and not call_bool:
        return "misses"
    if not truth_bool and call_bool:
        return "falsely_calls"
    return "correctly_rejects"


def _class_outcome(true_class: Any, predicted_class: Any, status: str) -> str:
    status = str(status)
    if status == "not_computed":
        return "not_computed"
    if status == "not_applicable":
        return "not_applicable"
    if status == "non_identifiable":
        return "non_identifiable"
    if status in {"unavailable", "numerical_degeneracy"}:
        return status
    if status in {"insufficient_coverage", "representation_unstable"}:
        return status
    true_class = str(true_class)
    predicted_class = str(predicted_class)
    if predicted_class == true_class:
        return "correctly_rejects" if true_class == "neutral" else "detects_correctly"
    if true_class != "neutral" and predicted_class == "neutral":
        return "misses"
    return "falsely_calls"


def _ablation_row(
    rows: list[dict[str, Any]],
    rec: Mapping[str, Any],
    *,
    variant: str,
    failure_mode: str,
    unit: str,
    target: Any,
    truth: Any,
    call: Any,
    status: str,
    outcome: str,
    evidence: str,
) -> None:
    rows.append(
        {
            **_context(rec),
            "variant": variant,
            "variant_label": _ABLATION_VARIANTS[variant],
            "failure_mode": failure_mode,
            "unit": unit,
            "target": str(target),
            "truth": truth,
            "call": call,
            "status": status,
            "outcome": outcome,
            "evidence": evidence,
        }
    )


def _not_computed_rows(
    rows: list[dict[str, Any]],
    units: pd.DataFrame,
    *,
    variants: Sequence[str],
    failure_mode: str,
    unit: str,
    target_col: str,
    truth_col: str,
    evidence: str,
) -> None:
    for rec in units.to_dict(orient="records"):
        for variant in variants:
            _ablation_row(
                rows,
                rec,
                variant=variant,
                failure_mode=failure_mode,
                unit=unit,
                target=rec.get(target_col),
                truth=rec.get(truth_col, False),
                call=np.nan,
                status="not_computed",
                outcome="not_computed",
                evidence=evidence,
            )


def framework_ablation(
    tables: Mapping[str, pd.DataFrame],
    *,
    final_split: str = "evaluation",
) -> pd.DataFrame:
    """
    Build a tidy framework-ablation table from synthetic benchmark outputs.

    The ablation compares five explicitly named framework variants and records
    per-unit outcomes without creating a composite score. Calibration and
    held-out evaluation rows are preserved; ``final_split`` only marks which
    rows should be used for final reporting.
    """
    if not isinstance(tables, Mapping):
        raise TypeError("tables must be a mapping of benchmark table names to pandas DataFrames")

    state_metrics = _benchmark_table(tables, "state_metrics")
    consensus = _benchmark_table(tables, "representation_consensus")
    distortion = _benchmark_table(tables, "distorted_state_detection")
    quality = _benchmark_table(tables, "representation_quality_outlier")
    representation_distortion = _benchmark_table(tables, "representation_distortion_detection")
    explicit_corruption = _benchmark_table(tables, "explicit_corruption_detection")
    alignment = _benchmark_table(tables, "alignment_accuracy")
    abundance = _benchmark_table(tables, "abundance_truth")
    condition_shape = _benchmark_table(tables, "condition_shape_truth")
    coverage = _benchmark_table(tables, "bootstrap_coverage")
    rows: list[dict[str, Any]] = []

    if not state_metrics.empty:
        state_units = state_metrics.drop_duplicates([*_ABLATION_CONTEXT_COLUMNS, "state"]).copy()
    else:
        state_units = pd.DataFrame(columns=[*_ABLATION_CONTEXT_COLUMNS, "state", "true_shifted"])

    consensus_index = {
        _context_key(rec, rec.get("state")): rec
        for rec in consensus.to_dict(orient="records")
    }

    mean_rows = state_metrics[state_metrics.get("method") == "shift_mean"] if not state_metrics.empty else pd.DataFrame()
    for rec in mean_rows.to_dict(orient="records"):
        status = (
            "non_identifiable"
            if _as_bool(rec.get("non_identifiable", False))
            else "assessed" if _finite(rec.get("estimated_magnitude")) else "unavailable"
        )
        outcome = _binary_outcome(rec.get("true_shifted"), rec.get("predicted_shifted"), status)
        _ablation_row(
            rows,
            rec,
            variant="A_mean_shift_one_rep",
            failure_mode="centroid_shift",
            unit="state",
            target=rec.get("state"),
            truth=_as_bool(rec.get("true_shifted")),
            call=_as_bool(rec.get("predicted_shifted")),
            status=status,
            outcome=outcome,
            evidence="scgeo.tl.shift on X_truth",
        )

    robust_rows = (
        state_metrics[state_metrics.get("method") == "robust_geometric_median"]
        if not state_metrics.empty
        else pd.DataFrame()
    )
    for rec in robust_rows.to_dict(orient="records"):
        status = (
            "non_identifiable"
            if _as_bool(rec.get("non_identifiable", False))
            else "assessed" if _finite(rec.get("estimated_magnitude")) else "unavailable"
        )
        outcome = _binary_outcome(rec.get("true_shifted"), rec.get("predicted_shifted"), status)
        _ablation_row(
            rows,
            rec,
            variant="B_robust_shift_one_rep",
            failure_mode="centroid_shift",
            unit="state",
            target=rec.get("state"),
            truth=_as_bool(rec.get("true_shifted")),
            call=_as_bool(rec.get("predicted_shifted")),
            status=status,
            outcome=outcome,
            evidence="robust_shift geometric_median on X_truth",
        )

        cons = consensus_index.get(_context_key(rec, rec.get("state")))
        consensus_status = "unavailable" if cons is None else str(cons.get("status", "unavailable"))
        if _as_bool(rec.get("non_identifiable", False)):
            consensus_status = "non_identifiable"
        for variant in (
            "C_robust_shift_representation_consensus",
            "D_plus_local_geometry",
            "E_plus_dynamics",
        ):
            outcome = _binary_outcome(rec.get("true_shifted"), rec.get("predicted_shifted"), consensus_status)
            _ablation_row(
                rows,
                rec,
                variant=variant,
                failure_mode="centroid_shift",
                unit="state",
                target=rec.get("state"),
                truth=_as_bool(rec.get("true_shifted")),
                call=_as_bool(rec.get("predicted_shifted")),
                status=consensus_status,
                outcome=outcome,
                evidence="robust_shift X_truth gated by representation consensus",
            )

    if not state_units.empty:
        state_units = state_units.copy()
        state_units["true_representation_instability"] = (
            (state_units["scenario"].astype(str) == "representation_corruption")
            & state_units.get("true_shifted", False).astype(bool)
        )
        _not_computed_rows(
            rows,
            state_units,
            variants=("A_mean_shift_one_rep", "B_robust_shift_one_rep"),
            failure_mode="representation_instability",
            unit="state",
            target_col="state",
            truth_col="true_representation_instability",
            evidence="representation consensus not included",
        )
        for rec in state_units.to_dict(orient="records"):
            cons = consensus_index.get(_context_key(rec, rec.get("state")))
            status = "unavailable" if cons is None else str(cons.get("status", "unavailable"))
            call = status == "representation_unstable"
            for variant in (
                "C_robust_shift_representation_consensus",
                "D_plus_local_geometry",
                "E_plus_dynamics",
            ):
                outcome = _binary_outcome(
                    rec.get("true_representation_instability"),
                    call,
                    "assessed" if status == "representation_unstable" else status,
                    unstable_is_call=True,
                )
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="representation_instability",
                    unit="state",
                    target=rec.get("state"),
                    truth=_as_bool(rec.get("true_representation_instability")),
                    call=call,
                    status=status,
                    outcome=outcome,
                    evidence="representation_stability consensus_label",
                )

    if not distortion.empty:
        _not_computed_rows(
            rows,
            distortion,
            variants=(
                "A_mean_shift_one_rep",
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
            ),
            failure_mode="representation_local_distortion",
            unit="representation_pair_state",
            target_col="target",
            truth_col="true_representation_local_distortion",
            evidence="local geometry diagnostics not included",
        )
        for rec in distortion.to_dict(orient="records"):
            status = str(rec.get("status", "unavailable"))
            if status == "assessed" and not _finite(rec.get("local_distortion_score")):
                status = "unavailable"
            outcome = _binary_outcome(
                rec.get("true_representation_local_distortion"),
                rec.get("predicted_representation_local_distortion"),
                status,
            )
            for variant in ("D_plus_local_geometry", "E_plus_dynamics"):
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="representation_local_distortion",
                    unit="representation_pair_state",
                    target=rec.get("target"),
                    truth=_as_bool(rec.get("true_representation_local_distortion")),
                    call=_as_bool(rec.get("predicted_representation_local_distortion")),
                    status=status,
                    outcome=outcome,
                    evidence="local_geometry_stability state_pair_summary",
                )

    if not condition_shape.empty:
        _not_computed_rows(
            rows,
            condition_shape,
            variants=tuple(_ABLATION_VARIANTS),
            failure_mode="condition_distribution_shape_change",
            unit="state",
            target_col="state",
            truth_col="true_condition_distribution_shape_changed",
            evidence="condition-wise shape truth is reported separately from representation geometry",
        )

    if not quality.empty:
        _not_computed_rows(
            rows,
            quality,
            variants=(
                "A_mean_shift_one_rep",
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
            ),
            failure_mode="representation_quality_outlier",
            unit="representation",
            target_col="rep",
            truth_col="true_quality_outlier",
            evidence="local representation diagnostics not included",
        )
        for rec in quality.to_dict(orient="records"):
            status = str(rec.get("status", "unavailable"))
            outcome = _binary_outcome(rec.get("true_quality_outlier"), rec.get("predicted_quality_outlier"), status)
            for variant in ("D_plus_local_geometry", "E_plus_dynamics"):
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="representation_quality_outlier",
                    unit="representation",
                    target=rec.get("rep"),
                    truth=_as_bool(rec.get("true_quality_outlier")),
                    call=_as_bool(rec.get("predicted_quality_outlier")),
                    status=status,
                    outcome=outcome,
                    evidence="local_geometry_stability representation_summary flags",
                )

    if not representation_distortion.empty:
        _not_computed_rows(
            rows,
            representation_distortion,
            variants=(
                "A_mean_shift_one_rep",
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
            ),
            failure_mode="representation_distortion_detection",
            unit="representation",
            target_col="rep",
            truth_col="true_representation_distorted",
            evidence="local representation diagnostics not included",
        )
        for rec in representation_distortion.to_dict(orient="records"):
            status = str(rec.get("status", "unavailable"))
            outcome = _binary_outcome(
                rec.get("true_representation_distorted"),
                rec.get("predicted_representation_distorted"),
                status,
            )
            for variant in ("D_plus_local_geometry", "E_plus_dynamics"):
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="representation_distortion_detection",
                    unit="representation",
                    target=rec.get("rep"),
                    truth=_as_bool(rec.get("true_representation_distorted")),
                    call=_as_bool(rec.get("predicted_representation_distorted")),
                    status=status,
                    outcome=outcome,
                    evidence="local_geometry_stability representation_summary distortion flags",
                )

    if not explicit_corruption.empty:
        _not_computed_rows(
            rows,
            explicit_corruption,
            variants=(
                "A_mean_shift_one_rep",
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
            ),
            failure_mode="explicit_corruption_detection",
            unit="representation",
            target_col="rep",
            truth_col="true_corrupted",
            evidence="local representation diagnostics not included",
        )
        for rec in explicit_corruption.to_dict(orient="records"):
            status = str(rec.get("status", "unavailable"))
            outcome = _binary_outcome(rec.get("true_corrupted"), rec.get("predicted_corrupted"), status)
            for variant in ("D_plus_local_geometry", "E_plus_dynamics"):
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="explicit_corruption_detection",
                    unit="representation",
                    target=rec.get("rep"),
                    truth=_as_bool(rec.get("true_corrupted")),
                    call=_as_bool(rec.get("predicted_corrupted")),
                    status=status,
                    outcome=outcome,
                    evidence="local_geometry_stability representation_summary flags",
                )

    if not alignment.empty:
        consensus_alignment = alignment[alignment.get("source") == "consensus"].copy()
        _not_computed_rows(
            rows,
            consensus_alignment,
            variants=(
                "A_mean_shift_one_rep",
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
                "D_plus_local_geometry",
            ),
            failure_mode="dynamics_alignment",
            unit="state",
            target_col="state",
            truth_col="true_class",
            evidence="dynamics agreement not included",
        )
        for rec in consensus_alignment.to_dict(orient="records"):
            cons = consensus_index.get(_context_key(rec, rec.get("state")))
            consensus_status = "unavailable" if cons is None else str(cons.get("status", "unavailable"))
            if not _as_bool(rec.get("velocity_requested", False)):
                status = "unavailable"
            elif consensus_status != "assessed":
                status = consensus_status
            elif str(rec.get("predicted_class")) == "unavailable":
                status = "unavailable"
            else:
                status = "assessed"
            outcome = _class_outcome(rec.get("true_class"), rec.get("predicted_class"), status)
            _ablation_row(
                rows,
                rec,
                variant="E_plus_dynamics",
                failure_mode="dynamics_alignment",
                unit="state",
                target=rec.get("state"),
                truth=rec.get("true_class"),
                call=rec.get("predicted_class"),
                status=status,
                outcome=outcome,
                evidence="representation_stability consensus velocity alignment",
            )

    if not abundance.empty:
        _not_computed_rows(
            rows,
            abundance,
            variants=tuple(_ABLATION_VARIANTS),
            failure_mode="abundance_change",
            unit="state",
            target_col="state",
            truth_col="true_abundance_changed",
            evidence="abundance module not part of ablation variants",
        )

    if not coverage.empty:
        _not_computed_rows(
            rows,
            coverage,
            variants=("A_mean_shift_one_rep",),
            failure_mode="bootstrap_interval_coverage",
            unit="state",
            target_col="state",
            truth_col="covered",
            evidence="original mean shift has no bootstrap interval in this ablation",
        )
        for rec in coverage.to_dict(orient="records"):
            status = str(rec.get("coverage_status", "unavailable"))
            if status not in {"assessed", "not_applicable"}:
                status = "unavailable"
            truth_value = True if status == "assessed" else np.nan
            outcome = _binary_outcome(truth_value, rec.get("covered"), status)
            for variant in (
                "B_robust_shift_one_rep",
                "C_robust_shift_representation_consensus",
                "D_plus_local_geometry",
                "E_plus_dynamics",
            ):
                _ablation_row(
                    rows,
                    rec,
                    variant=variant,
                    failure_mode="bootstrap_interval_coverage",
                    unit="state",
                    target=rec.get("state"),
                    truth=truth_value,
                    call=_as_bool(rec.get("covered")),
                    status=status,
                    outcome=outcome,
                    evidence="robust_shift bootstrap_magnitude_ci95 uncertainty interval",
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                *_ABLATION_CONTEXT_COLUMNS,
                "variant",
                "variant_label",
                "failure_mode",
                "unit",
                "target",
                "truth",
                "call",
                "status",
                "outcome",
                "evidence",
                "final_evaluation",
            ]
        )
    out["final_evaluation"] = out["seed_split"].astype(str) == str(final_split)
    return out


def _seed_level_framework_ablation(ablation_table: pd.DataFrame, *, split: str = "evaluation") -> pd.DataFrame:
    if not isinstance(ablation_table, pd.DataFrame) or ablation_table.empty:
        return pd.DataFrame()
    df = ablation_table.copy()
    if "seed_split" in df:
        df = df[df["seed_split"].astype(str) == str(split)].copy()
    if df.empty:
        return pd.DataFrame()
    job_cols = [
        "seed_split",
        "profile",
        "job_id",
        "scenario",
        "seed",
        "variant",
        "variant_label",
        "failure_mode",
    ]
    for col in job_cols:
        if col not in df:
            df[col] = np.nan
    count = df.groupby([*job_cols, "outcome"], dropna=False).size().reset_index(name="n")
    totals = df.groupby(job_cols, dropna=False).size().reset_index(name="n_total")
    grid = []
    for rec in totals.to_dict(orient="records"):
        for outcome in _ABLATION_OUTCOME_ORDER:
            grid.append({**rec, "outcome": outcome})
    out = pd.DataFrame(grid)
    out = out.merge(count, on=[*job_cols, "outcome"], how="left")
    out["n"] = out["n"].fillna(0).astype(int)
    out["rate"] = out["n"] / out["n_total"].replace(0, np.nan)
    out["performance_applicable"] = out["outcome"].isin(_PERFORMANCE_OUTCOMES)
    out["support_status"] = out["outcome"].isin(_SUPPORT_OUTCOMES)
    return out


def _normal_ci95(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    mean = float(np.mean(arr))
    return max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def _summarize_framework_ablation(ablation_table: pd.DataFrame, *, split: str = "evaluation") -> pd.DataFrame:
    seed_summary = _seed_level_framework_ablation(ablation_table, split=split)
    if seed_summary.empty:
        return pd.DataFrame()
    group_cols = ["seed_split", "scenario", "variant", "variant_label", "failure_mode", "outcome"]
    rows = []
    for key, df in seed_summary.groupby(group_cols, dropna=False, sort=False):
        rates = df["rate"].to_numpy(dtype=float)
        rates = rates[np.isfinite(rates)]
        ci_low, ci_high = _normal_ci95(df["rate"])
        if rates.size:
            q25, q75 = np.percentile(rates, [25.0, 75.0])
            mean_rate = float(np.mean(rates))
            median_rate = float(np.median(rates))
            std_rate = float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0
            min_rate = float(np.min(rates))
            max_rate = float(np.max(rates))
        else:
            q25 = q75 = mean_rate = median_rate = std_rate = min_rate = max_rate = np.nan
        rec = dict(zip(group_cols, key))
        rows.append(
            {
                **rec,
                "n_jobs": int(df["job_id"].nunique()),
                "n": int(df["n"].sum()),
                "n_total": int(df["n_total"].sum()),
                "mean_n_units_per_job": float(df["n_total"].mean()),
                "rate": mean_rate,
                "mean_rate": mean_rate,
                "median_rate": median_rate,
                "std_rate": std_rate,
                "min_rate": min_rate,
                "max_rate": max_rate,
                "q25_rate": float(q25),
                "q75_rate": float(q75),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "plot_framework_ablation requires matplotlib. Install matplotlib to generate the summary figure."
        ) from exc
    return plt


def _ablation_variant_order() -> list[str]:
    return list(_ABLATION_VARIANTS)


def _capability_matrix() -> pd.DataFrame:
    rows = []
    for variant in _ablation_variant_order():
        capabilities = set(_CAPABILITY_BY_VARIANT.get(variant, ()))
        rows.append({column: float(column in capabilities) for column in _CAPABILITY_COLUMNS})
    return pd.DataFrame(rows, index=[_VARIANT_SHORT_LABELS[variant] for variant in _ablation_variant_order()])


def _row_spec_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "failure_mode": failure_mode,
                "row_label": label,
                "row_order": order,
            }
            for order, (scenario, failure_mode, label) in enumerate(_MAIN_ABLATION_ROW_SPECS)
        ]
    )


def _short_ablation_row_label(scenario: Any, failure_mode: Any) -> str:
    key = (str(scenario), str(failure_mode))
    if key in _MAIN_ABLATION_LABELS:
        return _MAIN_ABLATION_LABELS[key]
    scenario_label = str(scenario).replace("_", " ")
    failure_label = str(failure_mode).replace("_", " ")
    replacements = {
        "balanced replicate heterogeneity": "Replicate het.",
        "batch condition confounding": "Batch confound.",
        "representation corruption": "Rep. corrupt.",
        "representation instability": "Rep. stability",
        "bootstrap interval coverage": "Uncertainty",
        "explicit corruption detection": "Corruption",
        "representation distortion detection": "Distortion",
        "representation local distortion": "Local geometry",
        "centroid shift": "Shift",
        "dynamics alignment": "Dynamics",
    }
    return f"{replacements.get(scenario_label, scenario_label.title())} | {replacements.get(failure_label, failure_label.title())}"


def _status_label(status: Any) -> str:
    return {
        "representation_unstable": "unstable",
        "insufficient_coverage": "low coverage",
        "non_identifiable": "non-ident.",
        "unavailable": "unavailable",
    }.get(str(status), str(status).replace("_", " "))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _wrap_label(label: Any, *, width: int) -> str:
    text = str(label)
    width = max(4, int(width))
    if len(text) <= width and "\n" not in text:
        return text
    parts = []
    for part in text.split("\n"):
        wrapped = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False)
        parts.extend(wrapped or [""])
    return "\n".join(parts)


def _display_labels(labels: Sequence[Any], *, wrap_width: int) -> list[str]:
    return [_wrap_label(label, width=wrap_width) for label in labels]


def _max_wrapped_line_length(labels: Sequence[str]) -> int:
    max_len = 0
    for label in labels:
        for line in str(label).split("\n"):
            max_len = max(max_len, len(line))
    return max_len


def _auto_panel_widths(
    *,
    performance_labels: Sequence[str],
    support_labels: Sequence[str],
    wrap_width: int,
) -> dict[str, float]:
    perf_wrapped = _display_labels(performance_labels, wrap_width=wrap_width)
    support_wrapped = _display_labels(support_labels, wrap_width=wrap_width)
    perf_len = _max_wrapped_line_length(perf_wrapped)
    support_len = _max_wrapped_line_length(support_wrapped)
    return {
        "capability": _clamp(
            _PLOT_AUTO_MIN_PANEL_WIDTHS["capability"] + 0.03 * len(_CAPABILITY_COLUMNS),
            _PLOT_AUTO_MIN_PANEL_WIDTHS["capability"],
            _PLOT_AUTO_MAX_PANEL_WIDTHS["capability"],
        ),
        "performance": _clamp(
            _PLOT_AUTO_MIN_PANEL_WIDTHS["performance"] + 0.055 * perf_len,
            _PLOT_AUTO_MIN_PANEL_WIDTHS["performance"],
            _PLOT_AUTO_MAX_PANEL_WIDTHS["performance"],
        ),
        "support": _clamp(
            _PLOT_AUTO_MIN_PANEL_WIDTHS["support"] + 0.050 * support_len,
            _PLOT_AUTO_MIN_PANEL_WIDTHS["support"],
            _PLOT_AUTO_MAX_PANEL_WIDTHS["support"],
        ),
    }


def _auto_main_figsize(
    *,
    n_performance_rows: int,
    n_support_rows: int,
    performance_labels: Sequence[str],
    support_labels: Sequence[str],
    row_height: float,
    min_height: float,
    wrap_width: int,
) -> tuple[tuple[float, float], dict[str, float]]:
    panel_widths = _auto_panel_widths(
        performance_labels=performance_labels,
        support_labels=support_labels,
        wrap_width=wrap_width,
    )
    body_rows = max(5, int(n_performance_rows), int(n_support_rows), 1)
    support_extra = 0.08 * max(0, int(n_support_rows) - 2)
    height = max(float(min_height), 1.45 + float(row_height) * body_rows + support_extra)
    width = sum(panel_widths.values()) + 1.05
    return (float(width), float(height)), panel_widths


def _masked_cell_count(matrix: pd.DataFrame) -> int:
    if matrix.empty:
        return 0
    values = matrix.to_numpy(dtype=float)
    return int(np.size(values) - np.isfinite(values).sum())


def _heatmap_annotation_font_size(n_rows: int, n_cols: int) -> float:
    return _clamp(7.0 - 0.08 * max(0, int(n_rows) - 9) - 0.22 * max(0, int(n_cols) - 5), 4.5, 7.0)


def _drawn_axis_dimensions(fig, axes: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    fig.canvas.draw()
    width, height = fig.get_size_inches()
    out = {}
    for name, ax in axes.items():
        bbox = ax.get_position()
        out[name] = {
            "x0": float(bbox.x0 * width),
            "y0": float(bbox.y0 * height),
            "width": float(bbox.width * width),
            "height": float(bbox.height * height),
        }
    return out


def _compact_performance_matrix(
    ablation_table: pd.DataFrame,
    *,
    variant_order: Sequence[str],
    normalize: bool,
) -> pd.DataFrame:
    specs = _row_spec_frame()
    perf = ablation_table[ablation_table["outcome"].isin(_PERFORMANCE_OUTCOMES)].copy()
    if perf.empty:
        return pd.DataFrame(index=[], columns=list(variant_order), dtype=float)
    perf = perf.merge(specs, on=["scenario", "failure_mode"], how="inner")
    if perf.empty:
        return pd.DataFrame(index=[], columns=list(variant_order), dtype=float)
    perf["success"] = perf["outcome"].isin({"detects_correctly", "correctly_rejects"}).astype(float)
    job_rate = (
        perf.groupby(["job_id", "row_label", "row_order", "variant"], dropna=False)["success"]
        .agg("mean" if normalize else "sum")
        .reset_index(name="applicable_success_rate")
    )
    matrix = job_rate.pivot_table(
        index=["row_order", "row_label"],
        columns="variant",
        values="applicable_success_rate",
        aggfunc="mean" if normalize else "sum",
    )
    matrix = matrix.sort_index(level="row_order")
    matrix.index = matrix.index.get_level_values("row_label")
    return matrix.reindex(columns=list(variant_order))


def _support_status_matrix(
    ablation_table: pd.DataFrame,
    *,
    variant_order: Sequence[str],
    normalize: bool,
) -> pd.DataFrame:
    base = ablation_table.copy()
    expected_missing_dynamics = (
        (base["failure_mode"].astype(str) == "dynamics_alignment")
        & (~base["scenario"].astype(str).isin(_DYNAMICS_SCENARIOS))
        & (base["outcome"].astype(str) == "unavailable")
    )
    base = base[~expected_missing_dynamics].copy()
    base = base[~base["outcome"].isin({"not_computed", "not_applicable"})].copy()
    if base.empty:
        return pd.DataFrame(index=[], columns=list(variant_order), dtype=float)
    denominator_cols = ["job_id", "scenario", "failure_mode", "variant"]
    totals = base.groupby(denominator_cols, dropna=False).size().reset_index(name="n_total")
    support = base[base["outcome"].isin(_SUPPORT_STATUS_ORDER)].copy()
    if support.empty:
        return pd.DataFrame(index=[], columns=list(variant_order), dtype=float)
    counts = (
        support.groupby([*denominator_cols, "outcome"], dropna=False)
        .size()
        .reset_index(name="n")
        .merge(totals, on=denominator_cols, how="left")
    )
    counts["support_rate"] = counts["n"] / counts["n_total"].replace(0, np.nan)
    if not normalize:
        counts["support_rate"] = counts["n"]
    main_order = {
        (scenario, failure_mode): order
        for order, (scenario, failure_mode, _label) in enumerate(_MAIN_ABLATION_ROW_SPECS)
    }
    fallback_pairs = sorted(
        {
            (str(scenario), str(failure_mode))
            for scenario, failure_mode in zip(counts["scenario"], counts["failure_mode"])
            if (str(scenario), str(failure_mode)) not in main_order
        }
    )
    fallback_order = {pair: 100 + order for order, pair in enumerate(fallback_pairs)}
    status_order = {status: order for order, status in enumerate(_SUPPORT_STATUS_ORDER)}
    counts["base_label"] = [
        _short_ablation_row_label(scenario, failure_mode)
        for scenario, failure_mode in zip(counts["scenario"], counts["failure_mode"])
    ]
    counts["row_label"] = counts["base_label"] + " | " + counts["outcome"].map(_status_label)
    counts["row_order"] = [
        main_order[pair] if pair in main_order else fallback_order[pair]
        for scenario, failure_mode in zip(counts["scenario"], counts["failure_mode"])
        for pair in [(str(scenario), str(failure_mode))]
    ]
    counts["status_order"] = counts["outcome"].map(status_order).fillna(100).astype(int)
    job_rate = (
        counts.groupby(["job_id", "row_label", "row_order", "status_order", "variant"], dropna=False)["support_rate"]
        .mean()
        .reset_index(name="job_support_rate")
    )
    matrix = job_rate.pivot_table(
        index=["row_order", "status_order", "row_label"],
        columns="variant",
        values="job_support_rate",
        aggfunc="mean" if normalize else "sum",
    )
    matrix = matrix.sort_index(level=["row_order", "status_order", "row_label"])
    matrix.index = matrix.index.get_level_values("row_label")
    matrix = matrix.reindex(columns=list(variant_order))
    return matrix.loc[matrix.fillna(0.0).sum(axis=1) > 0.0]


def _framework_ablation_plot_matrices(
    ablation_table: pd.DataFrame,
    *,
    normalize: bool = True,
) -> dict[str, pd.DataFrame]:
    variant_order = _ablation_variant_order()
    return {
        "capability": _capability_matrix(),
        "performance": _compact_performance_matrix(
            ablation_table,
            variant_order=variant_order,
            normalize=normalize,
        ),
        "support": _support_status_matrix(
            ablation_table,
            variant_order=variant_order,
            normalize=normalize,
        ),
    }


def plot_framework_ablation(
    ablation_table: pd.DataFrame,
    *,
    split: str = "evaluation",
    normalize: bool = True,
    figsize="auto",
    row_height: float = _PLOT_AUTO_ROW_HEIGHT,
    min_height: float = _PLOT_AUTO_MIN_HEIGHT,
    wrap_width: int = 18,
    show_values: bool | str = "auto",
    title: Optional[str] = None,
    save_path=None,
    show: bool = True,
):
    """
    Plot a manuscript-ready framework-ablation summary.

    ``figsize='auto'`` sizes the figure from performance/support row counts,
    row height, the documented minimum height, and wrapped label lengths.
    The figure separates capability coverage, applicable performance outcomes,
    and support statuses so unavailable or not-computed evidence is not plotted
    on the same color scale as performance.
    """
    if not isinstance(ablation_table, pd.DataFrame) or ablation_table.empty:
        raise ValueError("ablation_table contains no rows to plot")
    df = ablation_table.copy()
    if "seed_split" in df:
        df = df[df["seed_split"].astype(str) == str(split)].copy()
    if df.empty:
        raise ValueError("ablation_table contains no rows to plot")
    plt = _import_matplotlib_pyplot()
    matrices = _framework_ablation_plot_matrices(df, normalize=normalize)
    capability = matrices["capability"]
    perf_matrix = matrices["performance"]
    support_matrix = matrices["support"]

    perf_original_labels = [str(label) for label in perf_matrix.index]
    support_original_labels = [str(label) for label in support_matrix.index]
    perf_display_labels = _display_labels(perf_original_labels, wrap_width=wrap_width)
    support_display_labels = _display_labels(support_original_labels, wrap_width=wrap_width)
    capability_display_columns = [_CAPABILITY_DISPLAY_LABELS[col] for col in capability.columns]

    if figsize == "auto" or figsize is None:
        figsize, panel_widths = _auto_main_figsize(
            n_performance_rows=perf_matrix.shape[0],
            n_support_rows=support_matrix.shape[0],
            performance_labels=perf_original_labels,
            support_labels=support_original_labels,
            row_height=row_height,
            min_height=min_height,
            wrap_width=wrap_width,
        )
    else:
        panel_widths = _auto_panel_widths(
            performance_labels=perf_original_labels,
            support_labels=support_original_labels,
            wrap_width=wrap_width,
        )
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.075],
        width_ratios=[
            panel_widths["capability"],
            panel_widths["performance"],
            panel_widths["support"],
        ],
    )
    ax_cap = fig.add_subplot(grid[0, 0])
    ax_perf = fig.add_subplot(grid[0, 1])
    ax_support = fig.add_subplot(grid[0, 2])
    cax_perf = fig.add_subplot(grid[1, 1])
    cax_support = fig.add_subplot(grid[1, 2])

    def _draw_capability_table(ax, matrix: pd.DataFrame, *, title_text: str):
        ax.axis("off")
        table_text = [["✓" if bool(value) else "" for value in row] for row in matrix.to_numpy(dtype=float)]
        table = ax.table(
            cellText=table_text,
            rowLabels=list(matrix.index),
            colLabels=capability_display_columns,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.04, 1.28)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#c8c8c8")
            cell.set_linewidth(0.6)
            text = cell.get_text()
            text.set_ha("center")
            text.set_va("center")
            text.set_multialignment("center")
            if row == 0:
                cell.set_facecolor("#f2f2f2")
                cell.set_height(cell.get_height() * 1.70)
                text.set_fontsize(6.4)
                text.set_weight("bold")
            elif col == -1:
                cell.set_facecolor("#f2f2f2")
                text.set_fontsize(7.8)
                text.set_weight("bold")
            elif cell.get_text().get_text():
                cell.set_facecolor("#d9ead3")
                text.set_fontsize(8.6)
            else:
                cell.set_facecolor("#ffffff")
                text.set_fontsize(8.0)
        ax.set_title(title_text, fontsize=9)

    def _draw_rate_heatmap(ax, matrix: pd.DataFrame, *, title_text: str, cmap: str, vmin: float, vmax: float):
        if matrix.empty:
            ax.text(0.5, 0.5, "No applicable rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title_text, fontsize=9)
            return None
        values = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#ffffff")
        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([_VARIANT_SHORT_LABELS.get(str(x), str(x)[:1]) for x in matrix.columns], fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ylabels = perf_display_labels if matrix is perf_matrix else support_display_labels
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(title_text, fontsize=9)
        n_cells = int(matrix.shape[0] * matrix.shape[1])
        draw_values = bool(show_values) if isinstance(show_values, bool) else n_cells <= 125
        if draw_values:
            font_size = _heatmap_annotation_font_size(matrix.shape[0], matrix.shape[1])
            midpoint = vmin + 0.55 * (vmax - vmin)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = values[i, j]
                    if not np.isfinite(value):
                        continue
                    color = "#ffffff" if value >= midpoint else "#111111"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=font_size, color=color)
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#e0e0e0", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        return im

    _draw_capability_table(ax_cap, capability, title_text="Capability")

    im1 = _draw_rate_heatmap(
        ax_perf,
        perf_matrix,
        title_text="Applicable Performance",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0 if normalize else max(1.0, float(np.nanmax(perf_matrix.to_numpy(dtype=float))) if not perf_matrix.empty else 1.0),
    )
    ax_perf.set_xlabel("Variant", fontsize=8)

    im2 = _draw_rate_heatmap(
        ax_support,
        support_matrix,
        title_text="Support Status",
        cmap="OrRd",
        vmin=0.0,
        vmax=1.0 if normalize else max(1.0, float(np.nanmax(support_matrix.to_numpy(dtype=float))) if not support_matrix.empty else 1.0),
    )
    ax_support.set_xlabel("Variant", fontsize=8)
    if not support_matrix.empty:
        box_aspect = _clamp(0.70 * max(1, support_matrix.shape[0]) / max(1, support_matrix.shape[1]), 0.18, 0.85)
        ax_support.set_box_aspect(box_aspect)
        ax_support.set_anchor("N")

    if im1 is not None:
        fig.colorbar(
            im1,
            cax=cax_perf,
            orientation="horizontal",
            label="Success rate" if normalize else "Success count",
        )
        cax_perf.tick_params(labelsize=7)
        cax_perf.xaxis.label.set_size(7)
    else:
        cax_perf.axis("off")
    if im2 is not None:
        fig.colorbar(
            im2,
            cax=cax_support,
            orientation="horizontal",
            label="Status rate" if normalize else "Status count",
        )
        cax_support.tick_params(labelsize=7)
        cax_support.xaxis.label.set_size(7)
    else:
        cax_support.axis("off")
    fig.scgeo_ablation_matrices = matrices
    fig.suptitle(title or f"Framework ablation ({split} seeds)", fontsize=10)
    axes_by_name = {
        "capability": ax_cap,
        "performance": ax_perf,
        "support": ax_support,
        "performance_colorbar": cax_perf,
        "support_colorbar": cax_support,
    }
    panel_dimensions = _drawn_axis_dimensions(fig, axes_by_name)
    fig.scgeo_ablation_plot_metadata = {
        "figure_dimensions": {
            "width": float(fig.get_size_inches()[0]),
            "height": float(fig.get_size_inches()[1]),
        },
        "panel_dimensions": panel_dimensions,
        "displayed_labels": {
            "capability_columns": capability_display_columns,
            "performance_rows": perf_display_labels,
            "support_rows": support_display_labels,
            "variants": [_VARIANT_SHORT_LABELS.get(str(x), str(x)[:1]) for x in perf_matrix.columns],
        },
        "original_labels": {
            "capability_columns": list(capability.columns),
            "performance_rows": perf_original_labels,
            "support_rows": support_original_labels,
            "variants": list(perf_matrix.columns),
        },
        "page_count": 1,
        "masked_cell_count": {
            "performance": _masked_cell_count(perf_matrix),
            "support": _masked_cell_count(support_matrix),
            "total": _masked_cell_count(perf_matrix) + _masked_cell_count(support_matrix),
        },
        "auto_sizing": {
            "figsize": figsize,
            "row_height": float(row_height),
            "min_height": float(min_height),
            "wrap_width": int(wrap_width),
            "panel_widths": panel_widths,
        },
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _plot_framework_ablation_supplemental(
    ablation_table: pd.DataFrame,
    *,
    split: str = "evaluation",
    normalize: bool = True,
    figsize="auto",
    row_height: float = _PLOT_AUTO_ROW_HEIGHT,
    min_height: float = _PLOT_AUTO_MIN_HEIGHT,
    wrap_width: int = 22,
    show_values: bool | str = "auto",
    max_rows_per_page: int = 25,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    save_path=None,
    show: bool = True,
):
    if not isinstance(ablation_table, pd.DataFrame) or ablation_table.empty:
        raise ValueError("ablation_table contains no rows to plot")
    df = ablation_table.copy()
    if "seed_split" in df:
        df = df[df["seed_split"].astype(str) == str(split)].copy()
    if df.empty:
        raise ValueError("ablation_table contains no rows to plot")
    plt = _import_matplotlib_pyplot()
    variant_order = _ablation_variant_order()
    perf = df[df["outcome"].isin(_PERFORMANCE_OUTCOMES)].copy()
    if perf.empty:
        matrix = pd.DataFrame(index=[], columns=variant_order, dtype=float)
        row_original = []
    else:
        perf["success"] = perf["outcome"].isin({"detects_correctly", "correctly_rejects"}).astype(float)
        perf["original_label"] = perf["scenario"].astype(str) + " | " + perf["failure_mode"].astype(str)
        perf["row_label"] = [
            _short_ablation_row_label(scenario, failure_mode)
            for scenario, failure_mode in zip(perf["scenario"], perf["failure_mode"])
        ]
        if group_by == "scenario":
            perf["row_sort"] = perf["scenario"].astype(str) + " | " + perf["failure_mode"].astype(str)
        elif group_by == "evidence_layer":
            perf["row_sort"] = perf["failure_mode"].astype(str) + " | " + perf["scenario"].astype(str)
        else:
            perf["row_sort"] = perf["row_label"].astype(str)
        job_rate = (
            perf.groupby(["job_id", "row_label", "original_label", "row_sort", "variant"], dropna=False)["success"]
            .agg("mean" if normalize else "sum")
            .reset_index(name="success")
        )
        matrix = job_rate.pivot_table(
            index=["row_sort", "row_label", "original_label"],
            columns="variant",
            values="success",
            aggfunc="mean" if normalize else "sum",
        )
        matrix = matrix.sort_index(level="row_sort").reindex(columns=variant_order)
        row_original = list(matrix.index.get_level_values("original_label"))
        matrix.index = matrix.index.get_level_values("row_label")

    max_rows_per_page = max(1, int(max_rows_per_page))
    page_count = max(1, int(np.ceil(max(1, matrix.shape[0]) / max_rows_per_page)))
    figures = []
    paths: list[str] = []
    page_metadata = []
    save_path = None if save_path is None else Path(save_path)

    def _page_path(path: Optional[Path], page: int) -> Optional[Path]:
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if page_count == 1:
            return path
        return path.with_name(f"{path.stem}_page{page + 1:02d}{path.suffix}")

    for page in range(page_count):
        start = page * max_rows_per_page
        stop = min(start + max_rows_per_page, matrix.shape[0])
        page_matrix = matrix.iloc[start:stop].copy() if not matrix.empty else matrix.copy()
        page_original = row_original[start:stop] if row_original else []
        page_display = _display_labels([str(label) for label in page_matrix.index], wrap_width=wrap_width)
        page_widths = _auto_panel_widths(
            performance_labels=[str(label) for label in page_matrix.index],
            support_labels=[],
            wrap_width=wrap_width,
        )
        if figsize == "auto" or figsize is None:
            page_figsize = (
                page_widths["performance"] + 1.05,
                max(float(min_height), 1.30 + float(row_height) * max(4, page_matrix.shape[0])),
            )
        else:
            page_figsize = figsize
        fig = plt.figure(figsize=page_figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.07])
        ax = fig.add_subplot(grid[0, 0])
        cax = fig.add_subplot(grid[1, 0])
        if page_matrix.empty:
            ax.text(0.5, 0.5, "No performance-applicable rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            cax.axis("off")
        else:
            values = page_matrix.to_numpy(dtype=float)
            masked = np.ma.masked_invalid(values)
            cmap_obj = plt.get_cmap("YlGnBu").copy()
            cmap_obj.set_bad(color="#ffffff")
            im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=0.0, vmax=1.0)
            ax.set_xticks(np.arange(page_matrix.shape[1]))
            ax.set_xticklabels([_VARIANT_SHORT_LABELS.get(str(x), str(x)[:1]) for x in page_matrix.columns], fontsize=8)
            ax.set_yticks(np.arange(page_matrix.shape[0]))
            ax.set_yticklabels(page_display, fontsize=7)
            n_cells = int(page_matrix.shape[0] * page_matrix.shape[1])
            draw_values = bool(show_values) if isinstance(show_values, bool) else n_cells <= 125
            if draw_values:
                font_size = _heatmap_annotation_font_size(page_matrix.shape[0], page_matrix.shape[1])
                for i in range(page_matrix.shape[0]):
                    for j in range(page_matrix.shape[1]):
                        value = values[i, j]
                        if not np.isfinite(value):
                            continue
                        color = "#ffffff" if value >= 0.55 else "#111111"
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=font_size, color=color)
            ax.set_xticks(np.arange(-0.5, page_matrix.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, page_matrix.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#e0e0e0", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            fig.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
                label="Success rate" if normalize else "Success count",
            )
            cax.tick_params(labelsize=7)
            cax.xaxis.label.set_size(7)
        page_title = title or f"Supplemental ablation detail ({split} seeds)"
        if page_count > 1:
            page_title = f"{page_title} - page {page + 1}/{page_count}"
        ax.set_title(page_title, fontsize=10)
        ax.set_xlabel("Variant", fontsize=8)
        panel_dimensions = _drawn_axis_dimensions(fig, {"supplemental": ax, "colorbar": cax})
        page_path = _page_path(save_path, page)
        if page_path is not None:
            fig.savefig(page_path, bbox_inches="tight")
            paths.append(str(page_path))
        if show:
            plt.show()
        figures.append(fig)
        page_metadata.append(
            {
                "page": page + 1,
                "row_start": int(start),
                "row_stop": int(stop),
                "figure_dimensions": {
                    "width": float(fig.get_size_inches()[0]),
                    "height": float(fig.get_size_inches()[1]),
                },
                "panel_dimensions": panel_dimensions,
                "displayed_labels": {"rows": page_display},
                "original_labels": {"rows": page_original},
                "masked_cell_count": _masked_cell_count(page_matrix),
                "path": None if page_path is None else str(page_path),
            }
        )
    return {
        "figures": figures,
        "paths": paths,
        "metadata": {
            "page_count": page_count,
            "max_rows_per_page": max_rows_per_page,
            "group_by": group_by,
            "pages": page_metadata,
            "masked_cell_count": _masked_cell_count(matrix),
        },
    }


_PROFILE_CONFIGS = {
    "smoke": {
        "n_samples_per_condition": 4,
        "cells_per_sample": 160,
        "n_boot": 25,
        "k_values": (15,),
        "max_exact_cells": 750,
        "default_calibration_seeds": [0],
        "default_evaluation_seeds": [1],
        "approx_cells": "1,000-2,000",
    },
    "quick": {
        "n_samples_per_condition": 4,
        "cells_per_sample": 500,
        "n_boot": 100,
        "k_values": (15, 30),
        "max_exact_cells": 1500,
        "default_calibration_seeds": [0, 1],
        "default_evaluation_seeds": [2, 3, 4],
        "approx_cells": "3,000-5,000",
    },
    "manuscript": {
        "n_samples_per_condition": 4,
        "cells_per_sample": 800,
        "n_boot": 300,
        "k_values": (15, 30, 50),
        "max_exact_cells": 3000,
        "default_calibration_seeds": list(range(5)),
        "default_evaluation_seeds": list(range(5, 20)),
        "approx_cells": "5,000-10,000",
    },
}


def _profile_config(profile: str) -> dict[str, Any]:
    profile = str(profile)
    if profile not in _PROFILE_CONFIGS:
        allowed = ", ".join(sorted(_PROFILE_CONFIGS))
        raise ValueError(f"profile must be one of {{{allowed}}}, got {profile!r}")
    return dict(_PROFILE_CONFIGS[profile])


def _normalize_scenarios(scenarios) -> list[str]:
    if scenarios is None:
        return sorted(_SCENARIOS)
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    out = [_check_scenario(str(scenario)) for scenario in scenarios]
    if not out:
        raise ValueError("scenarios must contain at least one scenario")
    return out


def _normalize_seeds(profile_cfg: Mapping[str, Any], seeds) -> list[tuple[str, int]]:
    if seeds is None:
        calibration = list(profile_cfg["default_calibration_seeds"])
        evaluation = list(profile_cfg["default_evaluation_seeds"])
    elif isinstance(seeds, Mapping):
        calibration = [int(seed) for seed in seeds.get("calibration", [])]
        evaluation = [int(seed) for seed in seeds.get("evaluation", [])]
    else:
        calibration = []
        evaluation = [int(seed) for seed in seeds]
    jobs = [("calibration", seed) for seed in calibration] + [("evaluation", seed) for seed in evaluation]
    if not jobs:
        raise ValueError("At least one calibration or evaluation seed is required")
    overlap = set(calibration) & set(evaluation)
    if overlap:
        raise ValueError(f"Calibration and evaluation seeds must be disjoint; overlap={sorted(overlap)}")
    return jobs


def _job_id(profile: str, scenario: str, split: str, seed: int) -> str:
    return f"{profile}_{scenario}_{split}_seed{int(seed)}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2), encoding="utf-8")


def _load_completed_job(output_dir: Path, job_id: str) -> Optional[dict[str, pd.DataFrame]]:
    status_path = output_dir / f"{job_id}_status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return None
    tables = {}
    for path in output_dir.glob(f"{job_id}_*.csv"):
        name = path.name[len(job_id) + 1 : -4]
        tables[name] = pd.read_csv(path)
    return tables


def _add_job_columns(tables: dict[str, pd.DataFrame], *, job_id: str, split: str, profile: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name, df in tables.items():
        df = df.copy()
        if "job_id" not in df:
            df.insert(0, "job_id", job_id)
        if "seed_split" not in df:
            df.insert(1, "seed_split", split)
        if "profile" not in df:
            df.insert(2, "profile", profile)
        out[name] = df
    return out


def _run_scgeo_stack(adata, *, profile_cfg: Mapping[str, Any]) -> None:
    import scgeo as sg

    truth = _require_truth(adata)
    consensus_reps = list(
        truth.get(
            "consensus_representations",
            list(truth["equivalent_representations"]) + list(truth.get("corrupted_representations", [])),
        )
    )
    local_geometry_reps = list(truth.get("representations", consensus_reps))
    velocity_keys = truth.get("velocity_keys")
    if velocity_keys is not None:
        velocity_keys = {rep: velocity_keys.get(rep) for rep in consensus_reps}
    n_boot = int(profile_cfg["n_boot"])
    sg.tl.robust_shift(
        adata,
        rep="X_truth",
        condition_key=truth["condition_key"],
        group0=truth["group0"],
        group1=truth["group1"],
        by=truth["state_key"],
        sample_key=truth["sample_key"],
        center="geometric_median",
        n_boot=n_boot,
        seed=int(truth["seed"]),
        store_key="robust_shift",
    )
    sg.tl.representation_stability(
        adata,
        reps=consensus_reps,
        node_key=truth["state_key"],
        condition_key=truth["condition_key"],
        group0=truth["group0"],
        group1=truth["group1"],
        sample_key=truth["sample_key"],
        center="geometric_median",
        n_boot=n_boot,
        velocity_keys=velocity_keys,
        min_cells=10,
        seed=int(truth["seed"]),
        store_key="representation_stability",
    )
    sg.tl.local_geometry_stability(
        adata,
        reps=local_geometry_reps,
        node_key=truth["state_key"],
        sample_key=truth["sample_key"],
        k_values=tuple(profile_cfg["k_values"]),
        n_boot=max(0, min(n_boot, 50)),
        max_exact_cells=int(profile_cfg["max_exact_cells"]),
        seed=int(truth["seed"]),
        store_key="local_geometry_stability",
    )


def _run_one_job(job: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(job["output_dir"]) if job.get("output_dir") is not None else None
    job_id = str(job["job_id"])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / f"{job_id}_config.json", job)
        if job.get("resume"):
            loaded = _load_completed_job(output_dir, job_id)
            if loaded is not None:
                return {"job_id": job_id, "status": "resumed", "tables": loaded, "error": ""}
    start = time.perf_counter()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        adata = simulate_perturbation_geometry(
            scenario=job["scenario"],
            n_samples_per_condition=int(job["profile_cfg"]["n_samples_per_condition"]),
            cells_per_sample=int(job["profile_cfg"]["cells_per_sample"]),
            seed=int(job["seed"]),
            velocity_mode=job.get("velocity_mode"),
        )
        _run_scgeo_stack(adata, profile_cfg=job["profile_cfg"])
        elapsed = time.perf_counter() - start
        peak_mem = max(start_mem, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        adata.uns["benchmark_runtime"] = {
            "runtime_seconds": float(elapsed),
            "peak_memory_mb": float(peak_mem / 1024.0),
        }
        tables = evaluate_ground_truth(adata)
        runtime = pd.DataFrame(
            [
                {
                    "scenario": job["scenario"],
                    "seed": int(job["seed"]),
                    "runtime_seconds": float(elapsed),
                    "peak_memory_mb": float(peak_mem / 1024.0),
                    "n_obs": int(adata.n_obs),
                    "n_vars": int(adata.n_vars),
                }
            ]
        )
        tables["runtime_summary"] = runtime
        tables = _add_job_columns(tables, job_id=job_id, split=job["seed_split"], profile=job["profile"])
        if output_dir is not None:
            for name, df in tables.items():
                df.to_csv(output_dir / f"{job_id}_{name}.csv", index=False)
            _write_json(
                output_dir / f"{job_id}_status.json",
                {"job_id": job_id, "status": "completed", "runtime_seconds": float(elapsed)},
            )
        return {"job_id": job_id, "status": "completed", "tables": tables, "error": ""}
    except Exception as exc:  # pragma: no cover - exercised in integration failure cases
        elapsed = time.perf_counter() - start
        error = f"{type(exc).__name__}: {exc}"
        if output_dir is not None:
            _write_json(
                output_dir / f"{job_id}_status.json",
                {"job_id": job_id, "status": "failed", "runtime_seconds": float(elapsed), "error": error},
            )
        return {"job_id": job_id, "status": "failed", "tables": {}, "error": error}


def _concat_tables(results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    by_name: dict[str, list[pd.DataFrame]] = {}
    for result in results:
        for name, df in result.get("tables", {}).items():
            by_name.setdefault(name, []).append(df)
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in by_name.items()
    }


def run_simulation_suite(
    *,
    profile: str = "smoke",
    scenarios=None,
    seeds=None,
    output_dir=None,
    resume: bool = True,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Run a reproducible synthetic ScGeo benchmark suite.

    Calibration and evaluation seeds are kept separate in all result tables.
    The suite does not alter consensus thresholds automatically; threshold
    sensitivity tables are exported for downstream calibration analyses. Final
    ablation summaries aggregate outcomes within simulation seed/job before
    summarizing across held-out jobs.
    """
    profile = str(profile)
    profile_cfg = _profile_config(profile)
    scenario_list = _normalize_scenarios(scenarios)
    seed_jobs = _normalize_seeds(profile_cfg, seeds)
    n_jobs = max(1, int(n_jobs))
    output_path = None if output_dir is None else Path(output_dir)
    jobs = []
    for scenario in scenario_list:
        velocity_mode = None
        if scenario == "aligned_dynamics":
            velocity_mode = "aligned"
        elif scenario == "discordant_dynamics":
            velocity_mode = "discordant"
        for split, seed in seed_jobs:
            jid = _job_id(profile, scenario, split, seed)
            jobs.append(
                {
                    "job_id": jid,
                    "profile": profile,
                    "scenario": scenario,
                    "seed_split": split,
                    "seed": int(seed),
                    "profile_cfg": dict(profile_cfg),
                    "velocity_mode": velocity_mode,
                    "output_dir": None if output_path is None else str(output_path),
                    "resume": bool(resume),
                }
            )

    approx_cells = 2 * int(profile_cfg["n_samples_per_condition"]) * int(profile_cfg["cells_per_sample"])
    print(
        "ScGeo simulation suite dry run: "
        f"profile={profile}, scenarios={len(scenario_list)}, jobs={len(jobs)}, "
        f"approx_cells_per_job={approx_cells} ({profile_cfg['approx_cells']} target), "
        f"n_boot={profile_cfg['n_boot']}, k={tuple(profile_cfg['k_values'])}, "
        f"max_exact_cells={profile_cfg['max_exact_cells']}, n_jobs={n_jobs}, "
        f"output_dir={output_path if output_path is not None else 'not saved'}"
    )

    results: list[dict[str, Any]] = []
    if n_jobs == 1:
        for job in jobs:
            results.append(_run_one_job(job))
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_run_one_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    tables = _concat_tables(results)
    if tables:
        ablation = framework_ablation(tables)
        tables["framework_ablation"] = ablation
        tables["framework_ablation_seed_summary"] = _seed_level_framework_ablation(ablation, split="evaluation")
        tables["framework_ablation_summary"] = _summarize_framework_ablation(ablation, split="evaluation")
    jobs_df = pd.DataFrame(
        [
            {
                "job_id": result["job_id"],
                "status": result["status"],
                "error": result.get("error", ""),
            }
            for result in results
        ]
    )
    errors = jobs_df[jobs_df["status"] == "failed"].copy()
    ablation_figure_paths: list[str] = []
    if output_path is not None and "framework_ablation" in tables:
        tables["framework_ablation"].to_csv(output_path / f"{profile}_framework_ablation.csv", index=False)
        tables["framework_ablation_seed_summary"].to_csv(
            output_path / f"{profile}_framework_ablation_seed_summary.csv",
            index=False,
        )
        tables["framework_ablation_summary"].to_csv(output_path / f"{profile}_framework_ablation_summary.csv", index=False)
        if not tables["framework_ablation_summary"].empty:
            for suffix in ("png", "svg"):
                fig_path = output_path / f"{profile}_framework_ablation_summary.{suffix}"
                fig = plot_framework_ablation(
                    tables["framework_ablation"],
                    split="evaluation",
                    save_path=fig_path,
                    show=False,
                    title=f"ScGeo framework ablation ({profile}, held-out evaluation seeds)",
                )
                _import_matplotlib_pyplot().close(fig)
                ablation_figure_paths.append(str(fig_path))
                supplemental_path = output_path / f"{profile}_framework_ablation_supplemental.{suffix}"
                supplemental = _plot_framework_ablation_supplemental(
                    tables["framework_ablation"],
                    split="evaluation",
                    save_path=supplemental_path,
                    show=False,
                    title=f"ScGeo framework ablation supplemental ({profile}, held-out evaluation seeds)",
                )
                for supplemental_fig in supplemental["figures"]:
                    _import_matplotlib_pyplot().close(supplemental_fig)
                ablation_figure_paths.extend(supplemental["paths"])
    out = {
        "profile": profile,
        "profile_config": profile_cfg,
        "jobs": jobs_df,
        "tables": tables,
        "errors": errors,
        "calibration_seeds": [seed for split, seed in seed_jobs if split == "calibration"],
        "evaluation_seeds": [seed for split, seed in seed_jobs if split == "evaluation"],
        "ablation_figure_paths": ablation_figure_paths,
    }
    if output_path is not None:
        jobs_df.to_csv(output_path / f"{profile}_jobs.csv", index=False)
        _write_json(
            output_path / f"{profile}_suite_summary.json",
            {
                "profile": profile,
                "scenarios": scenario_list,
                "calibration_seeds": out["calibration_seeds"],
                "evaluation_seeds": out["evaluation_seeds"],
                "n_jobs": len(jobs),
                "n_failed": int((jobs_df["status"] == "failed").sum()),
            },
        )
    return out
