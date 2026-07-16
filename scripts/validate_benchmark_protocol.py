from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "benchmark" / "manuscript_protocol_v1.json"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scgeo-matplotlib"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(val) for val in value]
    if isinstance(value, list):
        return [_jsonable(val) for val in value]
    return value


def _compare(name: str, frozen: Any, current: Any, errors: list[str]) -> None:
    frozen_json = _jsonable(frozen)
    current_json = _jsonable(current)
    if frozen_json != current_json:
        errors.append(
            f"{name} drifted:\n"
            f"  frozen={json.dumps(frozen_json, sort_keys=True)}\n"
            f"  current={json.dumps(current_json, sort_keys=True)}"
        )


def _current_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return proc.stdout.strip()


def _validate_batch_condition_confounding_exclusion(errors: list[str]) -> None:
    from scgeo.bench import _simulation as sim

    adata = sim.simulate_perturbation_geometry(
        scenario="batch_condition_confounding",
        n_states=4,
        n_samples_per_condition=2,
        cells_per_sample=40,
        seed=101,
    )
    tables = sim.evaluate_ground_truth(adata)
    truth = tables["non_identifiable_truth"]
    if truth.empty or not truth["true_non_identifiable"].astype(bool).all():
        errors.append("batch_condition_confounding did not mark every state non-identifiable")

    metrics = tables["shift_detection"]
    if metrics.empty:
        errors.append("batch_condition_confounding produced no shift_detection table")
        return

    if not (metrics["n_evaluable_states"].astype(int) == 0).all():
        errors.append("batch_condition_confounding has states entering ordinary shift metrics")

    ordinary_counts = metrics[["tp", "fp", "fn"]].fillna(0).astype(int)
    if int(ordinary_counts.to_numpy().sum()) != 0:
        errors.append("batch_condition_confounding contributed ordinary TP/FP/FN counts")

    state_metrics = tables["state_metrics"]
    if not state_metrics.empty and state_metrics["biological_shift_evaluable"].astype(bool).any():
        errors.append("batch_condition_confounding state_metrics contains evaluable biological shifts")


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    import scgeo as sg
    from scgeo.bench import _simulation as sim
    from scgeo.tl import _representation_stability as rs

    errors: list[str] = []

    _compare("package.version", protocol["package"]["version"], sg.__version__, errors)
    _compare(
        "profile.selected",
        protocol["profile"]["selected"],
        "manuscript",
        errors,
    )

    current_profile = sim._profile_config("manuscript")
    frozen_profile = {
        "n_samples_per_condition": protocol["profile"]["n_samples_per_condition"],
        "cells_per_sample": protocol["profile"]["cells_per_sample"],
        "n_boot": protocol["profile"]["n_boot"],
        "k_values": protocol["profile"]["k_values"],
        "max_exact_cells": protocol["profile"]["max_exact_cells"],
        "default_calibration_seeds": protocol["seeds"]["calibration"],
        "default_evaluation_seeds": protocol["seeds"]["evaluation"],
        "approx_cells": protocol["profile"]["approx_cells"],
    }
    _compare("manuscript profile defaults", frozen_profile, current_profile, errors)

    _compare(
        "shift_detection threshold",
        protocol["thresholds"]["shift_detection"],
        sim._PREDECLARED_SHIFT_DETECTION_THRESHOLD,
        errors,
    )
    _compare(
        "zero-effect epsilon",
        protocol["thresholds"]["zero_effect_epsilon"],
        sim._MAGNITUDE_ZERO_EPSILON,
        errors,
    )
    _compare(
        "shift sensitivity thresholds",
        protocol["thresholds"]["shift_sensitivity"],
        sim._SHIFT_SENSITIVITY_THRESHOLDS,
        errors,
    )
    _compare(
        "scenario list",
        protocol["scenarios"],
        sorted(sim._SCENARIOS),
        errors,
    )
    _compare(
        "representation_stability consensus rules",
        protocol["consensus_rules"]["representation_stability"],
        rs._CONSENSUS_LABEL_RULES,
        errors,
    )

    calibration = set(protocol["seeds"]["calibration"])
    evaluation = set(protocol["seeds"]["evaluation"])
    overlap = sorted(calibration & evaluation)
    if overlap:
        errors.append(f"calibration and evaluation seeds overlap: {overlap}")

    invalid = []
    for scenario in protocol["scenarios"]:
        try:
            sim._check_scenario(scenario)
        except ValueError:
            invalid.append(scenario)
    if invalid:
        errors.append(f"invalid scenario names: {invalid}")

    expected_methods = [
        "shift_mean",
        "robust_mean",
        "robust_median",
        "robust_trimmed_mean",
        "robust_geometric_median",
    ]
    frozen_methods = [entry["method"] for entry in protocol["estimators"]]
    _compare("estimator methods", frozen_methods, expected_methods, errors)

    stack = protocol["benchmark_stack"]
    if int(stack["robust_shift"]["n_boot"]) != int(current_profile["n_boot"]):
        errors.append("benchmark_stack.robust_shift.n_boot does not match manuscript profile n_boot")
    if int(stack["representation_stability"]["n_boot"]) != int(current_profile["n_boot"]):
        errors.append("benchmark_stack.representation_stability.n_boot does not match manuscript profile n_boot")
    expected_local_boot = min(int(current_profile["n_boot"]), 50)
    if int(stack["local_geometry_stability"]["n_boot"]) != expected_local_boot:
        errors.append("benchmark_stack.local_geometry_stability.n_boot does not match current cap")

    _validate_batch_condition_confounding_exclusion(errors)
    return errors


def _print_summary(protocol: dict[str, Any]) -> None:
    frozen_commit = protocol.get("git_commit", "unavailable")
    current_commit = _current_git_commit()
    profile = protocol["profile"]
    seeds = protocol["seeds"]
    print("Benchmark protocol validation passed.")
    print(f"protocol: {protocol['protocol_name']}")
    print(f"date_frozen: {protocol['date_frozen']}")
    print(f"frozen_commit: {frozen_commit}")
    print(f"current_commit: {current_commit}")
    print(f"package_version: {protocol['package']['version']}")
    print(
        "profile: "
        f"{profile['selected']} "
        f"(samples/condition={profile['n_samples_per_condition']}, "
        f"cells/sample={profile['cells_per_sample']}, "
        f"n_boot={profile['n_boot']}, "
        f"k={tuple(profile['k_values'])})"
    )
    print(
        "seeds: "
        f"calibration={len(seeds['calibration'])} "
        f"evaluation={len(seeds['evaluation'])}"
    )
    print(f"scenarios: {len(protocol['scenarios'])}")
    print(f"shift_threshold: {protocol['thresholds']['shift_detection']}")
    print(f"zero_effect_epsilon: {protocol['thresholds']['zero_effect_epsilon']}")
    print("post_evaluation_tuning: forbidden by protocol")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen ScGeo benchmark protocol.")
    parser.add_argument(
        "protocol",
        nargs="?",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="Path to the frozen benchmark protocol JSON.",
    )
    args = parser.parse_args(argv)

    protocol = _load_json(args.protocol)
    errors = validate_protocol(protocol)
    if errors:
        print("Benchmark protocol validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    _print_summary(protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
