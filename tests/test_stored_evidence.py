"""Source identity is checked without inventing provenance or analysis."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scgeo.pl._stored_evidence import validate_stored_evidence


@pytest.fixture
def source_tables():
    shift = pd.DataFrame({"state": ["A", "B"], "delta_norm": [1.0, 2.0]})
    align = pd.DataFrame({"node": ["B", "A"], "alignment_cosine": [0.3, -0.5]})
    metadata = {
        "rep": "X_pca",
        "group0": "control",
        "group1": "treated",
        "condition_key": "condition",
        "node_key": "state",
        "center": "geometric_median",
        "n_dimensions": 30,
        "scale": "log1p_scaled",
        "sample_key": "mouse",
        "sample_weighting": "equal_sample",
        "scope": "state",
        "reference": "control",
        "graph_mode": "pooled",
    }
    shift.attrs["provenance"] = deepcopy(metadata)
    align.attrs["params"] = deepcopy(metadata)
    return shift, align


@pytest.mark.parametrize(
    "field,value",
    [
        ("rep", "X_other"),
        ("group0", "other"),
        ("group1", "other"),
        ("condition_key", "batch"),
        ("node_key", "substate"),
        ("center", "mean"),
        ("n_dimensions", 20),
        ("scale", "raw"),
        ("sample_key", "animal"),
        ("sample_weighting", "pooled_cells"),
        ("scope", "global"),
        ("reference", "pooled"),
        ("graph_mode", "within_condition"),
    ],
)
def test_recorded_contradictions_between_sources_raise(source_tables, field, value):
    shift, align = source_tables
    align.attrs["params"][field] = value
    with pytest.raises(ValueError, match="Conflicting stored provenance"):
        validate_stored_evidence(shift, align)


def test_column_attrs_and_declarations_all_participate(source_tables):
    shift, align = source_tables
    shift["rep"] = ["X_pca", "X_other"]
    with pytest.raises(ValueError, match="analysis_rep"):
        validate_stored_evidence(shift, align)
    shift["rep"] = "X_other"
    with pytest.raises(ValueError, match="analysis_rep"):
        validate_stored_evidence(shift, align)
    shift["rep"] = "X_pca"
    with pytest.raises(ValueError, match="group0"):
        validate_stored_evidence(shift, align, group0="treated", group1="control")


def test_state_report_nested_provenance_and_representation_params(source_tables):
    shift, align = source_tables
    shift.attrs = {
        "provenance": {
            "effect_representation": "X_pca",
            "representations": ["X_pca", "X_latent"],
            "group0": "control",
            "group1": "treated",
            "center": "geometric_median",
            "condition_key": "condition",
            "sample_key": "mouse",
        }
    }
    align["rep"] = "X_pca"
    align.attrs = {
        "params": {
            "reps": ["X_pca", "X_latent"],
            "group0": "control",
            "group1": "treated",
            "center": "geometric_median",
            "condition_key": "condition",
            "sample_key": "mouse",
        }
    }
    before_shift, before_align = deepcopy(shift), deepcopy(align)
    provenance, _ = validate_stored_evidence(
        shift, align, analysis_rep="X_pca", strict=True
    )
    assert provenance["status"] == "recorded_consistent"
    assert provenance["analysis_rep"] == "X_pca"
    assert provenance["n_dimensions"] is None
    assert provenance["sample_weighting"] is None
    assert provenance["velocity_weighting"] is None
    pd.testing.assert_frame_equal(shift, before_shift)
    pd.testing.assert_frame_equal(align, before_align)
    assert shift.attrs == before_shift.attrs and align.attrs == before_align.attrs
    json.dumps(provenance, allow_nan=False)


def test_unknown_sources_and_declarations_remain_unverified():
    shift = pd.DataFrame({"state": ["A"]})
    align = pd.DataFrame({"state": ["A"]})
    unknown, warnings = validate_stored_evidence(shift, align)
    assert unknown["status"] == "unverified"
    assert set(unknown["missing_fields"]) == {"analysis_rep", "group0", "group1"}
    assert warnings
    with pytest.raises(ValueError, match="missing"):
        validate_stored_evidence(shift, align, analysis_rep="X_pca", strict=True)
    declared, warnings = validate_stored_evidence(
        shift,
        align,
        analysis_rep="X_pca",
        group0="control",
        group1="treated",
        strict=True,
    )
    assert declared["status"] == "declared_unverified"
    assert declared["field_status"]["analysis_rep"] == "declared_unverified"
    assert declared["table_provenance"] == {"shift_df": {}, "alignment_df": {}}
    assert (
        declared["sample_weighting"] is None and declared["velocity_weighting"] is None
    )
    assert warnings


def test_one_recorded_source_does_not_certify_the_other(source_tables):
    shift, align = source_tables
    align.attrs = {}
    provenance, _ = validate_stored_evidence(
        shift,
        align,
        analysis_rep="X_pca",
        group0="control",
        group1="treated",
        strict=True,
    )
    assert provenance["status"] == "partially_recorded"
    assert provenance["field_status"]["analysis_rep"] == "partially_recorded"
    assert provenance["table_provenance"]["alignment_df"] == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("group0", " control "),
        ("group1", "treated "),
        ("node_key", " state"),
        ("condition_key", "condition "),
        ("analysis_rep", " X_pca"),
    ],
)
def test_identifier_whitespace_is_not_silently_normalized(source_tables, field, value):
    shift, align = source_tables
    with pytest.raises(ValueError, match="Conflicting stored provenance"):
        validate_stored_evidence(shift, align, **{field: value})


def test_velocity_and_displacement_weighting_are_distinct(source_tables):
    shift, align = source_tables
    align.attrs["params"]["velocity_weighting"] = "pooled_cells"
    align.attrs["params"]["velocity_aggregation"] = "cell_mean"
    align.attrs["params"]["velocity_scope"] = "contrast"
    provenance, _ = validate_stored_evidence(shift, align, strict=True)
    assert provenance["sample_weighting"] == "equal_sample"
    assert provenance["velocity_weighting"] == "pooled_cells"
    assert provenance["velocity_aggregation"] == "cell_mean"
    assert provenance["velocity_scope"] == "contrast"
    shift.attrs["provenance"]["velocity_weighting"] = "equal_sample"
    with pytest.raises(ValueError, match="velocity_weighting"):
        validate_stored_evidence(shift, align)


def test_stored_metadata_ignores_display_coordinates_and_ensemble_members(
    source_tables,
):
    shift, align = source_tables
    before, _ = validate_stored_evidence(shift, align)
    shift["umap_x"] = [1e7, -1e7]
    shift["umap_y"] = [np.nan, 9.0]
    shift.attrs["display_basis"] = "X_umap_rotated"
    shift.attrs["provenance"]["representations"] = ["X_pca", "X_umap"]
    after, _ = validate_stored_evidence(shift, align)
    assert before == after


def test_display_like_analysis_is_recorded_and_warned_not_banned(source_tables):
    shift, align = source_tables
    shift.attrs["provenance"]["rep"] = "X_umap"
    align.attrs["params"]["rep"] = "X_umap"
    provenance, warnings = validate_stored_evidence(shift, align, strict=True)
    assert provenance["analysis_rep"] == "X_umap"
    assert any("display embedding" in warning for warning in warnings)


@pytest.mark.parametrize("dimension", [0, -2, 2.5, np.inf, True])
def test_invalid_recorded_dimensions_rejected(source_tables, dimension):
    shift, align = source_tables
    shift.attrs["provenance"]["n_dimensions"] = dimension
    with pytest.raises(ValueError):
        validate_stored_evidence(shift, align)


def test_numeric_scale_arrays_are_detached_json_metadata(source_tables):
    shift, align = source_tables
    shift.attrs["provenance"]["scale"] = np.array([1.0, 2.0])
    align.attrs["params"]["scale"] = [1.0, 2.0]
    provenance, _ = validate_stored_evidence(shift, align)
    assert provenance["scale"] == [1.0, 2.0]
    json.dumps(provenance, allow_nan=False)
    provenance["scale"][0] = 99
    assert shift.attrs["provenance"]["scale"][0] == 1
