from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from scgeo.pl import robustness_matrix


def _tidy_df():
    return pd.DataFrame(
        {
            "feature": [
                "shift_A", "shift_A", "shift_A",
                "shift_B", "shift_B", "shift_B",
                "ood_C", "ood_C", "ood_C",
            ],
            "setting": [
                "raw", "scanorama", "pca20",
                "raw", "scanorama", "pca20",
                "raw", "scanorama", "pca20",
            ],
            "value": [
                0.91, 0.88, 0.86,
                0.72, 0.75, 0.70,
                0.95, 0.94, 0.89,
            ],
            "label": [
                "0.91", "0.88", "0.86",
                "0.72", "0.75", "0.70",
                "0.95", "0.94", "0.89",
            ],
        }
    )


def test_robustness_matrix_tidy_smoke_return_data():
    df = _tidy_df()

    fig, axes, out = robustness_matrix(
        df,
        row_key="feature",
        col_key="setting",
        value_key="value",
        annot_key="label",
        summary="mean",
        return_data=True,
        show=False,
    )

    ax, ax_summary = axes
    assert fig is not None
    assert ax is not None
    assert ax_summary is not None
    assert "matrix" in out
    assert "row_summary" in out
    assert out["matrix"].shape == (3, 3)
    assert np.isclose(out["row_summary"].loc["shift_A"], (0.91 + 0.88 + 0.86) / 3, atol=1e-8)


def test_robustness_matrix_wide_input():
    wide = pd.DataFrame(
        {
            "raw": [0.9, 0.8],
            "scanorama": [0.85, 0.82],
        },
        index=["feat1", "feat2"],
    )

    fig, ax, out = robustness_matrix(
        wide,
        summary=None,
        return_data=True,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert out["matrix"].shape == (2, 2)


def test_robustness_matrix_pass_rate_summary():
    df = _tidy_df()

    _, _, out = robustness_matrix(
        df,
        row_key="feature",
        col_key="setting",
        value_key="value",
        summary="pass_rate",
        pass_threshold=0.9,
        return_data=True,
        show=False,
    )

    rs = out["row_summary"]
    assert np.isclose(rs.loc["shift_A"], 1 / 3, atol=1e-8)
    assert np.isclose(rs.loc["shift_B"], 0.0, atol=1e-8)
    assert np.isclose(rs.loc["ood_C"], 2 / 3, atol=1e-8)


def test_robustness_matrix_sort_rows_by_summary():
    df = _tidy_df()

    _, _, out = robustness_matrix(
        df,
        row_key="feature",
        col_key="setting",
        value_key="value",
        summary="mean",
        sort_rows_by="summary",
        ascending=False,
        return_data=True,
        show=False,
    )

    rows = list(out["matrix"].index)
    assert rows[0] == "ood_C"


def test_robustness_matrix_invalid_summary_raises():
    df = _tidy_df()

    with pytest.raises(ValueError, match="summary must be one of"):
        robustness_matrix(
            df,
            row_key="feature",
            col_key="setting",
            value_key="value",
            summary="weird_summary",
            show=False,
        )


def test_robustness_matrix_pass_rate_requires_threshold():
    df = _tidy_df()

    with pytest.raises(ValueError, match="pass_threshold"):
        robustness_matrix(
            df,
            row_key="feature",
            col_key="setting",
            value_key="value",
            summary="pass_rate",
            show=False,
        )


def test_robustness_matrix_row_order_unknown_raises():
    df = _tidy_df()

    with pytest.raises(ValueError, match="unknown rows"):
        robustness_matrix(
            df,
            row_key="feature",
            col_key="setting",
            value_key="value",
            row_order=["shift_A", "ghost_row"],
            show=False,
        )


def test_robustness_matrix_col_order_unknown_raises():
    df = _tidy_df()

    with pytest.raises(ValueError, match="unknown columns"):
        robustness_matrix(
            df,
            row_key="feature",
            col_key="setting",
            value_key="value",
            col_order=["raw", "ghost_col"],
            show=False,
        )


def test_robustness_matrix_no_finite_values_raises():
    df = pd.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "setting": ["x", "y", "x", "y"],
            "value": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="No finite values"):
        robustness_matrix(
            df,
            row_key="feature",
            col_key="setting",
            value_key="value",
            show=False,
        )