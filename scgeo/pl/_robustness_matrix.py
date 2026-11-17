from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _coerce_matrix_df(
    data,
    *,
    row_key: str = "feature",
    col_key: str = "setting",
    value_key: str = "value",
    annot_key: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Accept either:
    1) a tidy DataFrame with row/col/value columns
    2) a wide DataFrame with row labels in index and columns as settings

    Returns
    -------
    value_df, annot_df
    """
    if isinstance(data, pd.DataFrame):
        if (
            row_key in data.columns
            and col_key in data.columns
            and value_key in data.columns
        ):
            value_df = data.pivot(index=row_key, columns=col_key, values=value_key)
            annot_df = None
            if annot_key is not None and annot_key in data.columns:
                annot_df = data.pivot(index=row_key, columns=col_key, values=annot_key)
            return value_df, annot_df

        # assume already wide
        value_df = data.copy()
        annot_df = None
        return value_df, annot_df

    raise TypeError("data must be a pandas DataFrame.")


def _compute_row_summary(
    value_df: pd.DataFrame,
    *,
    summary: str = "mean",
    pass_threshold: Optional[float] = None,
    missing_policy: str = "exclude",
) -> pd.Series:
    if missing_policy not in {"exclude", "fail"}:
        raise ValueError("missing_policy must be exclude or fail.")
    value_df = value_df.where(np.isfinite(value_df.to_numpy(dtype=float)))
    if summary == "mean":
        return value_df.mean(axis=1, skipna=True)
    if summary == "median":
        return value_df.median(axis=1, skipna=True)
    if summary == "min":
        return value_df.min(axis=1, skipna=True)
    if summary == "max":
        return value_df.max(axis=1, skipna=True)
    if summary == "pass_rate":
        if pass_threshold is None:
            raise ValueError(
                "pass_threshold must be provided when summary='pass_rate'."
            )
        if not np.isfinite(pass_threshold):
            raise ValueError("pass_threshold must be finite.")
        passed = value_df >= float(pass_threshold)
        if missing_policy == "exclude":
            passed = passed.where(value_df.notna())
        return passed.mean(axis=1, skipna=True)
    raise ValueError(
        "summary must be one of {'mean', 'median', 'min', 'max', 'pass_rate'}."
    )


def _annotation_color(image, value):
    """Choose black or white against the rendered cell's relative luminance."""
    rgba = np.asarray(image.cmap(image.norm(value)), dtype=float)
    axes_rgba = np.asarray(image.axes.get_facecolor(), dtype=float)
    figure_rgb = np.asarray(image.axes.figure.get_facecolor(), dtype=float)[:3]
    background = axes_rgba[:3] * axes_rgba[3] + figure_rgb * (1 - axes_rgba[3])
    rgb = rgba[:3] * rgba[3] + background * (1 - rgba[3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "white" if white_contrast > black_contrast else "black"


def robustness_matrix(
    data: pd.DataFrame,
    *,
    row_key: str = "feature",
    col_key: str = "setting",
    value_key: str = "value",
    annot_key: Optional[str] = None,
    row_order: Optional[Sequence[str]] = None,
    col_order: Optional[Sequence[str]] = None,
    sort_rows_by: Optional[str] = None,
    ascending: bool = False,
    summary: Optional[str] = "mean",
    pass_threshold: Optional[float] = None,
    missing_policy: str = "exclude",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    show_values: bool = True,
    value_fmt: str = ".2f",
    annot_fontsize: int = 8,
    na_color: str = "#d9d9d9",
    grid_lw: float = 0.8,
    grid_color: str = "white",
    cbar_label: Optional[str] = None,
    summary_label: Optional[str] = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: Optional[str] = None,
    axes: Optional[Mapping[str, Any]] = None,
    show_coverage: bool = True,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """
    Plot a robustness heatmap with optional row-summary side bar.

    Parameters
    ----------
    data
        Either a tidy DataFrame with [row_key, col_key, value_key] columns,
        or a wide DataFrame with rows as findings and columns as settings.
    summary
        Optional row summary shown in a right-side bar plot. One of
        {'mean', 'median', 'min', 'max', 'pass_rate'} or None.
    missing_policy
        For pass rates, exclude unavailable settings from the denominator
        (default) or count them as failures with ``"fail"`` (historical behavior).
        Evaluation coverage is always returned separately. Rows without any
        evaluated setting have NaN pass rate under ``"exclude"``.
    axes
        Optional named mapping with ``heatmap`` and, when enabled, ``summary``.
    center
        Optional visual midpoint for diverging data. If provided, color scaling
        is made symmetric around this value when vmin/vmax are not supplied.
    """
    from ._result import PlotResult

    if missing_policy not in {"exclude", "fail"}:
        raise ValueError("missing_policy must be exclude or fail.")
    value_df, annot_df = _coerce_matrix_df(
        data,
        row_key=row_key,
        col_key=col_key,
        value_key=value_key,
        annot_key=annot_key,
    )

    value_df = value_df.astype(float)
    value_df = value_df.where(np.isfinite(value_df.to_numpy()))
    if value_df.empty or not value_df.index.is_unique or not value_df.columns.is_unique:
        raise ValueError(
            "The robustness matrix must have nonempty unique rows and columns."
        )

    if row_order is not None:
        missing = [x for x in row_order if x not in value_df.index]
        if missing:
            raise ValueError(f"row_order contains unknown rows: {missing}")
        value_df = value_df.loc[list(row_order)]
        if annot_df is not None:
            annot_df = annot_df.loc[list(row_order)]

    if col_order is not None:
        missing = [x for x in col_order if x not in value_df.columns]
        if missing:
            raise ValueError(f"col_order contains unknown columns: {missing}")
        value_df = value_df.loc[:, list(col_order)]
        if annot_df is not None:
            annot_df = annot_df.loc[:, list(col_order)]

    row_summary = None
    if summary is not None:
        row_summary = _compute_row_summary(
            value_df,
            summary=summary,
            pass_threshold=pass_threshold,
            missing_policy=missing_policy,
        )

    if sort_rows_by is not None:
        if sort_rows_by == "summary":
            if row_summary is None:
                raise ValueError(
                    "sort_rows_by='summary' requires summary to be enabled."
                )
            order = row_summary.sort_values(ascending=ascending).index
        elif sort_rows_by == "mean":
            order = (
                value_df.mean(axis=1, skipna=True)
                .sort_values(ascending=ascending)
                .index
            )
        elif sort_rows_by == "median":
            order = (
                value_df.median(axis=1, skipna=True)
                .sort_values(ascending=ascending)
                .index
            )
        elif sort_rows_by == "min":
            order = (
                value_df.min(axis=1, skipna=True).sort_values(ascending=ascending).index
            )
        elif sort_rows_by == "max":
            order = (
                value_df.max(axis=1, skipna=True).sort_values(ascending=ascending).index
            )
        else:
            raise ValueError(
                "sort_rows_by must be one of {'summary', 'mean', 'median', 'min', 'max'}."
            )

        value_df = value_df.loc[order]
        if annot_df is not None:
            annot_df = annot_df.loc[order]
        if row_summary is not None:
            row_summary = row_summary.loc[order]

    arr = value_df.to_numpy(dtype=float)

    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("No finite values available to plot.")

    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    if center is not None and vmin is None and vmax is None:
        span = max(
            abs(float(np.nanmin(arr)) - center), abs(float(np.nanmax(arr)) - center)
        )
        vmin = center - span
        vmax = center + span
    elif center is not None:
        span = max(abs(vmin - center), abs(vmax - center))
        vmin = center - span
        vmax = center + span

    cmap_obj = plt.get_cmap(cmap).with_extremes(bad=na_color)

    owned = axes is None
    if axes is not None:
        names = ["heatmap"] + ([] if summary is None else ["summary"])
        if not isinstance(axes, Mapping) or any(name not in axes for name in names):
            raise ValueError(f"axes must contain {names}.")
        ax = axes["heatmap"]
        ax_summary = None if summary is None else axes["summary"]
        fig = ax.figure
        if ax_summary is not None and (
            ax_summary.figure is not fig or ax_summary is ax
        ):
            raise ValueError("The summary requires a distinct axis in the same figure.")
    elif summary is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax_summary = None
    else:
        fig, (ax, ax_summary) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [4.0, 1.1]},
        )

    im = ax.imshow(
        np.ma.masked_invalid(arr), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax
    )

    ax.set_xticks(np.arange(value_df.shape[1]))
    ax.set_xticklabels(value_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(value_df.shape[0]))
    ax.set_yticklabels(value_df.index)

    # grid
    ax.set_xticks(np.arange(-0.5, value_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, value_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_values:
        for i in range(value_df.shape[0]):
            for j in range(value_df.shape[1]):
                val = arr[i, j]
                if np.isnan(val):
                    txt = "" if annot_df is None else str(annot_df.iloc[i, j])
                else:
                    txt = format(val, value_fmt)
                    if annot_df is not None and pd.notna(annot_df.iloc[i, j]):
                        txt = str(annot_df.iloc[i, j])

                if txt:
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=annot_fontsize,
                        color=_annotation_color(im, val),
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label is None:
        cbar_label = value_key if value_key in getattr(data, "columns", []) else "value"
    cbar.set_label(cbar_label)

    n_evaluated = value_df.notna().sum(axis=1)
    coverage = n_evaluated / value_df.shape[1]
    evaluated_pass_rate = None
    if pass_threshold is not None:
        evaluated_pass_rate = _compute_row_summary(
            value_df,
            summary="pass_rate",
            pass_threshold=pass_threshold,
            missing_policy="exclude",
        )

    if ax_summary is not None and row_summary is not None:
        y = np.arange(len(row_summary))
        ax_summary.barh(
            y, row_summary.to_numpy(dtype=float), color="dimgray", alpha=0.9
        )
        ax_summary.set_yticks(y)
        ax_summary.set_yticklabels([])
        ax_summary.invert_yaxis()
        ax_summary.grid(axis="x", alpha=0.25)
        for spine in ("top", "right", "left"):
            ax_summary.spines[spine].set_visible(False)

        if show_coverage:
            for i, n in enumerate(n_evaluated):
                ax_summary.text(
                    1.03,
                    i,
                    f"{n}/{value_df.shape[1]}",
                    transform=ax_summary.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="0.35",
                )
            ax_summary.set_title(
                "Evaluated" if summary != "pass_rate" else "Pass rate / evaluated",
                fontsize=9,
            )
        if summary_label is None:
            summary_label = (
                f"pass rate ({missing_policy})" if summary == "pass_rate" else summary
            )
        if summary == "pass_rate":
            ax_summary.set_xlim(0, 1)
        ax_summary.set_xlabel(summary_label)

    if title is None:
        title = "ScGeo: robustness matrix"
    ax.set_title(title)

    if owned:
        fig.tight_layout()

    if show:
        plt.show()

    out = {
        "matrix": value_df,
        "annotation_matrix": annot_df,
        "row_summary": row_summary,
        "evaluated_pass_rate": evaluated_pass_rate,
        "coverage": coverage,
        "n_evaluated": n_evaluated,
        "n_settings": value_df.shape[1],
        "missing_policy": missing_policy,
        "vmin": vmin,
        "vmax": vmax,
    }
    if return_result:
        named_axes = {"heatmap": ax}
        if ax_summary is not None:
            named_axes["summary"] = ax_summary
        return PlotResult(
            fig=fig,
            axes=named_axes,
            data=out,
            artists={"heatmap": im},
            metadata={
                "plot": "robustness_matrix",
                "summary": summary,
                "pass_threshold": pass_threshold,
                "missing_policy": missing_policy,
                "coverage_definition": "evaluated settings / displayed settings",
                "row_order": value_df.index.tolist(),
                "col_order": value_df.columns.tolist(),
                "cmap": cmap_obj.name,
                "vmin": vmin,
                "vmax": vmax,
                "center": center,
                "na_color": na_color,
                "show_coverage": show_coverage,
            },
        )
    legacy_axes = ax if summary is None else (ax, ax_summary)
    if return_data:
        return fig, legacy_axes, out
    return fig, legacy_axes
