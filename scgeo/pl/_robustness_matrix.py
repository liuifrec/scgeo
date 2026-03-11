from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        if row_key in data.columns and col_key in data.columns and value_key in data.columns:
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
) -> pd.Series:
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
            raise ValueError("pass_threshold must be provided when summary='pass_rate'.")
        return (value_df >= float(pass_threshold)).mean(axis=1, skipna=True)
    raise ValueError("summary must be one of {'mean', 'median', 'min', 'max', 'pass_rate'}.")


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
    return_data: bool = False,
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
    center
        Optional visual midpoint for diverging data. If provided, color scaling
        is made symmetric around this value when vmin/vmax are not supplied.
    """
    value_df, annot_df = _coerce_matrix_df(
        data,
        row_key=row_key,
        col_key=col_key,
        value_key=value_key,
        annot_key=annot_key,
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
        )

    if sort_rows_by is not None:
        if sort_rows_by == "summary":
            if row_summary is None:
                raise ValueError("sort_rows_by='summary' requires summary to be enabled.")
            order = row_summary.sort_values(ascending=ascending).index
        elif sort_rows_by == "mean":
            order = value_df.mean(axis=1, skipna=True).sort_values(ascending=ascending).index
        elif sort_rows_by == "median":
            order = value_df.median(axis=1, skipna=True).sort_values(ascending=ascending).index
        elif sort_rows_by == "min":
            order = value_df.min(axis=1, skipna=True).sort_values(ascending=ascending).index
        elif sort_rows_by == "max":
            order = value_df.max(axis=1, skipna=True).sort_values(ascending=ascending).index
        else:
            raise ValueError("sort_rows_by must be one of {'summary', 'mean', 'median', 'min', 'max'}.")

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
        span = max(abs(float(np.nanmin(arr)) - center), abs(float(np.nanmax(arr)) - center))
        vmin = center - span
        vmax = center + span
    elif center is not None:
        span = max(abs(vmin - center), abs(vmax - center))
        vmin = center - span
        vmax = center + span

    cmap_obj = plt.get_cmap(cmap).copy()
    try:
        cmap_obj.set_bad(color=na_color)
    except Exception:
        pass

    if summary is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax_summary = None
    else:
        fig, (ax, ax_summary) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [4.0, 1.1]},
        )

    im = ax.imshow(arr, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

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
                        color="black",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label is None:
        cbar_label = value_key if value_key in getattr(data, "columns", []) else "value"
    cbar.set_label(cbar_label)

    if ax_summary is not None and row_summary is not None:
        y = np.arange(len(row_summary))
        ax_summary.barh(y, row_summary.to_numpy(dtype=float), color="dimgray", alpha=0.9)
        ax_summary.set_yticks(y)
        ax_summary.set_yticklabels([])
        ax_summary.invert_yaxis()
        ax_summary.grid(axis="x", alpha=0.25)
        for spine in ("top", "right", "left"):
            ax_summary.spines[spine].set_visible(False)

        if summary_label is None:
            summary_label = summary
        ax_summary.set_xlabel(summary_label)

    if title is None:
        title = "ScGeo: robustness matrix"
    ax.set_title(title)

    fig.tight_layout()

    if show:
        plt.show()

    if return_data:
        out = {
            "matrix": value_df,
            "annotation_matrix": annot_df,
            "row_summary": row_summary,
            "vmin": vmin,
            "vmax": vmax,
        }
        if summary is None:
            return fig, ax, out
        return fig, (ax, ax_summary), out

    if summary is None:
        return fig, ax
    return fig, (ax, ax_summary)