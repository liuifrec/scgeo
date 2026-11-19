from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._utils import _get_ax
from ._result import PlotResult
from ._table_plot_utils import numeric_column


def _get_dt_table(adata, key: str):
    if "scgeo" not in adata.uns:
        raise KeyError("adata.uns['scgeo'] not found")
    obj = adata.uns["scgeo"].get(key)
    if obj is None:
        raise KeyError(f"adata.uns['scgeo']['{key}'] not found")

    # common patterns
    if isinstance(obj, dict) and "table" in obj:
        obj = obj["table"]

    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame(obj)


def distribution_test_volcano(
    adata,
    *,
    key: str = "distribution_test",
    effect_col: str = "effect",
    p_col: str = "p_adj",
    label_col: str = "group",
    top_n_labels: int = 10,
    ax=None,
    figsize=(6.0, 5.0),
    title: str | None = None,
    show: bool = True,
    return_result: bool = False,
    p_threshold: float | None = 0.05,
):
    """
    Volcano-like plot for distribution/comparison results.

    x = effect (distance / signed effect)
    y = -log10(p_adj)

    Requires a table with at least: effect_col, p_col.
    Optionally label the top hits by p-value.

    Missing values remain unavailable. Finite probabilities must lie in [0, 1];
    only values below 1e-300 are clipped for the log display. ``p_threshold``
    controls the descriptive reference line (None disables it), not a new test.
    The legacy return is ``(fig, ax, source_table)``. ``return_result=True``
    provides original rows, exact plotted coordinates, labels and omissions.
    """
    df = _get_dt_table(adata, key=key)

    if effect_col not in df.columns:
        raise KeyError(f"effect_col '{effect_col}' not found in table columns: {list(df.columns)}")
    if p_col not in df.columns:
        raise KeyError(f"p_col '{p_col}' not found in table columns: {list(df.columns)}")

    if isinstance(top_n_labels, bool) or not isinstance(top_n_labels, (int, np.integer)) or top_n_labels < 0:
        raise ValueError("top_n_labels must be a nonnegative integer.")
    if p_threshold is not None and (isinstance(p_threshold, bool) or not np.isfinite(p_threshold)
                                    or not 0 < p_threshold <= 1):
        raise ValueError("p_threshold must lie in (0, 1] or be None.")
    x = numeric_column(df, effect_col)
    p = numeric_column(df, p_col, probability=True)
    y = -np.log10(np.maximum(p, 1e-300))
    displayed = np.isfinite(x) & np.isfinite(y)
    labelled = np.zeros(len(df), dtype=bool)
    owns_figure = ax is None

    fig, ax = _get_ax(ax=ax, figsize=figsize)
    artists = {"points": ax.scatter(x[displayed], y[displayed], s=18, alpha=0.8,
                                    linewidths=0, color="#527F9A")}

    # label top hits by p (smallest p => largest y)
    if label_col in df.columns and top_n_labels > 0:
        eligible = np.flatnonzero(displayed)
        order = eligible[np.argsort(p[eligible], kind="stable")][:top_n_labels]
        labelled[order] = True
        artists["labels"] = [ax.annotate(str(df.iloc[i][label_col]), (x[i], y[i]),
                                         xytext=(4, 3), textcoords="offset points", fontsize=8)
                             for i in order]

    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")
    ax.set_title(title or f"Volcano: {key}")
    if p_threshold is not None:
        artists["threshold"] = ax.axhline(-np.log10(p_threshold), linestyle="--", linewidth=1)
    notes = []
    if not displayed.all():
        note = f"{int((~displayed).sum())} rows lack a finite effect/p pair; omitted."
        ax.text(.02, .98, note, transform=ax.transAxes, va="top", fontsize=8, color="#687680")
        notes.append(note)
    if (p < 1e-300).any():
        notes.append("p-values below 1e-300 use that floor for the log display only.")

    if show and owns_figure:
        plt.show()

    if return_result:
        points = pd.DataFrame({"source_row": np.arange(len(df)), "effect": x,
                               "p_value": p, "neglog10_p": y, "displayed": displayed,
                               "labelled": labelled, "p_clipped_for_display": p < 1e-300})
        if label_col in df:
            points["label"] = df[label_col].to_numpy()
        return PlotResult(fig, {"volcano": ax}, {"source": df, "points": points},
                          {"plot": "distribution_test_volcano", "key": key,
                           "effect_col": effect_col, "p_col": p_col, "label_col": label_col,
                           "top_n_labels": top_n_labels, "p_threshold": p_threshold,
                           "p_display_floor": 1e-300, "analysis_recomputed": False},
                          warnings=notes, artists=artists)
    return fig, ax, df
