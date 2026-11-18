from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ._result import PlotResult
from ._table_plot_utils import numeric_column
from ._utils import _get_ax


def delta_rank(
    adata,
    store_key: str = "scgeo",
    kind: str = "shift",
    level: str = "by",
    *,
    top_n: int | None = None,
    rotate_xticks: int = 60,
    ax=None,
    figsize=None,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = False,
):
    """
    Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.

    Notes
    -----
    Uses sg.get.table(...). Expected to contain:
      - delta_norm column
      - some label column identifying the group (often 'label', or sometimes `level`)

    Supplied axes retain their layout. ``return_result=True`` exports original
    and ranked rows; ``return_data=True`` returns ``(ax, ranked_table)``.
    The default return remains an Axes and does not call ``plt.show``.
    """
    sg = __import__("scgeo")
    df = sg.get.table(adata, store_key=store_key, kind=kind, level=level)
    source = df.copy(deep=True)

    if "delta_norm" not in df.columns:
        raise KeyError(f"delta_rank expects 'delta_norm' in table; got columns={list(df.columns)}")

    values = numeric_column(df, "delta_norm")
    if (values < 0).any():
        raise ValueError("delta_norm magnitudes must be nonnegative or missing.")
    df = df.assign(source_row=np.arange(len(df)), delta_norm=values)
    df = df.sort_values("delta_norm", ascending=False, kind="stable", na_position="last")

    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 0:
            raise ValueError("top_n must be a nonnegative integer or None.")
        df = df.head(int(top_n))

    # Pick a readable label column for x-axis
    label_col = None
    for cand in ("label", level, "group", "cluster", "name"):
        if cand in df.columns:
            label_col = cand
            break

    if label_col is not None:
        x_labels = df[label_col].astype(str).tolist()
    else:
        # fallback: use index if the table already sets it meaningfully
        x_labels = [str(i) for i in df.index.tolist()]

    y = df["delta_norm"].to_numpy()

    owns_figure = ax is None
    fig, ax = _get_ax(ax=ax, figsize=figsize)
    line, = ax.plot(range(len(df)), y, marker="o", linestyle="-")
    df = df.assign(rank=np.arange(1, len(df) + 1), displayed=np.isfinite(y))

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right")

    ax.set_xlabel(label_col if label_col is not None else level)
    ax.set_ylabel("||Δ||")
    ax.set_title("ScGeo: delta magnitude ranking")
    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    if return_result:
        return PlotResult(fig, {"rank": ax}, {"source": source, "ranked": df},
                          {"plot": "delta_rank", "store_key": store_key, "kind": kind,
                           "level": level, "top_n": top_n, "analysis_recomputed": False},
                          artists={"rank": line})
    if return_data:
        return ax, df
    return ax
