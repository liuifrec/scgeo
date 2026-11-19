from __future__ import annotations

from typing import Iterable, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._result import PlotResult
from ._table_plot_utils import panel_axes


def legend_from_data(
    legend_data: Iterable[Dict[str, Any]],
    *,
    max_items: int = 20,
    ncol: int = 1,
    fontsize: int = 8,
    markersize: int = 6,
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    show: bool = True,
    ax=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """
    Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

    legend_data items:
      {"group": str, "fraction": float, "count": int, "color": <matplotlib color>}

    ``ax`` accepts a caller-owned legend area. Source exports retain every
    supplied entry, with a displayed flag when ``max_items`` truncates it.
    The default return remains ``(fig, ax)``.
    """
    if isinstance(max_items, bool) or not isinstance(max_items, (int, np.integer)) or max_items < 0:
        raise ValueError("max_items must be a nonnegative integer.")
    if isinstance(ncol, bool) or not isinstance(ncol, (int, np.integer)) or ncol < 1:
        raise ValueError("ncol must be a positive integer.")
    entries = [dict(entry) for entry in legend_data]
    source = pd.DataFrame(entries).copy(deep=True)
    source["source_row"] = np.arange(len(source))
    source["displayed"] = np.arange(len(source)) < max_items
    legend_data = entries[:max_items]

    if figsize is None:
        # heuristic: tiny, scales gently with number of entries
        h = max(1.2, 0.25 * len(legend_data))
        figsize = (2.8 if ncol == 1 else 4.2, h)

    fig, panels, owns = panel_axes(["legend"], None if ax is None else {"legend": ax}, figsize)
    ax = panels["legend"]
    ax.axis("off")

    handles = []
    labels = []
    for d in legend_data:
        col = d.get("color", "k")
        grp = str(d.get("group", ""))
        frac = float(d.get("fraction", 0.0))
        cnt = int(d.get("count", 0))

        h = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=col,
            markeredgecolor="none",
            markersize=markersize,
        )
        handles.append(h)
        labels.append(f"{grp} ({frac*100:.1f}%, n={cnt})")

    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        ncol=int(ncol),
        fontsize=int(fontsize),
        handletextpad=0.4,
        labelspacing=0.35,
        columnspacing=0.9,
        title=title,
        borderaxespad=0.0,
    )

    if owns:
        fig.tight_layout(pad=0.2)
    if show:
        plt.show()
    if return_result:
        return PlotResult(fig, panels, source,
            {"plot": "legend_from_data", "max_items": max_items, "n_total": len(source),
             "n_displayed": len(legend_data), "ncol": ncol, "title": title,
             "owns_figure": owns, "export_scope": "created_figure" if owns else "parent_figure"},
            warnings=[f"Legend displays the first {max_items} of {len(source)} entries."] if len(source) > max_items else [],
            artists={"legend": legend})
    if return_data:
        return fig, ax, source
    return fig, ax
