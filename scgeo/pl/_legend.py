from __future__ import annotations

from typing import Iterable, Dict, Any, Tuple
import matplotlib.pyplot as plt


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
):
    """
    Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

    legend_data items:
      {"group": str, "fraction": float, "count": int, "color": <matplotlib color>}
    """
    legend_data = list(legend_data)[: int(max_items)]

    if figsize is None:
        # heuristic: tiny, scales gently with number of entries
        h = max(1.2, 0.25 * len(legend_data))
        figsize = (2.8 if ncol == 1 else 4.2, h)

    fig, ax = plt.subplots(figsize=figsize)
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

    ax.legend(
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

    fig.tight_layout(pad=0.2)
    if show:
        plt.show()
    return fig, ax
