from __future__ import annotations

from typing import Optional

from ._score import _score_distribution_panel


def ambiguity_panel(
    adata,
    score_key: str,
    *,
    basis: str = "umap",
    topk: Optional[int] = 200,
    cmap: str = "inferno",
    figsize=(11, 5),
    title: Optional[str] = None,
    show: bool = True,
    axes=None,
    bins=40,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: bool = True,
    max_points=None,
    random_state=0,
    rasterized="auto",
    return_data: bool = False,
    return_result: bool = False,
):
    """Plot stored ambiguity scores and their full-data histogram.

    Highlights share the embedding color normalization. ``axes`` accepts named
    ``embedding`` and ``histogram`` axes, plus ``colorbar`` unless
    ``colorbar=False``. Supplied figures retain their size and layout.
    ``max_points`` affects background drawing only and must accommodate every
    drawable top-k cell. The histogram always uses all finite input scores.
    No analysis or biological interpretation is recomputed.

    The legacy return is ``(fig, axes_array)``. ``return_data=True`` adds cell
    and histogram tables. ``return_result=True`` supplies a PlotResult with
    exact bin edges/counts, cell identities, selections, and displayed flags.
    """
    return _score_distribution_panel(
        adata, score_key, basis=basis, topk=topk, cmap=cmap, figsize=figsize,
        title=title or f"Ambiguity score: {score_key}", axes=axes, bins=bins,
        norm=norm, vmin=vmin, vmax=vmax, colorbar=colorbar, max_points=max_points,
        random_state=random_state, rasterized=rasterized, plot_name="ambiguity_panel",
        return_data=return_data, return_result=return_result, show=show)
