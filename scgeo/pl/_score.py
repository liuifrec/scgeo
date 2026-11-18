from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _get_ax, display_indices, resolve_embedding, observation_labels


def score_embedding(
    adata,
    score_key: str,
    basis: str = "umap",
    *,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    size: float = 6.0,
    alpha: float = 0.8,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    norm=None,
    palette: Optional[Mapping[str, str]] = None,
    na_color: str = "lightgrey",
    figsize=None,
    max_points: Optional[int] = None,
    stratify_by: Optional[str] = None,
    random_state: int = 0,
    required_indices=None,
    rasterized="auto",
    colorbar: bool = True,
    cax=None,
    legend: bool = True,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """Plot an observation score or category on the first two embedding axes.

    Both ``umap`` and ``X_umap`` resolve to the same coordinates. Continuous
    color limits use all finite input scores, even when ``max_points`` selects
    a reproducible display subset. ``stratify_by`` preserves at least one point
    per observed state; a budget too small for those states raises an error.
    ``required_indices`` preserves selected original cell positions when sampling;
    these cells must have finite display coordinates. Categorical scores use a
    stable label-to-color mapping and no colorbar.

    The legacy return is ``(fig, ax)``. ``return_data=True`` adds the full source
    table with a ``displayed`` indicator; ``return_result=True`` supplies a
    :class:`PlotResult` with the original displayed indices and random seed.
    The retained ``layer`` argument has no effect: scores come from ``obs``.
    ``colorbar=False`` suppresses the numeric scale; ``cax`` places it on an
    explicit axis without changing the embedding panel's allocated space.
    ``legend=False`` suppresses categorical legends. Valid Scanpy category
    colors are reused unless ``palette`` is explicitly supplied. Category
    artists are always accessible through ``artists['categories'][label]``;
    this avoids collisions with auxiliary artist names such as ``'legend'``.
    True missing values receive a distinct label and never merge with a literal
    category. Invalid category/norm combinations fail before creating a figure.
    """
    from ._result import PlotResult
    from ._style import category_palette

    emb_key, coords = resolve_embedding(adata, basis)
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if rasterized not in (True, False, "auto"):
        raise ValueError("rasterized must be True, False, or 'auto'.")
    if not isinstance(colorbar, bool) or not isinstance(legend, bool):
        raise TypeError("colorbar and legend must be booleans.")
    if cax is not None:
        if ax is None or not colorbar:
            raise ValueError("cax requires ax and colorbar=True.")
        from ._table_plot_utils import panel_axes
        panel_axes(["embedding", "colorbar"], {"embedding": ax, "colorbar": cax}, None)
    scores = adata.obs[score_key]
    categorical = not pd.api.types.is_numeric_dtype(
        scores.dtype
    ) or pd.api.types.is_bool_dtype(scores.dtype)
    if categorical:
        if norm is not None or vmin is not None or vmax is not None:
            raise ValueError(
                "norm and numeric color limits cannot be used with categorical scores."
            )
        mapped_labels, observed_labels, missing_label = observation_labels(scores, score_key)
        if cax is not None:
            raise ValueError("Categorical scores have no numeric colorbar; omit cax.")
        categories = (
            list(scores.cat.categories.astype(str))
            if isinstance(scores.dtype, pd.CategoricalDtype)
            else [label for label in observed_labels if label != missing_label]
        )
        if len(set(categories)) != len(categories):
            raise ValueError("Distinct categories collide after string conversion.")
        palette_origin = "provided" if palette is not None else "deterministic_hash"
        if palette is None:
            from ._highlight import _get_scanpy_palette
            from matplotlib.colors import is_color_like
            stored = _get_scanpy_palette(adata, score_key)
            if stored is not None and all(is_color_like(c) for c in stored.values()):
                palette = stored
                palette_origin = "scanpy_uns"
        colors = category_palette(categories, palette=palette)
    if stratify_by is not None and stratify_by not in adata.obs:
        raise KeyError(f"{stratify_by} not found in adata.obs")
    labels = (
        adata.obs[stratify_by]
        if stratify_by is not None
        else (scores if categorical else pd.Series("all", index=adata.obs.index))
    )
    eligible = np.flatnonzero(np.isfinite(coords[:, :2]).all(axis=1))
    if not len(eligible):
        raise ValueError("No finite embedding coordinates are available to plot.")
    required_relative = None
    if required_indices is not None:
        required = np.asarray(required_indices)
        if required.ndim != 1 or (required.size and required.dtype.kind not in "iu"):
            raise ValueError(
                "required_indices must be a one-dimensional sequence of integer cell positions."
            )
        required = required.astype(int)
        if np.any(required < 0) or np.any(required >= adata.n_obs):
            raise ValueError("required_indices contains an out-of-range cell position.")
        if not np.isin(required, eligible).all():
            raise ValueError("Required cells must have finite display coordinates.")
        required_relative = np.searchsorted(eligible, required)
    relative = display_indices(
        labels.iloc[eligible].to_numpy(),
        max_points=max_points,
        random_state=random_state,
        required_indices=required_relative,
    )
    displayed = eligible[relative]
    rasterize = len(displayed) > 10000 if rasterized == "auto" else bool(rasterized)
    owns_figure = ax is None
    fig, ax = _get_ax(ax=ax, figsize=figsize)
    source = pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "obs_position": np.arange(adata.n_obs),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "value": scores.to_numpy(),
            "displayed": False,
        }
    )
    source.loc[displayed, "displayed"] = True
    if stratify_by is not None:
        source["stratum"] = labels.to_numpy()
    artists = {}
    metadata = {
        "plot": "score_embedding",
        "score_key": score_key,
        "basis": emb_key,
        "categorical": categorical,
        "random_state": int(random_state),
        "max_points": max_points,
        "stratify_by": stratify_by,
        "rasterized": rasterize,
        "required_indices": [] if required_indices is None else required.tolist(),
        "displayed_indices": displayed.tolist(),
        "n_total": len(source),
        "n_displayed": len(displayed),
        "color_scale_uses_full_data": True,
        "marker_size": size,
        "alpha": alpha,
        "na_color": na_color,
        "colorbar": colorbar and not categorical,
        "legend": legend and categorical,
        "owns_figure": owns_figure,
        "export_scope": "created_figure" if owns_figure else "parent_figure",
    }
    if categorical:
        source["value_label"] = mapped_labels.to_numpy()
        artists["categories"] = {}
        values = scores.iloc[displayed]
        missing = values.isna().to_numpy()
        if missing.any():
            artists["missing"] = ax.scatter(
                coords[displayed[missing], 0],
                coords[displayed[missing], 1],
                s=size,
                alpha=alpha,
                c=na_color,
                linewidths=0,
                rasterized=rasterize,
                label=missing_label,
            )
        for category in categories:
            mask = (~missing) & values.astype(str).eq(category).to_numpy()
            if mask.any():
                artist = ax.scatter(
                    coords[displayed[mask], 0],
                    coords[displayed[mask], 1],
                    s=size,
                    alpha=alpha,
                    c=[colors[category]],
                    linewidths=0,
                    rasterized=rasterize,
                    label=category,
                )
                artists["categories"][category] = artist
                if category not in {"categories", "missing", "legend", "scores", "colorbar"}:
                    artists[category] = artist
        if legend:
            artists["legend"] = ax.legend(frameon=False, title=score_key, markerscale=1.5)
        metadata["palette"] = colors
        metadata["palette_origin"] = palette_origin
        metadata["missing_label"] = missing_label
    else:
        values = pd.to_numeric(scores, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        finite = np.isfinite(values)
        if norm is None:
            finite_values = values[finite]
            low = float(np.min(finite_values)) if len(finite_values) else 0.0
            high = float(np.max(finite_values)) if len(finite_values) else 1.0
            norm = plt.Normalize(
                vmin=low if vmin is None else vmin, vmax=high if vmax is None else vmax
            )
        else:
            # Initialize an unset caller-supplied norm using the full score array.
            norm.autoscale_None(values[finite])
        bad = displayed[~finite[displayed]]
        good = displayed[finite[displayed]]
        if len(bad):
            artists["missing"] = ax.scatter(
                coords[bad, 0],
                coords[bad, 1],
                s=size,
                alpha=alpha,
                c=na_color,
                linewidths=0,
                rasterized=rasterize,
            )
        artists["scores"] = ax.scatter(
            coords[good, 0],
            coords[good, 1],
            s=size,
            alpha=alpha,
            c=values[good],
            cmap=cmap,
            norm=norm,
            linewidths=0,
            rasterized=rasterize,
        )
        if colorbar:
            artists["colorbar"] = fig.colorbar(artists["scores"], ax=ax, cax=cax)
            artists["colorbar"].set_label(score_key)
        metadata["cmap"] = artists["scores"].get_cmap().name
        metadata["norm"] = {
            "type": type(norm).__name__,
            "vmin": norm.vmin,
            "vmax": norm.vmax,
        }
    basis_name = emb_key.removeprefix("X_")
    ax.set(
        xlabel=f"{basis_name.upper()}1",
        ylabel=f"{basis_name.upper()}2",
        title=title or score_key,
    )
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes={"embedding": ax},
            artists=artists,
            data=source,
            metadata=metadata,
        )
    if return_data:
        return fig, ax, source
    return fig, ax


def score_umap(adata, score_key: str, **kwargs):
    return score_embedding(adata, score_key=score_key, basis="umap", **kwargs)


def _score_distribution_panel(
    adata, score_key, *, basis, topk, cmap, figsize, title, axes, bins,
    norm, vmin, vmax, colorbar, max_points, random_state, rasterized, plot_name,
    return_data, return_result, show, singular_values=None, store_key=None,
):
    """Shared composition of the existing score and highlight renderers."""
    from ._highlight import _topk_mask_robust, highlight_topk_cells
    from ._result import PlotResult
    from ._table_plot_utils import panel_axes

    key, coords = resolve_embedding(adata, basis)
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")
    if topk is not None and (isinstance(topk, bool) or not isinstance(topk, (int, np.integer)) or topk < 0):
        raise ValueError("topk must be a nonnegative integer or None.")
    if not pd.api.types.is_numeric_dtype(adata.obs[score_key]) or pd.api.types.is_bool_dtype(adata.obs[score_key]):
        raise TypeError("Score distribution panels require a numeric observation column.")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    scores = pd.to_numeric(adata.obs[score_key], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(scores)
    selected = _topk_mask_robust(scores, topk or 0)
    required = np.flatnonzero(selected & np.isfinite(coords[:, :2]).all(axis=1))
    # Both count computation and sampling validation precede canvas changes.
    counts, edges = np.histogram(scores[finite], bins=bins)
    if singular_values is not None and np.asarray(singular_values).ndim != 1:
        raise ValueError("Stored singular_values must be one-dimensional.")
    eligible = np.flatnonzero(np.isfinite(coords[:, :2]).all(axis=1))
    if not len(eligible):
        raise ValueError("No finite embedding coordinates are available to plot.")
    display_indices(np.full(len(eligible), "all"), max_points=max_points,
                    random_state=random_state, required_indices=np.searchsorted(eligible, required))
    names = ["embedding", "histogram"]
    if axes is not None:
        if not isinstance(axes, Mapping):
            raise TypeError("axes must be a mapping from panel labels to Matplotlib Axes.")
        if colorbar and "colorbar" not in axes:
            raise ValueError("Supply axes['colorbar'] or set colorbar=False.")
        if "colorbar" in axes:
            names.append("colorbar")
    fig, mapped, owns_figure = panel_axes(names, axes, figsize)
    ax, hist_ax = mapped["embedding"], mapped["histogram"]
    score = score_embedding(
        adata, score_key, basis=basis, ax=ax, cmap=cmap, title=title, norm=norm,
        vmin=vmin, vmax=vmax, colorbar=colorbar, cax=mapped.get("colorbar") if colorbar else None,
        max_points=max_points, random_state=random_state, required_indices=required,
        rasterized=rasterized, return_result=True, show=False)
    artists = dict(score.artists)
    notes = []
    if topk:
        highlight = highlight_topk_cells(
            adata, score_key, basis=basis, topk=topk, ax=ax, cmap=cmap,
            norm=score.artists["scores"].norm, background=False, add_colorbar=False,
            title=title, return_result=True, show=False)
        artists["highlights"] = highlight.artists["highlights"]
        notes.extend(highlight.warnings)
    bars = hist_ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#527F9A")
    artists["histogram"] = bars
    hist_ax.set(title="Score distribution", xlabel=score_key, ylabel="Cell count")
    cells = score.data.copy(deep=True)
    cells["selected"] = selected
    cells["highlighted"] = selected & np.isfinite(coords[:, :2]).all(axis=1)
    cells["histogram_included"] = finite & (scores >= edges[0]) & (scores <= edges[-1])
    data = {"cells": cells,
            "histogram": pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})}
    if singular_values is not None:
        values = np.asarray(singular_values)
        data["singular_values"] = pd.DataFrame({"singular_value": values})
        artists["singular_values"] = hist_ax.text(
            .98, .98, f"singular_values:\n{values[:5]}", ha="right", va="top",
            transform=hist_ax.transAxes, fontsize=9)
    if not finite.all():
        notes.append(f"{int((~finite).sum())} nonfinite scores were excluded from the histogram.")
    if owns_figure:
        fig.tight_layout()
    result = PlotResult(
        fig, {"embedding": ax, "histogram": hist_ax}, data=data, artists=artists, warnings=notes,
        metadata={**score.metadata, "plot": plot_name, "basis": key, "topk": topk,
                  "histogram_source": "all finite input scores, independent of display sampling",
                  "histogram_n": int(counts.sum()), "store_key": store_key,
                  "owns_figure": owns_figure,
                  "export_scope": "created_figure" if owns_figure else "parent_figure",
                  "score_source": f"adata.obs[{score_key!r}]", "analysis_recomputed": False})
    if show:
        plt.show()
    if return_result:
        return result
    legacy_axes = np.array([ax, hist_ax], dtype=object)
    return (fig, legacy_axes, data) if return_data else (fig, legacy_axes)
