"""Display existing mapping outputs without fitting a mapping or OOD threshold."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like

from ._highlight import _get_scanpy_palette
from ._result import PlotResult
from ._score import score_embedding
from ._style import category_palette
from ._table_plot_utils import numeric_column, panel_axes
from ._utils import observation_labels, resolve_embedding


def mapping_confidence_umap(
    adata,
    conf_key: str = "map_confidence",
    *,
    basis: str = "umap",
    highlight_low_k: Optional[int] = 200,
    title: Optional[str] = None,
    show: bool = True,
    ax=None,
    colorbar: bool = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """Show stored confidence and the lowest finite scores on any embedding.

    Lowest-score selection uses all finite input scores, with input order for
    ties. Selected cells lacking finite coordinates remain in the source table.
    The legacy return remains ``(fig, array([ax]))``; ``return_data`` appends
    the cell table and ``return_result`` enables figure/source exports.
    """
    vals = numeric_column(adata.obs, conf_key)
    if highlight_low_k is not None and (
        isinstance(highlight_low_k, bool)
        or not isinstance(highlight_low_k, (int, np.integer)) or highlight_low_k < 0
    ):
        raise ValueError("highlight_low_k must be a nonnegative integer or None.")
    selected = np.zeros(adata.n_obs, dtype=bool)
    finite = np.flatnonzero(np.isfinite(vals))
    if highlight_low_k:
        selected[finite[np.argsort(vals[finite], kind="stable")[:highlight_low_k]]] = True
    result = score_embedding(
        adata, conf_key, basis=basis, ax=ax, figsize=(5., 4.2),
        title=title or "Mapping confidence", colorbar=colorbar, cax=cax,
        norm=norm, vmin=vmin, vmax=vmax, show=False, return_result=True,
    )
    frame = result.data
    frame["selected_low"] = selected
    frame["highlighted"] = selected & frame.displayed.to_numpy()
    highlight = frame.loc[frame.highlighted]
    axis = result.axes["embedding"]
    result.artists["low_confidence"] = axis.scatter(
        highlight.x, highlight.y, s=20, facecolors="none", edgecolors="black",
        linewidths=.7, alpha=.9, zorder=3,
    )
    result.metadata.update({"plot": "mapping_confidence_umap", "highlight_low_k": highlight_low_k,
                            "selection_scope": "all_finite_input_scores", "tie_break": "input_order",
                            "owns_figure": ax is None,
                            "export_scope": "created_figure" if ax is None else "parent_figure"})
    if show:
        plt.show()
    if return_result:
        return result
    legacy = (axis.figure, np.array([axis], dtype=object))
    return (*legacy, frame) if return_data else legacy


def ood_cells(
    adata,
    ood_key: str = "map_ood_score",
    *,
    basis: str = "umap",
    threshold: Optional[float] = None,
    show_only_flagged: bool = False,
    title: Optional[str] = None,
    show: bool = True,
    ax=None,
    colorbar: bool = True,
    cax=None,
    cmap: str = "viridis",
    norm=None,
    vmin=None,
    vmax=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """Display stored OOD scores or cells above an explicitly supplied threshold.

    No threshold is inferred: ``threshold=None`` draws all cells, as before,
    and exports nullable flags. With ``show_only_flagged=True`` and a threshold,
    the legacy monochrome flagged-cell view is retained (no numeric colorbar).
    Default returns ``(fig, array([ax]))``; data/results are opt-in.
    """
    values = numeric_column(adata.obs, ood_key)
    key, xy = resolve_embedding(adata, basis)
    if threshold is not None and (isinstance(threshold, bool) or not np.isfinite(threshold)):
        raise ValueError("threshold must be finite or None.")
    finite_xy = np.isfinite(xy[:, :2]).all(axis=1)
    flagged = pd.array([pd.NA] * adata.n_obs, dtype="boolean")
    if threshold is not None:
        finite = np.isfinite(values)
        flagged[finite] = values[finite] >= threshold
    only = show_only_flagged and threshold is not None
    if only:
        if cax is not None:
            raise ValueError("The monochrome flagged-only view has no colorbar; omit cax.")
        if norm is not None or vmin is not None or vmax is not None:
            raise ValueError("Numeric color limits do not apply to the monochrome flagged-only view.")
        fig, axes, _ = panel_axes(["embedding"], None if ax is None else {"embedding": ax}, (5., 4.2))
        axis = axes["embedding"]
        displayed = finite_xy & flagged.fillna(False).to_numpy(dtype=bool)
        points = axis.scatter(xy[displayed, 0], xy[displayed, 1], s=8, alpha=.9)
        basis_name = key.removeprefix("X_").upper()
        axis.set(xlabel=f"{basis_name}1", ylabel=f"{basis_name}2", title=title or "Out-of-distribution (OOD)")
        frame = pd.DataFrame({"cell": adata.obs_names.astype(str), "obs_position": np.arange(adata.n_obs),
                              "x": xy[:, 0], "y": xy[:, 1], "value": values, "displayed": displayed})
        result = PlotResult(fig, axes, frame, artists={"flagged": points}, metadata={"basis": key})
    else:
        result = score_embedding(
            adata, ood_key, basis=basis, ax=ax, figsize=(5., 4.2), size=8, alpha=.9,
            title=title or "Out-of-distribution (OOD)", colorbar=colorbar, cax=cax,
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, show=False, return_result=True,
        )
        axis = result.axes["embedding"]
    result.data["flagged"] = flagged
    result.metadata.update({"plot": "ood_cells", "ood_key": ood_key, "threshold": threshold,
                            "threshold_source": "not_supplied" if threshold is None else "user_supplied",
                            "show_only_flagged": show_only_flagged, "display_mode": "flagged_only" if only else "scores",
                            "owns_figure": ax is None,
                            "export_scope": "created_figure" if ax is None else "parent_figure"})
    if show:
        plt.show()
    if return_result:
        return result
    legacy = (axis.figure, np.array([axis], dtype=object))
    return (*legacy, result.data) if return_data else legacy


def mapping_qc_panel(
    adata,
    *,
    pred_key: str = "map_pred",
    conf_key: str = "map_confidence",
    ood_key: str = "map_ood_score",
    basis: str = "umap",
    show: bool = True,
    palette_from: str | None = None,
    condition_key: str | None = None,
    query_value: str | None = None,
    show_ref_as_grey: bool = True,
    return_legend_data: bool = False,
    axes=None,
    colorbar: bool = True,
    return_data: bool = False,
    return_result: bool = False,
):
    """Compose predictions, confidence and OOD scores from stored obs columns.

    ``axes`` names are ``prediction``, ``confidence``, ``ood``; when colorbars
    are enabled on supplied axes, also provide ``confidence_colorbar`` and
    ``ood_colorbar``. Reference colors are reused for matching predictions;
    novel labels receive deterministic colors and are never omitted. True
    missing values remain separate from literal labels such as ``'nan'``.
    Query filtering applies only to predictions; both scalar panels use all
    input observations. No mapping model or threshold is computed here.

    Legacy ``(fig, axes)`` and ``return_legend_data`` returns are preserved.
    ``return_data`` appends all source tables; ``return_result`` takes priority.
    """
    for key in (pred_key, conf_key, ood_key):
        if key not in adata.obs:
            raise KeyError(f"{key} not found in adata.obs")
    numeric_column(adata.obs, conf_key)
    numeric_column(adata.obs, ood_key)
    emb_key, xy = resolve_embedding(adata, basis)
    pred, categories, missing = observation_labels(adata.obs[pred_key], pred_key)
    mask = np.ones(adata.n_obs, dtype=bool)
    if condition_key is not None and query_value is not None:
        if condition_key not in adata.obs:
            raise KeyError(f"{condition_key} not found in adata.obs")
        conditions, _, _ = observation_labels(adata.obs[condition_key], condition_key)
        mask = conditions.eq(str(query_value)).to_numpy()
    finite_xy = np.isfinite(xy[:, :2]).all(axis=1)
    if not finite_xy.any():
        raise ValueError("No finite embedding coordinates are available to plot.")

    palette = None
    palette_origin = "deterministic_hash"
    for source in dict.fromkeys([palette_from, pred_key]):
        if source is None or source not in adata.obs:
            continue
        stored = _get_scanpy_palette(adata, source)
        if stored is not None and all(is_color_like(c) for c in stored.values()):
            palette = stored
            palette_origin = f"scanpy_uns:{source}"
            break
    colors = category_palette(categories)
    if palette is not None:
        colors.update({label: palette[label] for label in categories if label in palette})
        categories = [label for label in palette if label in categories] + [label for label in categories if label not in palette]
    if missing is not None:
        colors[missing] = "lightgrey"
    names = ["prediction", "confidence", "ood"]
    if axes is not None and colorbar:
        panel_axes(names + ["confidence_colorbar", "ood_colorbar"], axes, None)
    fig, panels, owns = panel_axes(names, axes, (14.2, 4.2))
    if owns:
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("constrained")
        else:  # Matplotlib 3.5
            fig.set_constrained_layout(True)
    artists = {}
    background = ~mask & finite_xy if show_ref_as_grey else np.zeros(adata.n_obs, dtype=bool)
    artists["reference"] = panels["prediction"].scatter(
        xy[background, 0], xy[background, 1], s=6, c="lightgrey", alpha=.35, linewidths=0,
    )
    legend_data = []
    for label in categories:
        selected = pred.eq(label).to_numpy() & mask
        draw = selected & finite_xy
        if selected.any():
            artists[f"prediction:{label}"] = panels["prediction"].scatter(
                xy[draw, 0], xy[draw, 1], s=8, c=[colors[label]], alpha=.9, linewidths=0,
            )
            legend_data.append({"group": label, "count": int(selected.sum()),
                                "fraction": float(selected.sum() / max(mask.sum(), 1)),
                                "color": colors[label]})
    label = emb_key.removeprefix("X_").upper()
    panels["prediction"].set(xlabel=f"{label}1", ylabel=f"{label}2", title="Predicted label")
    children = {}
    for name, score, title in (("confidence", conf_key, "Confidence"), ("ood", ood_key, "OOD score")):
        children[name] = score_embedding(
            adata, score, basis=basis, ax=panels[name], title=title, show=False,
            colorbar=colorbar, cax=None if axes is None or not colorbar else axes[f"{name}_colorbar"],
            return_result=True,
        )
        artists[name] = children[name].artists
    cells = children["confidence"].data[["cell", "obs_position", "x", "y"]].copy()
    cells["prediction"] = adata.obs[pred_key].to_numpy()
    cells["prediction_label"] = pred.to_numpy()
    cells["query"] = mask
    cells["prediction_displayed"] = mask & finite_xy
    cells["reference_displayed"] = background
    cells["confidence"] = numeric_column(adata.obs, conf_key)
    cells["ood_score"] = numeric_column(adata.obs, ood_key)
    cells["scalar_displayed"] = finite_xy
    legend_data.sort(key=lambda d: d["fraction"], reverse=True)
    data = {"cells": cells, "legend": pd.DataFrame(legend_data, columns=["group", "count", "fraction", "color"])}
    metadata = {"plot": "mapping_qc_panel", "basis": emb_key, "pred_key": pred_key,
                "conf_key": conf_key, "ood_key": ood_key, "condition_key": condition_key,
                "query_value": query_value, "palette_from": palette_from, "palette_origin": palette_origin,
                "palette": colors, "missing_label": missing, "prediction_scope": "query" if not mask.all() else "all",
                "scalar_scope": "all_input_observations", "legend_counts_include_nondrawable_cells": True,
                "confidence": children["confidence"].metadata, "ood": children["ood"].metadata,
                "owns_figure": owns, "export_scope": "created_figure" if owns else "parent_figure"}
    if show:
        plt.show()
    if return_result:
        return PlotResult(fig, panels, data, metadata, artists=artists)
    legacy = (fig, np.array(list(panels.values()), dtype=object))
    if return_legend_data:
        legacy = (*legacy, legend_data)
    return (*legacy, data) if return_data else legacy
