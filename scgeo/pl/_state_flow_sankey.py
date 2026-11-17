"""Interactive Sankey adapter over the shared categorical association counts."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Optional

from matplotlib.colors import to_rgba

from ._state_flow import _flow_palette, _number, _prepare_state_flow

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    go = None
    _PLOTLY_IMPORT_ERROR = e
else:
    _PLOTLY_IMPORT_ERROR = None


def _require_plotly():
    if go is None:
        raise ImportError(
            "plotly is required for state_flow_sankey. Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR


def _rgba(color, alpha=None):
    red, green, blue, opacity = to_rgba(color)
    opacity = opacity if alpha is None else alpha
    return f"rgba({red * 255:.3f},{green * 255:.3f},{blue * 255:.3f},{opacity:.3f})"


def state_flow_sankey(
    adata,
    *,
    columns: Sequence[str],
    min_count: int = 1,
    drop_na: bool = False,
    title: Optional[str] = None,
    pad: int = 18,
    thickness: int = 18,
    width: int = 1000,
    height: int = 550,
    arrangement: str = "snap",
    node_color: str = "rgba(120,120,120,0.85)",
    link_color: str = "rgba(120,120,120,0.28)",
    return_data: bool = False,
    show: bool = True,
    category_order=None,
    order="input",
    sort_categories=False,
    palette=None,
    color_by="target",
):
    """Plot interactive categorical associations between annotations on cells.

    Shared counts match ``state_flow_alluvial`` and ``state_flow_chord``.
    ``columns`` names ordered annotations on the same rows; links do not imply
    cell lineage, longitudinal transitions, or elapsed time. Missing values are
    detected before string conversion, so ``drop_na`` leaves literal 'NA' intact.
    ``category_order`` maps stages to complete observed orders (None denotes
    genuine missingness); input order respects categorical levels. Optional
    count or barycentric ordering and complete flat/nested palettes are shared
    with the Matplotlib views.

    ``min_count`` hides links without changing their raw counts or denominator.
    Native Plotly node geometry follows visible links after filtering, so its
    displayed node areas need not represent complete category marginals. Hover
    reports full counts alongside visible and omitted incoming/outgoing counts.
    For reserved gaps and full-marginal geometry, use alluvial or chord.

    The default return remains a Plotly Figure. ``return_data=True`` preserves
    (fig,nodes_df,displayed_links_df); link ``value`` is the raw cell count.
    Complete node/link counts and filtering provenance are retained in
    ``fig.layout.meta['scgeo']``. PlotResult/export_figure are Matplotlib APIs;
    Plotly's native HTML/JSON exports remain available for this interactive view.
    """
    _require_plotly()
    if color_by not in {"source", "target"}:
        raise ValueError("color_by must be 'source' or 'target'.")
    pad = _number(pad, "pad")
    thickness = _number(thickness, "thickness", strict=True)
    width = _number(width, "width", minimum=10)
    height = _number(height, "height", minimum=10)
    prepared = _prepare_state_flow(
        adata,
        columns=columns,
        min_count=min_count,
        drop_na=drop_na,
        category_order=category_order,
        order=order,
        sort_categories=sort_categories,
    )
    nodes = (
        prepared["nodes"]
        .rename(columns={"stage": "column", "stage_index": "stage"})
        .copy(deep=True)
    )
    # Plotly's source/target integers index the node list; reordered category
    # rows must not silently change these identities.
    nodes = nodes.sort_values("node_id", kind="stable").reset_index(drop=True)
    links = prepared["links"].loc[prepared["links"].displayed].copy()
    links["source"], links["target"] = links.source_id, links.target_id
    links["source_col"], links["target_col"] = links.source_stage, links.target_stage
    links["value"] = links["count"]
    if palette is None:
        node_colors = [node_color] * len(nodes)
        link_colors = [link_color] * len(links)
    else:
        colors = _flow_palette(prepared["nodes"], palette)
        node_colors = [_rgba(colors[node]) for node in nodes.node_id]
        link_colors = [
            _rgba(colors[node], 0.38)
            for node in links["source_id" if color_by == "source" else "target_id"]
        ]
    node_custom = [
        [
            escape(str(row.column)),
            row.count,
            row.fraction,
            row.visible_in_count,
            row.visible_out_count,
            row.omitted_in_count,
            row.omitted_out_count,
        ]
        for row in nodes.itertuples(index=False)
    ]
    link_custom = [
        [
            escape(str(row.source_col)),
            escape(str(row.target_col)),
            escape(str(row.source_display_label)),
            escape(str(row.target_display_label)),
            row.fraction,
        ]
        for row in links.itertuples(index=False)
    ]
    stage_total = len(prepared["metadata"]["columns"])
    y_positions = {}
    for _, stage_nodes in prepared["nodes"].groupby("stage", sort=False):
        offset = 0.0
        for row in stage_nodes.itertuples(index=False):
            y_positions[row.node_id] = offset + row.fraction / 2
            offset += row.fraction
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                node={
                    "pad": pad,
                    "thickness": thickness,
                    "line": {"color": "rgba(60,60,60,0.55)", "width": 0.6},
                    "label": [escape(str(label)) for label in nodes.display_label],
                    "color": node_colors,
                    "x": [
                        0.02 + 0.96 * stage / (stage_total - 1) for stage in nodes.stage
                    ],
                    "y": [y_positions[node] for node in nodes.node_id],
                    "customdata": node_custom,
                    "hovertemplate": "%{customdata[0]}: %{label}<br>Complete count: %{customdata[1]} cells<br>Full-denominator fraction: %{customdata[2]:.1%}<br>Visible incoming / outgoing: %{customdata[3]} / %{customdata[4]}<br>Omitted incoming / outgoing: %{customdata[5]} / %{customdata[6]}<extra></extra>",
                },
                link={
                    "source": links.source.tolist(),
                    "target": links.target.tolist(),
                    "value": links.value.tolist(),
                    "color": link_colors,
                    "customdata": link_custom,
                    "hovertemplate": "%{customdata[0]}: %{customdata[2]}<br>%{customdata[1]}: %{customdata[3]}<br>Cells: %{value}<br>Fraction of all included cells: %{customdata[4]:.1%}<extra></extra>",
                },
            )
        ]
    )
    metadata = {
        **prepared["metadata"],
        "plot": "state_flow_sankey",
        "node_geometry": "native Plotly geometry follows visible links; complete marginals are reported in hover",
        "complete_nodes": prepared["nodes"].to_dict(orient="records"),
        "complete_links": prepared["links"].to_dict(orient="records"),
    }
    fig.update_layout(
        title=title or "Same-cell categorical associations",
        font={"size": 12},
        width=width,
        height=height,
        margin={"l": 25, "r": 25, "t": 85, "b": 65},
        meta={"scgeo": metadata},
    )
    for index, column in enumerate(prepared["metadata"]["columns"]):
        fig.add_annotation(
            x=index / (stage_total - 1),
            y=1.08,
            xref="paper",
            yref="paper",
            text=f"<b>{escape(column)}</b>",
            showarrow=False,
        )
    hidden = int((~prepared["links"].displayed).sum())
    note = "Same-cell category associations; no lineage or time inferred."
    if hidden:
        note += f" {hidden} links below {min_count} cells hidden; node geometry follows visible links."
    if links.empty:
        note = "No links reach min_count. Complete counts are retained in the source metadata."
    fig.add_annotation(
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        text=note,
        showarrow=False,
        font={"size": 11, "color": "#617784"},
    )
    nodes.attrs["scgeo"] = prepared["metadata"]
    links.attrs["scgeo"] = prepared["metadata"]
    if show:
        fig.show()
    if return_data:
        return fig, nodes, links
    return fig
