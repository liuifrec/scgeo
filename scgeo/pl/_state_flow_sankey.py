from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    go = None
    _PLOTLY_IMPORT_ERROR = e
else:
    _PLOTLY_IMPORT_ERROR = None


def _require_plotly():
    if go is None:
        raise ImportError(
            "plotly is required for state_flow_sankey. "
            "Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR


def _coerce_str_categories(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            raise KeyError(f"'{c}' not found in adata.obs.")
        out[c] = out[c].astype(str).fillna("NA")
    return out


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
):
    """
    Plot a categorical state-flow Sankey diagram from columns in `adata.obs`.

    Parameters
    ----------
    adata
        AnnData object.
    columns
        Ordered categorical columns from `adata.obs`, e.g.
        ["timepoint", "alignment_group", "macrostates_fwd"].
    min_count
        Minimum flow count to retain.
    drop_na
        If True, rows containing 'NA' in any selected column are removed.
    return_data
        If True, return (fig, nodes_df, links_df).

    Notes
    -----
    This is a categorical flow summary, useful for showing how cells distribute
    across timepoints, alignment classes, clusters, macrostates, annotations, or OOD groups.
    """
    _require_plotly()

    if len(columns) < 2:
        raise ValueError("columns must contain at least 2 obs columns.")

    obs = _coerce_str_categories(adata.obs, columns)
    obs = obs[list(columns)].copy()

    if drop_na:
        for c in columns:
            obs = obs[obs[c] != "NA"].copy()

    # Build node table with stage-specific labels so repeated category names across columns do not collide.
    node_records = []
    for stage_idx, col in enumerate(columns):
        cats = pd.unique(obs[col])
        for cat in cats:
            node_records.append(
                {
                    "node_key": f"{col}::{cat}",
                    "label": str(cat),
                    "stage": stage_idx,
                    "column": col,
                }
            )
    nodes_df = pd.DataFrame(node_records).drop_duplicates("node_key").reset_index(drop=True)
    nodes_df["node_id"] = range(len(nodes_df))

    node_map = dict(zip(nodes_df["node_key"], nodes_df["node_id"]))

    # Build links between consecutive columns
    link_frames = []
    for left, right in zip(columns[:-1], columns[1:]):
        tmp = (
            obs.groupby([left, right], observed=False)
            .size()
            .rename("value")
            .reset_index()
        )
        tmp = tmp[tmp["value"] >= int(min_count)].copy()
        if tmp.empty:
            continue

        tmp["source_key"] = left + "::" + tmp[left].astype(str)
        tmp["target_key"] = right + "::" + tmp[right].astype(str)
        tmp["source"] = tmp["source_key"].map(node_map)
        tmp["target"] = tmp["target_key"].map(node_map)
        tmp["source_col"] = left
        tmp["target_col"] = right
        link_frames.append(
            tmp[
                [
                    "source",
                    "target",
                    "value",
                    "source_col",
                    "target_col",
                    left,
                    right,
                ]
            ].rename(columns={left: "source_label", right: "target_label"})
        )

    if not link_frames:
        raise ValueError("No links to plot. Check columns, filters, or min_count.")

    links_df = pd.concat(link_frames, ignore_index=True)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                node=dict(
                    pad=pad,
                    thickness=thickness,
                    line=dict(color="rgba(60,60,60,0.55)", width=0.6),
                    label=nodes_df["label"].tolist(),
                    color=[node_color] * len(nodes_df),
                ),
                link=dict(
                    source=links_df["source"].tolist(),
                    target=links_df["target"].tolist(),
                    value=links_df["value"].tolist(),
                    color=[link_color] * len(links_df),
                    customdata=links_df[["source_col", "target_col", "source_label", "target_label"]].values,
                    hovertemplate=(
                        "%{customdata[0]}: %{customdata[2]}"
                        "<br>%{customdata[1]}: %{customdata[3]}"
                        "<br>cells: %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )

    if title is None:
        title = "ScGeo state flow Sankey"

    fig.update_layout(
        title=title,
        font=dict(size=12),
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    if show:
        fig.show()

    if return_data:
        return fig, nodes_df, links_df
    return fig