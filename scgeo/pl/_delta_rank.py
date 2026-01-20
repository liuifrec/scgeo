from __future__ import annotations

import matplotlib.pyplot as plt


def delta_rank(
    adata,
    store_key: str = "scgeo",
    kind: str = "shift",
    level: str = "by",
    *,
    top_n: int | None = None,
    rotate_xticks: int = 60,
):
    """
    Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.

    Notes
    -----
    Uses sg.get.table(...). Expected to contain:
      - delta_norm column
      - some label column identifying the group (often 'label', or sometimes `level`)
    """
    sg = __import__("scgeo")
    df = sg.get.table(adata, store_key=store_key, kind=kind, level=level)

    if "delta_norm" not in df.columns:
        raise KeyError(f"delta_rank expects 'delta_norm' in table; got columns={list(df.columns)}")

    df = df.sort_values("delta_norm", ascending=False)

    if top_n is not None:
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

    fig, ax = plt.subplots()
    ax.plot(range(len(df)), y, marker="o", linestyle="-")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right")

    ax.set_xlabel(label_col if label_col is not None else level)
    ax.set_ylabel("||Δ||")
    ax.set_title("ScGeo: delta magnitude ranking")
    fig.tight_layout()
    return ax
