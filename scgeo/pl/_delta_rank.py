from __future__ import annotations

import matplotlib.pyplot as plt


def delta_rank(adata, store_key: str = "scgeo", kind: str = "shift", level: str = "by"):
    df = __import__("scgeo").get.table(adata, store_key=store_key, kind=kind, level=level)
    df = df.sort_values("delta_norm", ascending=False)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(range(len(df)), df["delta_norm"].values, marker="o", linestyle="-")
    ax.set_xlabel(level)
    ax.set_ylabel("||Δ||")
    ax.set_title("ScGeo: delta magnitude ranking")
    return ax
