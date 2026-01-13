from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

def paga_composition_stats(
    adata,
    group_key: str,                 # e.g. "leiden" (PAGA nodes)
    condition_key: str,
    group0: Any,
    group1: Any,
    *,
    sample_key: Optional[str] = None,
    method: str = "gee",            # gee | bootstrap
    n_boot: int = 1000,
    seed: int = 0,
    store_key: str = "scgeo",
) -> None:
    """
    For each cluster/node, test enrichment of condition1 vs condition0.

    If method="gee" and sample_key provided:
      - Fit GEE logistic regression: y = 1(cell in node), x = condition (0/1), cluster = sample
      - Returns OR and robust SE CI

    Fallback method="bootstrap":
      - Bootstrap over samples: compute per-sample proportions in node, compare groups, CI + effect
    """
    if group_key not in adata.obs:
        raise KeyError(f"obs key '{group_key}' not found")
    if condition_key not in adata.obs:
        raise KeyError(f"obs key '{condition_key}' not found")
    if sample_key is not None and sample_key not in adata.obs:
        raise KeyError(f"obs key '{sample_key}' not found")

    rs = np.random.RandomState(seed)

    df = pd.DataFrame({
        "node": adata.obs[group_key].astype(str).values,
        "cond": adata.obs[condition_key].astype(str).values,
    }, index=adata.obs_names)

    df["x"] = (df["cond"] == str(group1)).astype(int)
    keep = (df["cond"] == str(group0)) | (df["cond"] == str(group1))
    df = df.loc[keep].copy()

    if sample_key is not None:
        df["sample"] = adata.obs.loc[df.index, sample_key].astype(str).values
    else:
        df["sample"] = "all"

    nodes = np.unique(df["node"].values)
    rows = []

    if method == "gee":
        if sample_key is None:
            # still run but clusters are trivial
            pass
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.families import Binomial
            from statsmodels.genmod.cov_struct import Exchangeable
        except Exception as e:
            raise RuntimeError("statsmodels required for method='gee'") from e

        for node in nodes:
            tmp = df.copy()
            tmp["y"] = (tmp["node"] == node).astype(int)

            # design matrix: intercept + condition indicator
            X = sm.add_constant(tmp["x"].values, has_constant="add")
            y = tmp["y"].values
            groups = tmp["sample"].values

            # GEE with exchangeable correlation within sample
            model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
            res = model.fit()

            beta = float(res.params[1])
            se = float(res.bse[1])
            or_ = float(np.exp(beta))
            ci_lo = float(np.exp(beta - 1.96 * se))
            ci_hi = float(np.exp(beta + 1.96 * se))
            p = float(res.pvalues[1])

            rows.append(dict(node=node, method="gee", n=int(tmp.shape[0]), OR=or_, CI_low=ci_lo, CI_high=ci_hi, p=p))

    elif method == "bootstrap":
        # bootstrap over samples: compare mean node proportion between conditions
        samples = np.unique(df["sample"].values)
        # per-sample node proportions
        tab = df.groupby(["sample", "cond", "node"]).size().rename("n").reset_index()
        tot = df.groupby(["sample", "cond"]).size().rename("N").reset_index()
        tab = tab.merge(tot, on=["sample", "cond"], how="left")
        tab["prop"] = tab["n"] / tab["N"]

        # build dict sample->cond
        s2c = df.groupby("sample")["cond"].agg(lambda x: x.mode().iloc[0]).to_dict()
        s0 = [s for s in samples if s2c[s] == str(group0)]
        s1 = [s for s in samples if s2c[s] == str(group1)]

        for node in nodes:
            # observed effect: difference in mean prop (group1 - group0)
            p0 = tab[(tab["node"] == node) & (tab["cond"] == str(group0))].set_index("sample")["prop"]
            p1 = tab[(tab["node"] == node) & (tab["cond"] == str(group1))].set_index("sample")["prop"]
            obs = float(p1.reindex(s1, fill_value=0).mean() - p0.reindex(s0, fill_value=0).mean())

            boots = []
            for _ in range(n_boot):
                bs0 = rs.choice(s0, size=len(s0), replace=True)
                bs1 = rs.choice(s1, size=len(s1), replace=True)
                b = float(p1.reindex(bs1, fill_value=0).mean() - p0.reindex(bs0, fill_value=0).mean())
                boots.append(b)
            boots = np.array(boots)
            ci_lo = float(np.quantile(boots, 0.025))
            ci_hi = float(np.quantile(boots, 0.975))
            # two-sided p from bootstrap sign
            p = float(2 * min((boots >= 0).mean(), (boots <= 0).mean()))

            rows.append(dict(node=node, method="bootstrap", effect=obs, CI_low=ci_lo, CI_high=ci_hi, p=p, n_boot=int(n_boot)))
    else:
        raise ValueError("method must be 'gee' or 'bootstrap'")

    resdf = pd.DataFrame(rows)

    # BH-FDR
    if "p" in resdf.columns and resdf["p"].notna().any():
        pvals = resdf["p"].values.astype(float)
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = np.sum(~np.isnan(ranked))
        q = np.full_like(pvals, np.nan, dtype=float)
        if m > 0:
            ranks = np.arange(1, len(ranked) + 1)
            qtmp = ranked * m / ranks
            qtmp = np.minimum.accumulate(qtmp[::-1])[::-1]
            q[order] = qtmp
        resdf["q"] = q

    adata.uns.setdefault(store_key, {})
    adata.uns[store_key]["paga_composition_stats"] = {
        "params": dict(group_key=group_key, condition_key=condition_key, group0=group0, group1=group1, sample_key=sample_key, method=method, n_boot=n_boot, seed=seed),
        "table": resdf,
    }
