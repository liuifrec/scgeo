#!/usr/bin/env python3
"""
Generate a robust IO manifest for scgeo tl functions.

Tracks:
- added keys
- modified existing keys
- container-level writes (obs / obsm / uns / layers)

Output:
  scgeo_io_manifest.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from copy import deepcopy
from typing import Dict, Any

import numpy as np
import anndata as ad
import scgeo as sg


# ---------- helpers ----------

def shallow_snapshot(adata: ad.AnnData) -> Dict[str, Any]:
    """Track keys + shallow fingerprints (not values)."""
    return {
        "obs_cols": list(adata.obs.columns),
        "obsm_keys": list(adata.obsm.keys()),
        "layers_keys": list(adata.layers.keys()),
        "uns_keys": list(adata.uns.keys()),
    }


def diff_snapshot(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, list]]:
    out = {}
    for k in before:
        b, a = set(before[k]), set(after[k])
        out[k] = {
            "added": sorted(a - b),
            "removed": sorted(b - a),
            "touched": sorted(a & b),  # important!
        }
    return out


def make_toy_adata(n: int = 60) -> ad.AnnData:
    rng = np.random.RandomState(0)
    adata = ad.AnnData(X=rng.normal(size=(n, 10)))
    adata.obsm["X_pca"] = rng.normal(size=(n, 8))
    adata.obsm["X_umap"] = rng.normal(size=(n, 2))
    adata.obs["condition"] = rng.choice(["A", "B"], size=n)
    adata.obs["batch"] = rng.choice(["b0", "b1"], size=n)
    adata.obs["cluster"] = rng.choice(["c0", "c1", "c2"], size=n)
    return adata


# ---------- TL invocation policy ----------

def try_call(fn, adata):
    name = fn.__name__
    try:
        if name == "shift":
            fn(adata, rep="X_pca", condition_key="condition",
               group0="A", group1="B", store_key="shift_test")
        elif name == "mixscore":
            fn(adata, label_key="batch", rep="X_umap", store_key="mix_test")
        elif name == "density_overlap":
            fn(adata, rep="X_umap", condition_key="condition",
               group0="A", group1="B", store_key="dens_test")
        elif name in {"distribution_test", "wasserstein"}:
            fn(adata, rep="X_pca", condition_key="condition",
               group0="A", group1="B", store_key="dist_test")
        else:
            return False, "needs domain-specific inputs"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scgeo_io_manifest.json")
    args = ap.parse_args()

    manifest = {
        "scgeo_version": getattr(sg, "__version__", "unknown"),
        "tl": {},
        "skipped": {},
    }

    for name in dir(sg.tl):
        if name.startswith("_"):
            continue
        fn = getattr(sg.tl, name)
        if not callable(fn):
            continue

        adata = make_toy_adata()
        before = shallow_snapshot(adata)

        ok, note = try_call(fn, adata)
        after = shallow_snapshot(adata)

        entry = {
            "signature": str(inspect.signature(fn)),
            "ok": ok,
            "note": note,
            "writes": diff_snapshot(before, after),
        }

        key = f"scgeo.tl.{name}"
        if ok:
            manifest["tl"][key] = entry
        else:
            manifest["skipped"][key] = entry

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"OK: {len(manifest['tl'])}, skipped: {len(manifest['skipped'])}")


if __name__ == "__main__":
    main()
