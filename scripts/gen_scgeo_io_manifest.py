#!/usr/bin/env python3
"""
Generate a minimal IO/contract manifest by diffing AnnData before/after each tl function.

This does NOT prove scientific correctness.
It answers: "Which keys did this tl function write? (obs/obsm/uns/layers)"

Output:
  scgeo_io_manifest.json

Usage:
  python scripts/gen_scgeo_io_manifest.py --out scgeo_io_manifest.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import anndata as ad

import scgeo as sg


def snapshot(adata: ad.AnnData) -> Dict[str, Any]:
    def keys_uns(u):
        # keep only key names, not the heavy content
        return sorted(list(u.keys()))

    return {
        "obs_cols": sorted(list(adata.obs.columns)),
        "obsm_keys": sorted(list(adata.obsm.keys())),
        "layers_keys": sorted(list(adata.layers.keys())),
        "uns_keys": keys_uns(adata.uns),
    }


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k in a.keys():
        aa = set(a[k])
        bb = set(b[k])
        out[k + "_added"] = sorted(list(bb - aa))
        out[k + "_removed"] = sorted(list(aa - bb))
    return out


def make_toy_adata(n: int = 80, d: int = 12) -> ad.AnnData:
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    adata.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    adata.obs["condition"] = rng.choice(["A", "B"], size=n).astype(str)
    adata.obs["batch"] = rng.choice(["b0", "b1"], size=n).astype(str)
    adata.obs["cell_type"] = rng.choice(["T", "B", "Mono"], size=n).astype(str)
    return adata


def try_call_tl(fn, adata: ad.AnnData) -> Tuple[bool, str]:
    """
    Best-effort call for tl functions with reasonable defaults.
    If it fails, we record the error and keep going.
    """
    name = fn.__name__
    try:
        if name in {"shift", "mixscore", "density_overlap", "distribution_test", "wasserstein"}:
            fn(adata)  # defaults should work with our toy fields
        elif name in {"consensus_subspace"}:
            fn(adata, rep="X_pca", condition_key="condition", group0="A", group1="B")
        elif name in {"projection_disagreement"}:
            # requires sources; skip
            return False, "needs sources"
        elif name in {"align_vectors"}:
            return False, "needs vec_key"
        elif name in {"paga_composition_stats"}:
            # needs graph + groups
            return False, "needs paga / group_key"
        elif name in {"velocity_delta_alignment"}:
            return False, "needs velocity vectors"
        elif name.startswith("map_"):
            return False, "mapping funcs need proper ref/pool/graph"
        else:
            # unknown tl; attempt no-arg call
            fn(adata)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scgeo_io_manifest.json")
    args = ap.parse_args()

    out: Dict[str, Any] = {
        "scgeo_version": getattr(sg, "__version__", "unknown"),
        "tl": {},
        "skipped": {},
    }

    # find all tl.* functions (public-ish)
    tl_fns = []
    for k in dir(sg.tl):
        if k.startswith("_"):
            continue
        obj = getattr(sg.tl, k)
        if callable(obj):
            tl_fns.append((k, obj))

    for name, fn in sorted(tl_fns, key=lambda x: x[0]):
        adata = make_toy_adata()
        before = snapshot(adata)

        ok, note = try_call_tl(fn, adata)
        after = snapshot(adata)

        entry = {
            "ok": ok,
            "note": note,
            "signature": str(inspect.signature(fn)),
            "writes": diff(before, after),
        }
        if ok:
            out["tl"][f"scgeo.tl.{name}"] = entry
        else:
            out["skipped"][f"scgeo.tl.{name}"] = entry

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} (ok={len(out['tl'])}, skipped={len(out['skipped'])})")


if __name__ == "__main__":
    main()
