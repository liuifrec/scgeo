#!/usr/bin/env python3
"""
Convert GSE280305 *.loom files to per-sample .h5ad with preserved layers.

- Reads each GSM*_count.loom one-by-one (memory-safe)
- Fixes non-unique var_names
- Normalizes layer naming:
    - Uses adata.X = adata.layers["matrix"] if present
    - Keeps layers: spliced/unspliced/ambiguous/matrix (if present)
- Adds obs: timepoint (D8/D11/D14/D21), sample (GSM...), gsm (GSM...)
- Ensures unique cell IDs across samples
- Writes: data/GSE280305_h5ad/<stem>.h5ad
"""

from __future__ import annotations

from pathlib import Path
import re
import scanpy as sc
import numpy as np

IN_DIR = Path("data/GSE280305_RAW")
OUT_DIR = Path("data/GSE280305_h5ad")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_RE = re.compile(r"_D(\d+)_", re.IGNORECASE)  # matches _D8_, _D11_, etc.
GSM_RE = re.compile(r"^(GSM\d+)_", re.IGNORECASE)

PREFERRED_LAYERS = ["matrix", "spliced", "unspliced", "ambiguous"]


def parse_timepoint(fname: str) -> str:
    m = TP_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse timepoint from filename: {fname}")
    return f"D{m.group(1)}"


def parse_gsm(fname: str) -> str:
    m = GSM_RE.match(fname)
    if not m:
        raise ValueError(f"Cannot parse GSM from filename: {fname}")
    return m.group(1)


def to_float32_sparse_if_possible(adata):
    # Avoid accidental float64 bloat
    if hasattr(adata.X, "dtype") and adata.X.dtype != np.float32:
        try:
            adata.X = adata.X.astype(np.float32)
        except Exception:
            pass
    for k in list(adata.layers.keys()):
        mat = adata.layers[k]
        if hasattr(mat, "dtype") and mat.dtype != np.float32:
            try:
                adata.layers[k] = mat.astype(np.float32)
            except Exception:
                pass


def main():
    loom_files = sorted(IN_DIR.glob("GSM*_count.loom"))
    if not loom_files:
        raise SystemExit(f"No .loom files found in {IN_DIR}. Did you gunzip them?")

    for f in loom_files:
        tp = parse_timepoint(f.name)
        gsm = parse_gsm(f.name)
        stem = f.stem  # e.g. GSM..._D8_count

        print(f"\n=== [{stem}] loading {f} ===")
        ad = sc.read_loom(str(f))

        # Fix duplicated gene names (common in loom)
        ad.var_names_make_unique()

        # Add minimal obs annotations (you already saw only timepoint/sample existed)
        ad.obs["timepoint"] = tp
        ad.obs["sample"] = stem
        ad.obs["gsm"] = gsm

        # Make cell IDs globally unique across samples
        ad.obs_names = [f"{tp}:{gsm}:{cid}" for cid in ad.obs_names]

        # Normalize layer usage:
        # Many loom files store counts in layers["matrix"] and set X to something else.
        # We standardize: X = layers["matrix"] if present.
        if "matrix" in ad.layers:
            ad.X = ad.layers["matrix"]

        # Keep only the layers we care about (drop surprises that bloat disk)
        keep = [k for k in PREFERRED_LAYERS if k in ad.layers]
        drop = [k for k in ad.layers.keys() if k not in keep]
        for k in drop:
            del ad.layers[k]

        # Ensure float32 everywhere feasible
        to_float32_sparse_if_possible(ad)

        # Save
        out = OUT_DIR / f"{stem}.h5ad"
        ad.write(out)
        print(f"wrote: {out}  shape={ad.shape}  layers={list(ad.layers.keys())}")

        # Free memory explicitly (helpful under WSL)
        del ad

    print("\nDONE. Next: build joint AnnData + embeddings from data/GSE280305_h5ad/*.h5ad")


if __name__ == "__main__":
    main()