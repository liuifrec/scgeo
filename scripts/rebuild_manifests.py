#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np

import scgeo as sg

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "api_manifest.json"
RAW_PATH = ROOT / "scgeo_io_raw.json"
CLEAN_PATH = ROOT / "scgeo_io_manifest.json"
REFRESH_SIGNATURES = {
    "scgeo.tl.representation_stability",
    "scgeo.get.state_report",
    "scgeo.pl.local_distortion_map",
    "scgeo.pl.perturbation_report",
    "scgeo.pl.state_evidence_panel",
    "scgeo.pl.representation_stability_heatmap",
    "scgeo.pl.consensus_state_map",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _public_callables(module_name: str) -> List[Dict[str, str]]:
    mod = importlib.import_module(module_name)
    rows: List[Dict[str, str]] = []
    for name in getattr(mod, "__all__", []):
        fq_name = f"{module_name}.{name}"
        if name.startswith("_"):
            continue
        fn = getattr(mod, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            signature = str(inspect.signature(fn))
        except Exception:
            signature = "(signature unavailable)"
        doc = inspect.getdoc(fn) or ""
        doc1 = doc.splitlines()[0] if doc else ""
        rows.append({"name": fq_name, "signature": signature, "doc": doc1})
    rows.sort(key=lambda row: row["name"])
    return rows


def _make_toy_adata_raw(n: int = 80, d: int = 12) -> ad.AnnData:
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obsm["X_pca"] = rng.normal(size=(n, d)).astype(np.float32)
    adata.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    adata.obs["condition"] = rng.choice(["A", "B"], size=n).astype(str)
    adata.obs["batch"] = rng.choice(["b0", "b1"], size=n).astype(str)
    adata.obs["cell_type"] = rng.choice(["T", "B", "Mono"], size=n).astype(str)
    return adata


def _snapshot_raw(adata: ad.AnnData) -> Dict[str, List[str]]:
    return {
        "obs_cols": sorted(list(adata.obs.columns)),
        "obsm_keys": sorted(list(adata.obsm.keys())),
        "layers_keys": sorted(list(adata.layers.keys())),
        "uns_keys": sorted(list(adata.uns.keys())),
    }


def _diff_raw(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in a.keys():
        aa, bb = set(a[key]), set(b[key])
        out[f"{key}_added"] = sorted(list(bb - aa))
        out[f"{key}_removed"] = sorted(list(aa - bb))
    return out


def _call_policy_raw(fn, adata: ad.AnnData) -> Tuple[bool, str]:
    name = fn.__name__
    try:
        if name == "robust_shift":
            fn(adata, n_boot=25, seed=0)
        elif name == "representation_stability":
            fn(
                adata,
                reps=["X_pca", "X_umap"],
                node_key="cell_type",
                condition_key="condition",
                group0="A",
                group1="B",
                min_cells=2,
                n_boot=5,
                seed=0,
            )
        elif name == "local_geometry_stability":
            fn(
                adata,
                reps=["X_pca", "X_umap"],
                node_key="cell_type",
                k_values=(3,),
                n_boot=3,
                max_exact_cells=40,
                seed=0,
            )
        elif name in {"shift", "mixscore", "density_overlap", "distribution_test", "wasserstein"}:
            fn(adata)
        elif name in {"consensus_subspace"}:
            fn(adata, rep="X_pca", condition_key="condition", group0="A", group1="B")
        elif name in {"projection_disagreement"}:
            return False, "needs sources"
        elif name in {"align_vectors"}:
            return False, "needs vec_key"
        elif name in {"paga_composition_stats"}:
            return False, "needs paga / group_key"
        elif name in {"velocity_delta_alignment"}:
            return False, "needs velocity vectors"
        elif name.startswith("map_"):
            return False, "mapping funcs need proper ref/pool/graph"
        else:
            fn(adata)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _make_toy_adata_clean(n: int = 60) -> ad.AnnData:
    rng = np.random.RandomState(0)
    adata = ad.AnnData(X=rng.normal(size=(n, 10)))
    adata.obsm["X_pca"] = rng.normal(size=(n, 8))
    adata.obsm["X_umap"] = rng.normal(size=(n, 2))
    adata.obs["condition"] = rng.choice(["A", "B"], size=n)
    adata.obs["batch"] = rng.choice(["b0", "b1"], size=n)
    adata.obs["cluster"] = rng.choice(["c0", "c1", "c2"], size=n)
    return adata


def _snapshot_clean(adata: ad.AnnData) -> Dict[str, List[str]]:
    return {
        "obs_cols": list(adata.obs.columns),
        "obsm_keys": list(adata.obsm.keys()),
        "layers_keys": list(adata.layers.keys()),
        "uns_keys": list(adata.uns.keys()),
    }


def _diff_clean(before: Dict[str, List[str]], after: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for key in before:
        b, a = set(before[key]), set(after[key])
        out[key] = {
            "added": sorted(a - b),
            "removed": sorted(b - a),
            "touched": sorted(a & b),
        }
    return out


def _call_policy_clean(fn, adata: ad.AnnData) -> Tuple[bool, str]:
    name = fn.__name__
    try:
        if name == "shift":
            fn(adata, rep="X_pca", condition_key="condition", group0="A", group1="B", store_key="shift_test")
        elif name == "robust_shift":
            fn(
                adata,
                rep="X_pca",
                condition_key="condition",
                group0="A",
                group1="B",
                n_boot=25,
                seed=0,
                store_key="robust_shift_test",
            )
        elif name == "representation_stability":
            fn(
                adata,
                reps=["X_pca", "X_umap"],
                node_key="cluster",
                condition_key="condition",
                group0="A",
                group1="B",
                min_cells=2,
                n_boot=5,
                seed=0,
                store_key="representation_stability_test",
            )
        elif name == "local_geometry_stability":
            fn(
                adata,
                reps=["X_pca", "X_umap"],
                node_key="cluster",
                k_values=(3,),
                n_boot=3,
                max_exact_cells=40,
                seed=0,
                store_key="local_geometry_stability_test",
            )
        elif name == "mixscore":
            fn(adata, label_key="batch", rep="X_umap", store_key="mix_test")
        elif name == "density_overlap":
            fn(adata, rep="X_umap", condition_key="condition", group0="A", group1="B", store_key="dens_test")
        elif name in {"distribution_test", "wasserstein"}:
            fn(adata, rep="X_pca", condition_key="condition", group0="A", group1="B", store_key="dist_test")
        else:
            return False, "needs domain-specific inputs"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def rebuild() -> None:
    api_prev = _load_json(API_PATH)
    api_tl_prev = {entry["name"]: entry for entry in api_prev.get("tl", [])}
    api_pl_prev = {entry["name"]: entry for entry in api_prev.get("pl", [])}
    api_get_prev = {entry["name"]: entry for entry in api_prev.get("get", [])}
    api_manifest = {
        "scgeo_version": getattr(sg, "__version__", None),
        "tl": _public_callables("scgeo.tl"),
        "pl": _public_callables("scgeo.pl"),
        "get": _public_callables("scgeo.get"),
    }
    for section, prev_map in (("tl", api_tl_prev), ("pl", api_pl_prev), ("get", api_get_prev)):
        for row in api_manifest[section]:
            prev = prev_map.get(row["name"])
            if prev is not None and row["name"] not in REFRESH_SIGNATURES:
                row["signature"] = prev.get("signature", row["signature"])
                row["doc"] = prev.get("doc", row["doc"])
    _write_json(API_PATH, api_manifest)

    tl_names = [row["name"].split(".")[-1] for row in api_manifest["tl"]]
    raw_prev = _load_json(RAW_PATH)
    raw_prev_map = {**raw_prev.get("tl", {}), **raw_prev.get("skipped", {})}

    raw_tl: Dict[str, Any] = {}
    raw_skipped: Dict[str, Any] = {}
    for name in sorted(tl_names):
        fn = getattr(sg.tl, name)
        adata = _make_toy_adata_raw()
        before = _snapshot_raw(adata)
        ok, note = _call_policy_raw(fn, adata)
        after = _snapshot_raw(adata)
        fq = f"scgeo.tl.{name}"
        prev_entry = raw_prev_map.get(fq, {})
        signature = str(inspect.signature(fn)) if fq in REFRESH_SIGNATURES else prev_entry.get("signature", str(inspect.signature(fn)))
        entry = {
            "ok": ok,
            "note": note,
            "signature": signature,
            "writes": _diff_raw(before, after),
        }
        if ok:
            raw_tl[fq] = entry
        else:
            raw_skipped[fq] = entry

    raw_manifest = {
        "scgeo_version": getattr(sg, "__version__", "unknown"),
        "tl": raw_tl,
        "skipped": raw_skipped,
    }
    _write_json(RAW_PATH, raw_manifest)

    clean_prev = _load_json(CLEAN_PATH)
    clean_prev_map = {**clean_prev.get("tl", {}), **clean_prev.get("skipped", {})}

    clean_tl: Dict[str, Any] = {}
    clean_skipped: Dict[str, Any] = {}
    for name in sorted(tl_names):
        fn = getattr(sg.tl, name)
        adata = _make_toy_adata_clean()
        before = _snapshot_clean(adata)
        ok, note = _call_policy_clean(fn, adata)
        after = _snapshot_clean(adata)
        fq = f"scgeo.tl.{name}"
        prev_entry = clean_prev_map.get(fq, {})
        signature = str(inspect.signature(fn)) if fq in REFRESH_SIGNATURES else prev_entry.get("signature", str(inspect.signature(fn)))
        entry = {
            "signature": signature,
            "ok": ok,
            "note": note,
            "writes": _diff_clean(before, after),
        }
        if ok:
            clean_tl[fq] = entry
        else:
            clean_skipped[fq] = entry

    clean_manifest = {
        "scgeo_version": getattr(sg, "__version__", "unknown"),
        "tl": clean_tl,
        "skipped": clean_skipped,
    }
    _write_json(CLEAN_PATH, clean_manifest)


if __name__ == "__main__":
    rebuild()
