#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "api_manifest.json"
RAW_PATH = ROOT / "scgeo_io_raw.json"
CLEAN_PATH = ROOT / "scgeo_io_manifest.json"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tl_set(manifest: dict) -> set[str]:
    return set(manifest.get("tl", {}).keys()) | set(manifest.get("skipped", {}).keys())


def validate() -> None:
    api = _load(API_PATH)
    raw = _load(RAW_PATH)
    clean = _load(CLEAN_PATH)

    api_entries = api.get("tl", []) + api.get("pl", [])

    # (a) every api_manifest entry is importable
    for entry in api_entries:
        name = entry["name"]
        module_name, symbol = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol):
            raise AssertionError(f"Missing symbol: {name}")

    # (b) api/raw/clean sets are identical (TL contract)
    api_tl_set = {row["name"] for row in api.get("tl", [])}
    raw_set = _tl_set(raw)
    clean_set = _tl_set(clean)

    if api_tl_set != raw_set:
        raise AssertionError(
            f"api tl set != raw set\nmissing_in_raw={sorted(api_tl_set - raw_set)}\nextra_in_raw={sorted(raw_set - api_tl_set)}"
        )
    if api_tl_set != clean_set:
        raise AssertionError(
            f"api tl set != clean set\nmissing_in_clean={sorted(api_tl_set - clean_set)}\nextra_in_clean={sorted(clean_set - api_tl_set)}"
        )

    # (c) no underscore APIs in api_manifest
    underscored = [row["name"] for row in api_entries if row["name"].split(".")[-1].startswith("_")]
    if underscored:
        raise AssertionError(f"Underscore API names found: {underscored}")

    print("Manifest validation passed.")


if __name__ == "__main__":
    validate()
