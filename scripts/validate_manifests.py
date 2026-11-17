#!/usr/bin/env python3
"""Validate manifest versions, complete export sets and runtime signatures."""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scgeo as sg

NAMESPACES = ("tl", "pl", "get", "pp", "data", "bench")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path = ROOT) -> None:
    manifests = {name: _load(root / name) for name in
                 ("api_manifest.json", "scgeo_io_raw.json", "scgeo_io_manifest.json")}
    for filename, manifest in manifests.items():
        if manifest.get("scgeo_version") != sg.__version__:
            raise AssertionError(f"Version mismatch in {filename}: {manifest.get('scgeo_version')!r} != {sg.__version__!r}")
    api = manifests["api_manifest.json"]
    signatures = {}
    for section in NAMESPACES:
        module = importlib.import_module(f"scgeo.{section}")
        expected = {f"scgeo.{section}.{name}" for name in module.__all__
                    if not name.startswith("_") and callable(getattr(module, name))}
        rows = api.get(section, [])
        observed = [row["name"] for row in rows]
        if len(observed) != len(set(observed)):
            raise AssertionError(f"Duplicate public exports in scgeo.{section}")
        if set(observed) != expected:
            raise AssertionError(f"Export mismatch in scgeo.{section}: missing={sorted(expected - set(observed))}; extra={sorted(set(observed) - expected)}")
        for row in rows:
            name = row["name"]
            function = getattr(module, name.rsplit('.', 1)[1])
            runtime_signature = str(inspect.signature(function))
            if row.get("signature") != runtime_signature:
                raise AssertionError(f"Signature mismatch in api_manifest.json for {name}: {row.get('signature')!r} != {runtime_signature!r}")
            if row.get("docstring") != (inspect.getdoc(function) or ""):
                raise AssertionError(f"Docstring mismatch in api_manifest.json for {name}")
            signatures[name] = runtime_signature
    tl_names = {row["name"] for row in api["tl"]}
    for filename in ("scgeo_io_raw.json", "scgeo_io_manifest.json"):
        manifest = manifests[filename]
        assessed, skipped = manifest.get("tl", {}), manifest.get("skipped", {})
        if set(assessed) & set(skipped):
            raise AssertionError(f"Duplicate assessed/skipped entries in {filename}")
        entries = {**assessed, **skipped}
        if set(entries) != tl_names:
            raise AssertionError(f"I/O export mismatch in {filename}: missing={sorted(tl_names - set(entries))}; extra={sorted(set(entries) - tl_names)}")
        for name, entry in entries.items():
            if entry.get("signature") != signatures[name]:
                raise AssertionError(f"Signature mismatch in {filename} for {name}: {entry.get('signature')!r} != {signatures[name]!r}")
    print(f"PASS: manifest version {sg.__version__}, {len(signatures)} public exports, all signatures and both I/O export sets match runtime.")


if __name__ == "__main__":
    validate()
