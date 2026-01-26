from __future__ import annotations

import inspect
import json
import pkgutil
import importlib
from types import ModuleType
from typing import Any, Dict, List


def iter_public_callables(mod: ModuleType) -> List[tuple[str, Any]]:
    out = []
    if not hasattr(mod, "__all__"):
        return out
    for name in getattr(mod, "__all__", []):
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if callable(obj):
            out.append((name, obj))
    return out


def module_manifest(mod: ModuleType, prefix: str) -> List[Dict[str, Any]]:
    rows = []
    for name, fn in iter_public_callables(mod):
        fq = f"{prefix}.{name}"
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(signature unavailable)"
        doc = inspect.getdoc(fn) or ""
        doc1 = doc.splitlines()[0] if doc else ""
        rows.append(
            {
                "name": fq,
                "signature": sig,
                "doc": doc1,
            }
        )
    rows.sort(key=lambda r: r["name"])
    return rows


def import_submodules(pkg_name: str) -> List[ModuleType]:
    pkg = importlib.import_module(pkg_name)
    mods = [pkg]
    if not hasattr(pkg, "__path__"):
        return mods
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        mods.append(importlib.import_module(m.name))
    return mods


def main() -> None:
    import scgeo

    # Only trust __all__ exports as “public API”
    tl = importlib.import_module("scgeo.tl")
    pl = importlib.import_module("scgeo.pl")

    manifest = {
        "scgeo_version": getattr(scgeo, "__version__", None),
        "tl": module_manifest(tl, "scgeo.tl"),
        "pl": module_manifest(pl, "scgeo.pl"),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
