#!/usr/bin/env python3
import json
from pathlib import Path

RAW = Path("scgeo_io_raw.json")
OUT = Path("scgeo_io_manifest.json")

raw = json.loads(RAW.read_text())

entries = []

for fn, info in raw.get("tl", {}).items():
    writes = info.get("writes", {})
    for k, added in writes.items():
        if not added:
            continue
        if k.endswith("_added"):
            base = k.replace("_added", "")
            for item in added:
                entries.append(
                    {
                        "function": fn,
                        "mode": "write",
                        "path": f"{base}.{item}",
                        "type": "unknown",
                        "note": "auto-detected",
                    }
                )

OUT.write_text(json.dumps(entries, indent=2))
print(f"Wrote normalized IO manifest: {OUT}")
