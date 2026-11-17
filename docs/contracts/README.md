# API and I/O provenance

The [complete API reference](../api_reference.md) and [module pages](../api/README.md)
are generated together from the current package's public exports. Start with the
[quick start](../QUICKSTART.md) for a runnable analysis.

## Sources of documentation

| Source | What it records |
|---|---|
| `api_manifest.json` | Current package version, all public callables in `tl`, `pl`, `get`, `pp`, `data` and `bench`, exact runtime signatures, docstrings and source paths |
| `scgeo_io_raw.json` | Key additions/removals observed by the original small synthetic `tl` probe, plus explicit skipped cases |
| `scgeo_io_manifest.json` | Key presence before/after the normalized synthetic `tl` probe, plus explicit skipped cases |
| [api_notes.json](api_notes.json) | Curated interpretation and reporting notes; signatures and export lists are always generated from runtime |

The I/O probes record **key presence**, not an exhaustive read/write contract.
The legacy `touched` field lists keys present both before and after a call; it
does not establish that values changed. A skipped probe means its small fixture
does not supply the inputs needed by that function. It is not a failed API test.
These files describe the current package, separately from the paper's frozen
analysis configuration and evidence.

## Regenerate and validate

Use a development environment with ScGeo's dependencies, on Linux, WSL or macOS
(the benchmark namespace uses the Unix `resource` module). From the repository
root:

```bash
python scripts/rebuild_manifests.py
python scripts/validate_manifests.py
python scripts/gen_api_docs.py
python scripts/gen_api_docs.py --check
```

The rebuild inspects the checkout's public exports and runs only small synthetic
I/O probes. It does not download Census data or rerun manuscript analyses. The
validator checks version, parameter/default signatures and complete export
coverage, including APIs omitted from older references. The final command
checks generated Markdown without modifying it.

Edit curated prose in `api_notes.json`, then regenerate both views. Avoid editing
signatures or generated pages by hand: a later regeneration would overwrite
those changes.
