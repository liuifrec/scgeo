#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="notebooks/*.ipynb")
    ap.add_argument("--api-manifest", default="api_manifest.json")
    ap.add_argument("--io-manifest", default="scgeo_io_manifest.json")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    # Expand and filter notebooks (ignore checkpoints)
    nbs = sorted(glob.glob(args.glob))
    nbs = [p for p in nbs if ".ipynb_checkpoints" not in p]
    if not nbs:
        print(f"[ERROR] No notebooks matched: {args.glob}")
        return 2

    # Use a filtered glob if the user gave a broad one that might include checkpoints.
    # (We keep passing the original glob to other scripts, but they should ignore checkpoints too.)
    glob_arg = args.glob

    # 0) Syntax check (fast, catches broken quotes/parentheses)
    cmd = [sys.executable, "scripts/check_notebooks_syntax.py", "--glob", glob_arg]
    if args.fail_fast:
        cmd.append("--fail-fast")
    rc = run(cmd)
    if rc != 0:
        return rc

    # 1) API check (manifest-based kwargs/signature validation)
    cmd = [
        sys.executable,
        "scripts/check_scgeo_notebooks.py",
        "--manifest",
        args.api_manifest,
        "--glob",
        glob_arg,
    ]
    if args.fail_fast:
        cmd.append("--fail-fast")
    rc = run(cmd)
    if rc != 0:
        return rc

    # 2) IO check (static: verifies tl->writes and pl->reads)
    cmd = [
        sys.executable,
        "scripts/check_scgeo_notebooks_io.py",
        "--io-manifest",
        args.io_manifest,
        "--glob",
        glob_arg,
    ]
    if args.fail_fast:
        cmd.append("--fail-fast")
    rc = run(cmd)
    if rc != 0:
        return rc

    # 3) Runtime execution check
    cmd = [
        sys.executable,
        "scripts/check_scgeo_runtime.py",
        "--glob",
        glob_arg,
        "--timeout",
        str(args.timeout),
    ]
    if args.fail_fast:
        cmd.append("--fail-fast")
    rc = run(cmd)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
