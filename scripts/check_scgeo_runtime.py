#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path: str, timeout: int, kernel_name: str = "python3", out_dir: str = ".check_notebooks"):
    p = Path(path)
    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    out_path = outd / f"{p.stem}.executed.ipynb"

    nb = nbformat.read(str(p), as_version=4)

    # Execute in notebook's directory so relative paths behave
    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(p.parent)}})
    except Exception as e:
        nbformat.write(nb, str(out_path))
        return False, str(out_path), e

    nbformat.write(nb, str(out_path))
    return True, str(out_path), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. "notebooks/*.ipynb"')
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--out-dir", default=".check_notebooks")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[ERROR] No notebooks matched: {args.glob}", file=sys.stderr)
        sys.exit(2)

    n_fail = 0
    for p in paths:
        ok, out_path, err = run_notebook(p, timeout=args.timeout, kernel_name=args.kernel, out_dir=args.out_dir)
        if ok:
            print(f"[OK]   {p}")
        else:
            n_fail += 1
            print(f"[FAIL] {p}  ->  {out_path}", file=sys.stderr)
            print(f"       {type(err).__name__}: {err}", file=sys.stderr)
            if args.fail_fast:
                break

    if n_fail:
        print(f"\n[check_scgeo_runtime] {n_fail}/{len(paths)} notebooks failed.", file=sys.stderr)
        print(f"Executed outputs are in: {args.out_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"\n[check_scgeo_runtime] All {len(paths)} notebooks executed cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
