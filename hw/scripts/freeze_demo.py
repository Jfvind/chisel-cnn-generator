#!/usr/bin/env python3
"""freeze_demo.py — copy a verified MLP build into demo/mlp so `make demo` works without Vivado.

    python hw/scripts/freeze_demo.py --build hw/vivado/demo --quant generator/generated/quantized --out demo/mlp

Copies the overlay (.bit/.hwh/hw_manifest.json), the quantized export it was built from (network.json,
weights, biases, Python test vectors), the board results and the dashboard. Refuses unless the board
run of that build passed (reports/board_results.json "passed": true), so the frozen demo is known-good.
Writes FROZEN.json with the git commit, date and results. Commit demo/mlp afterwards.
"""

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="hw/vivado/demo")
    ap.add_argument("--quant", default="generator/generated/quantized")
    ap.add_argument("--out", default="demo/mlp")
    ap.add_argument("--name", default="accel")
    ap.add_argument("--force", action="store_true", help="freeze even without a passed board run")
    a = ap.parse_args()

    build, quant, out = pathlib.Path(a.build), pathlib.Path(a.quant), pathlib.Path(a.out)
    overlay, reports = build / "overlay", build / "reports"
    results = reports / "board_results.json"
    res = json.loads(results.read_text()) if results.exists() else {}
    if not res.get("passed") and not a.force:
        sys.exit(f"{results}: no passed board run (run `make demo-build` first, or --force)")
    manifest = json.loads((overlay / "hw_manifest.json").read_text())
    if manifest.get("core") != "mlp":
        sys.exit(f"{overlay} holds core '{manifest.get('core')}', not mlp")

    files = [overlay / f"{a.name}.bit", overlay / f"{a.name}.hwh", overlay / "hw_manifest.json",
             quant / "network.json", *sorted(quant.glob("*.csv"))]
    files += [f for f in (results, reports / "dashboard.html") if f.exists()]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, out / f.name)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    info = {"frozen": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit,
            "build": str(build), "cfg_hash": manifest.get("cfg_hash"),
            "accuracy": res.get("hw_accuracy"), "cycles": res.get("cycles_median"),
            "shift_policy": res.get("shift_policy"), "files": sorted(f.name for f in files)}
    (out / "FROZEN.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"frozen {len(files)} files into {out} (commit {commit}). Now: git add {out} && git commit, "
          f"then git tag mlp-demo-v1")


if __name__ == "__main__":
    main()