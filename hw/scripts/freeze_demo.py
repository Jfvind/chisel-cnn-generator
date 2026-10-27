#!/usr/bin/env python3
"""freeze_demo.py — copy a verified MLP build into demo/mlp so `make demo` works without Vivado.

    python hw/scripts/freeze_demo.py --build hw/vivado/demo --quant generator/generated/quantized --out demo/mlp

demo/mlp gets the layout of a build directory, so report.py and deploy.py work on it unchanged:
    overlay/   accel.bit, accel.hwh, hw_manifest.json
    reports/   *.rpt, board_results.json (the 10,000-image run), dashboard.html
    rtl/       shell_params.tcl
    quant/     network.json + weight/bias CSVs + Python vectors the bitstream was built from
    FROZEN.json  commit, date and headline results
Refuses unless the board run of that build passed. Commit demo/mlp afterwards.
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
    results = build / "reports" / "board_results.json"
    res = json.loads(results.read_text()) if results.exists() else {}
    if not res.get("passed") and not a.force:
        sys.exit(f"{results}: no passed board run (run `make demo-build` first, or --force)")
    if res.get("images", 0) < 10000 and not a.force:
        print(f"note: the frozen results cover {res.get('images')} images, not the full test set")
    manifest = json.loads((build / "overlay" / "hw_manifest.json").read_text())
    if manifest.get("core") != "mlp":
        sys.exit(f"{build / 'overlay'} holds core '{manifest.get('core')}', not mlp")

    # (source, folder inside out)
    files = [(build / "overlay" / f"{a.name}.bit", "overlay"), (build / "overlay" / f"{a.name}.hwh", "overlay"),
             (build / "overlay" / "hw_manifest.json", "overlay"),
             (build / "rtl" / "shell_params.tcl", "rtl"),
             (quant / "network.json", "quant")]
    files += [(f, "quant") for f in sorted(quant.glob("*.csv"))]
    files += [(f, "reports") for f in sorted((build / "reports").glob("*.rpt"))]
    files += [(results, "reports")]
    cpu = pathlib.Path("python/results/cpu_benchmark.json")       # PC numbers next to the board numbers
    if cpu.exists():
        files += [(cpu, "reports")]
    missing = [str(f) for f, _ in files if not f.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))

    if out.exists():
        shutil.rmtree(out)
    for f, sub in files:
        (out / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / sub / f.name)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    info = {"frozen": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit,
            "build": str(build), "cfg_hash": manifest.get("cfg_hash"), "images": res.get("images"),
            "accuracy": res.get("hw_accuracy"), "cycles": res.get("cycles_median"),
            "shift_policy": res.get("shift_policy"),
            "files": sorted(f"{sub}/{f.name}" for f, sub in files)}
    (out / "FROZEN.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"frozen {len(files)} files into {out} (commit {commit}). Now: git add {out} && git commit, "
          f"then git tag mlp-demo-v1")


if __name__ == "__main__":
    main()