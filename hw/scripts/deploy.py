#!/usr/bin/env python3
"""deploy.py — copy an overlay + the PYNQ driver to the board and run a script there.

    python hw/scripts/deploy.py --overlay hw/vivado/overlay/accel.bit [--host pynq-direct,pynq] [--run smoke_test.py]
                                [--args="--demo 5"] [--extra file ...] [--results board_results.json]

--host is a comma-separated list; the first host that answers on ssh is used.
--extra copies data files next to the script; the script's --results file is copied back to
<build>/reports/ (also when the script fails), where hw/scripts/report.py shows it.
Uses the OpenSSH client. One-time SSH key, sudo and
direct-cable setup: see hw/README.md.
"""

import argparse
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]  # hw/scripts/deploy.py -> repo root

def run(cmd, check=True):
    print("+", " ".join(cmd), flush=True)   # echo command
    r = subprocess.run(cmd)                 # still inherits console
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode

def pick_host(user, hosts):
    # hosts = "pynq-direct,pynq": use the first one that answers on ssh.
    # BatchMode: fail instead of asking for a password. ConnectTimeout: skip quickly.
    for h in hosts.split(","):
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes",
                            f"{user}@{h}", "true"], capture_output=True)
        if r.returncode == 0:
            print(f"board found at {h}", flush=True)
            return h
        print(f"no answer from {h}", flush=True)
    sys.exit(f"no board reachable at any of: {hosts}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--host", default="pynq-direct,pynq")
    ap.add_argument("--user", default="xilinx")
    ap.add_argument("--dest", default="/home/xilinx/jupyter_notebooks/cnn_accel")
    ap.add_argument("--run", default="smoke_test.py")
    ap.add_argument("--args", default="", help='extra arguments for the script, e.g. "--demo 5"')
    ap.add_argument("--extra", nargs="*", default=[], help="data files the script needs (copied next to it)")
    ap.add_argument("--results", default="board_results.json",
                    help="file the script writes; copied back to <build>/reports/ for the dashboard")
    a = ap.parse_args()

    bit = pathlib.Path(a.overlay).resolve()
    hwh = bit.with_suffix(".hwh")
    manifest = bit.parent / "hw_manifest.json"
    extra = [pathlib.Path(f).resolve() for f in a.extra]
    for f in (bit, hwh, manifest, *extra):
        if not f.exists():
            sys.exit(f"missing {f}")

    target = f"{a.user}@{pick_host(a.user, a.host)}"
    run(["ssh", target, f"mkdir -p {a.dest}"])
    files = [str(bit), str(hwh), str(manifest)] + [str(p) for p in sorted((REPO / "pynq").glob("*.py"))]
    files += [str(f) for f in extra]                # e.g. test vectors, weights, MNIST for mlp_test.py
    run(["scp", *files, f"{target}:{a.dest}/"])     # copy overlay, driver, test

    if a.run:
        # PYNQ 3.x needs two environment scripts: pynq_venv.sh (its Python venv) and
        # xrt_setup.sh (sets XILINX_XRT; without it pynq never registers the Zynq
        # device and Overlay() fails with "No Devices Found").
        venv = ("for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh; "
                "do [ -f $f ] && . $f; done")
        inner = (f"{venv}; cd {a.dest} && rm -f {a.results} && "
                 f"python3 {a.run} --overlay {bit.stem}.bit {a.args}")
        interactive = sys.stdin.isatty()    # started in terminal?
        sudo = "sudo" if interactive else "sudo -n"     # -n makes it fail if password is needed
        ssh = ["ssh", "-t", target] if interactive else ["ssh", target]
        code = run(ssh + [f"{sudo} bash -c '{inner}'"], check=False)

        # Results back to the build for the dashboard: hw/vivado/overlay/accel.bit -> hw/vivado/reports/
        reports = bit.parent.parent / "reports" if bit.parent.name == "overlay" else bit.parent
        reports.mkdir(parents=True, exist_ok=True)
        got = subprocess.run(["scp", f"{target}:{a.dest}/{a.results}", str(reports / a.results)],
                             capture_output=True).returncode == 0
        print(f"results -> {reports / a.results}" if got else f"(no {a.results} written by {a.run})", flush=True)
        if code != 0:
            sys.exit(f"{a.run} failed on the board ({code})")

if __name__ == "__main__":
    main()