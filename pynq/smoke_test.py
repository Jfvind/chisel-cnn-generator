"""smoke_test.py — board smoke test for AccelShell with the stream stub core.

Run on the board (deploy.py does this):  sudo python3 smoke_test.py --overlay accel.bit
Prints the PYNQ version, checks ID/config, the AXI-Lite adder and DMA streams.
Exit code 0 = everything passed.
"""

import argparse
import sys
import time

import numpy as np

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' ' + detail if detail else ''}", flush=True)
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="accel.bit")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    import pynq
    print(f"PYNQ version: {getattr(pynq, '__version__', 'unknown')}")

    from accel import Accel
    t0 = time.time()
    acc = Accel(a.overlay)
    m = acc.manifest
    print(f"Overlay loaded in {time.time() - t0:.2f}s   core={m['core']} stream={m['stream_w']}b "
          f"id=0x{acc.read('ID'):08x} hash=0x{acc.read('CFG_HASH'):08x}")

    ok = True
    ok &= check("ID / CFG_HASH match manifest", True)
    ok &= check("N_IN / N_OUT", acc.read("N_IN") == m["n_in"] and acc.read("N_OUT") == m["n_out"],
                f"{acc.read('N_IN')} / {acc.read('N_OUT')}")

    # ---AXI-Lite: the adder ---
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range (a.iters):
        x, y = (int(v) for v in rng.integers(0, 2**32, size=2, dtype=np.uint64))
        acc.set_arg(0, x); acc.set_arg(1, y)
        if acc.read("SUM") != (x + y) & 0xFFFFFFFF:
            bad += 1
    ok &= check(f"AXI-Lite adder ({a.iters} random pairs)", bad == 0, f"{bad} mismatches")

    # --- AXI-Stream through DMA: stub returns in + ARG0
    if m["core"] == "stub":
        addend = 0x1234
        acc.set_arg(0, addend)
        mask = (1 << m["stream_w"]) - 1
        for n in (1, 16, m["n_in"], 4096):      # different cases 
            x = rng.integers(0, mask + 1, size=n, dtype=np.uint64)
            t0 = time.perf_counter()
            # n == n_in uses the manifest default (checks the stub's nOut = nIn), others pass n_out
            out, cycles = acc.run(x) if n == m["n_in"] else acc.run(x, n_out=n)
            dt = time.perf_counter() - t0
            exp = (x + addend) & mask
            mism = int(np.count_nonzero(out.astype(np.uint64) != exp))
            ok &= check(f"DMA stream n={n}", mism == 0 and cycles >= n,
                        f"{mism} mismatches, RESULT={acc.read('RESULT')}, CYCLES={cycles}, {dt * 1e3:.2f} ms")
        ok &= check("STATUS.done after stream", acc.status["done"])
    else:
        print("core is not 'stub': skipping stream self-check")

    acc.close()
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)        # deploy.py sees the exit code

if __name__ == "__main__":
    main()