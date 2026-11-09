"""mlp_test.py — run the MLP on the board: bit-exactness, accuracy and speed vs. the ARM CPU.

Run on the board (deploy.py does this):  sudo python3 mlp_test.py --overlay accel.bit
Needs next to it (Makefile mlp_EXTRA copies them): network.json + <i>_weight.csv / <i>_bias.csv
(the quantized export), test_inputs.csv + expected_outputs.csv (Python reference vectors) and
t10k-images-idx3-ubyte.gz + t10k-labels-idx1-ubyte.gz (MNIST test set).

1. Vectors : the 64 Python reference images through the FPGA, every score bit-exact.
2. MNIST   : the 10,000 test images through the FPGA in batches of --batch (accuracy, throughput)   plus a latency sample one image at a time.
             and through the same integer network in NumPy on the ARM (accuracy, time);
             every FPGA score row is compared with the ARM result, so this is bit-exact on 10k.
3. Writes board_results.json; deploy.py copies it to <build>/reports/ for the dashboard.
   --demo N prints N digits as ASCII art with prediction and label.
Exit code 0 = bit-exact everywhere.
"""

import argparse
import gzip
import json
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------- integer reference
# Same arithmetic as python/scripts/integer_reference.py (that module is not on the board).

def load_network(d):
    net = json.loads((d / "network.json").read_text())
    assert net.get("activation_overflow") == "saturate", "export is older than the saturate contract"
    for l in net["layers"]:
        l["w"] = np.loadtxt(d / f"{l['index']}_weight.csv", delimiter=",", dtype=np.int64, ndmin=2)
        l["b"] = np.loadtxt(d / f"{l['index']}_bias.csv", dtype=np.int64, ndmin=1)
    return net


def quantize_pixels(raw_u8, bits):
    """MNIST bytes 0..255 -> floor(x * A + 0.5), x = byte / 255 (= the byte itself for 8 bit)."""
    a = (1 << bits) - 1
    return np.clip(np.floor(raw_u8.astype(np.float64) / 255.0 * a + 0.5), 0, a).astype(np.int64)


def infer(q, net):
    """Rows of integer pixels -> rows of raw last-layer scores (hidden: ReLU, >> shift, saturate)."""
    a = (1 << net["activation_bits"]) - 1
    for l in net["layers"]:
        acc = q @ l["w"] + l["b"]
        q = np.minimum(np.maximum(acc, 0) >> l["shift"], a) if l["activation"] == "relu" else acc
    return q


# --------------------------------------------------------------------------- data

def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    if int.from_bytes(data[2:3], "big") != 0x08:
        raise ValueError(f"{path}: not unsigned-byte idx data")
    dims = [int.from_bytes(data[4 + 4 * k: 8 + 4 * k], "big") for k in range(data[3])]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * len(dims)).reshape(dims)


def signed(raw, bits):
    """Stream beats (unsigned, stream_w bits) -> signed scores."""
    raw = raw.astype(np.int64) & ((1 << bits) - 1)
    return np.where(raw >= 1 << (bits - 1), raw - (1 << bits), raw)


def ascii_digit(img):
    shades = " .:-=+*#%@"
    return "\n".join("".join(shades[min(9, int(p) * 10 // 256)] for p in row) for row in img.reshape(28, 28))


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' ' + detail if detail else ''}", flush=True)
    return ok


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="accel.bit")
    ap.add_argument("--images", type=int, default=10000, help="MNIST test images to run (max 10000)")
    ap.add_argument("--batch", type=int, default=1000, help="images per DMA transfer in the throughput pass")
    ap.add_argument("--latency-images", type=int, default=500, help="images timed one at a time")
    ap.add_argument("--arm-single", type=int, default=200, help="images timed one by one on the ARM")
    ap.add_argument("--demo", type=int, default=0, help="print this many digits as ASCII art")
    ap.add_argument("--out", default="board_results.json")
    a = ap.parse_args()

    import pynq
    from accel import Accel

    acc = Accel(a.overlay)
    m = acc.manifest
    if m["core"] != "mlp":
        sys.exit(f"overlay core is '{m['core']}', not 'mlp'")
    net = load_network(HERE)
    bits, sw, fclk = net["input_bits"], m["stream_w"], m["fclk_mhz"]
    ppb = m.get("pixels_per_beat", 1)       # Fallback to 1 keeps the old frozen version working
    print(f"PYNQ {getattr(pynq, '__version__', '?')}  core=mlp  {m['n_in']} -> {m['n_out']}  stream={sw}b  "
          f"{bits}-bit  shift={net.get('shift_policy', '?')}  fclk={fclk} MHz", flush=True)
    ok = check("network matches overlay", net["layers"][0]["in_features"] == m["n_in"]
               and net["layers"][-1]["out_features"] == m["n_out"])

    def pack(q, ppb, bits):
        """(N, n_in) pixels -> (N, ceil(n_in / ppb)) uint32 words, pixel 0 in the lowest bits; images padded to whole words."""
        n, k = q.shape
        w = -(-k // ppb)
        if bits == 8 and ppb == 4 and k % 4 == 0:
            return np.ascontiguousarray(q, dtype=np.uint8).view("<u4")     # no copy: 4 bytes = 1 word
        padded = np.zeros((n, w * ppb), dtype=np.uint64)
        padded[:, :k] = q
        shifts = np.arange(ppb, dtype=np.uint64) * np.uint64(bits)
        return (padded.reshape(n, w, ppb) << shifts).sum(axis=2).astype(np.uint32)

    def run_hw(x):
        acc.set_arg(0, 1)
        out, cycles = acc.run(x)                        # run() writes CTRL start first: clears OVF_COUNT
        return signed(out, sw), cycles, acc.read("RESULT"), acc.read("OVF_COUNT")

    def run_hw_batch(xs):
        """len(xs) images in one DMA transfer -> scores[N, n_out], cycles for the whole batch."""
        acc.set_arg(0, len(xs))             # ARG0 = batch size, latched by the start in run()
        out, cycles = acc.run(xs.ravel(), n_out=len(xs) * m["n_out"])
        return signed(out, sw).reshape(len(xs), m["n_out"]), cycles

    # 1. Python reference vectors: bit-exact scores
    vin = np.loadtxt(HERE / "test_inputs.csv", delimiter=",", dtype=np.int64, ndmin=2)
    vexp = np.loadtxt(HERE / "expected_outputs.csv", delimiter=",", dtype=np.int64, ndmin=2)
    ok &= check("ARM reference reproduces Python vectors", np.array_equal(infer(vin, net), vexp),
                "(otherwise the export and the overlay come from different configs)")
    vx = pack(vin, ppb, bits)       # pixels -> stream words
    bad = [k for k in range(len(vin)) if not np.array_equal(run_hw(vx[k])[0], vexp[k])]
    ok &= check(f"FPGA bit-exact on {len(vin)} Python vectors", not bad, f"mismatches at {bad[:10]}" if bad else "")

    # 2. MNIST test set
    imgs = read_idx(HERE / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)[:a.images]
    labels = read_idx(HERE / "t10k-labels-idx1-ubyte.gz")[:a.images].astype(np.int64)
    q = quantize_pixels(imgs, bits)
    n = len(q)
    qw = pack(q, ppb, bits)     # pixels -> stream words

    t0 = time.perf_counter()
    ref = infer(q, net)                                  # ARM, one batched matrix product per layer
    arm_batched = (time.perf_counter() - t0) / n
    k = min(a.arm_single, n)
    t0 = time.perf_counter()
    for i in range(k):
        infer(q[i:i + 1], net)                           # ARM, one image at a time
    arm_single = (time.perf_counter() - t0) / k
    arm_pred = ref.argmax(1)

    # 2a. Latency: one image per DMA round trip, on a sample (the whole set this way takes ~20 s)
    k_lat = min(a.latency_images, n)
    walls, cyc_single, result_bad = [], [], 0
    for i in range(k_lat):
        t0 = time.perf_counter()
        scores, cyc, result, _ = run_hw(qw[i])
        walls.append(time.perf_counter() - t0)
        cyc_single.append(cyc)
        result_bad += result != scores.argmax()         # RESULT is only meaningful for single images

    # 2b. Throughput + bit-exactness on all n images: a.batch images per DMA transfer
    hw_scores = np.empty_like(ref)
    batch_cycles, ovf = 0, 0
    t_all = time.perf_counter()
    for s in range(0, n, a.batch):
        xs = qw[s:s + a.batch]
        hw_scores[s:s + len(xs)], cyc = run_hw_batch(xs)
        batch_cycles += cyc                             # CYCLES covers the whole batch (first beat in -> TLAST)
        ovf += acc.read("OVF_COUNT")                    # each run() start clears it: add up per batch
        print(f"  {s + len(xs)}/{n} images", flush=True)
    t_all = time.perf_counter() - t_all

    hw_pred = hw_scores.argmax(1)
    mism = int((hw_scores != ref).any(axis=1).sum())    # images where any score differs from the ARM
    for i in range(min(a.demo, n)):                     # demo from the batch pass: scores are already there
        print(ascii_digit(imgs[i]))
        print(f"  image {i}: FPGA says {hw_pred[i]}, label {labels[i]}, "
              f"{'bit-exact' if np.array_equal(hw_scores[i], ref[i]) else 'MISMATCH'}\n", flush=True)

    hw_acc = 100.0 * float((hw_pred == labels).mean())
    arm_acc = 100.0 * float((arm_pred == labels).mean())
    cyc = np.array(cyc_single)
    ok &= check(f"FPGA bit-exact vs ARM on {n} MNIST images (batches of {a.batch})", mism == 0,
                f"{mism} mismatching images")
    ok &= check(f"RESULT register = argmax of the scores ({k_lat} single images)", result_bad == 0,
                f"{result_bad} differ")
    print(f"Accuracy  FPGA {hw_acc:.2f} %   ARM NumPy {arm_acc:.2f} %   ({n} images)")
    print(f"CYCLES    single image: median {int(np.median(cyc))}  min {cyc.min()}  max {cyc.max()}"
          f" = {np.median(cyc) / fclk:.2f} us;  batched: {batch_cycles / n:.1f} per image")
    print(f"Per image FPGA one at a time {1e3 * np.median(walls):.3f} ms, in batches of {a.batch} "
          f"{1e3 * t_all / n:.4f} ms   ARM NumPy single {1e3 * arm_single:.3f} ms, batched {1e3 * arm_batched:.3f} ms")

    res = {
        "kind": "mlp_board_eval", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pynq_version": getattr(pynq, "__version__", "?"), "cpu": platform.machine(),
        "images": n, "fclk_mhz": fclk, "stream_w": sw, "bits": bits,
        "shift_policy": net.get("shift_policy", "?"), "cfg_hash": m["cfg_hash"],
        "vectors": len(vin), "vectors_bit_exact": not bad,
        "bit_exact_images": int(n - mism), "result_reg_ok": bool(result_bad == 0),
        "hw_accuracy": hw_acc, "arm_accuracy": arm_acc,
        "cycles_median": int(np.median(cyc)), "cycles_min": int(cyc.min()), "cycles_max": int(cyc.max()),
        "hw_core_ms": float(np.median(cyc)) / fclk / 1e3,
        "hw_wall_ms": 1e3 * float(np.median(walls)),            # latency: one image end-to-end
        "latency_images": k_lat,
        "batch": a.batch, "pixels_per_beat": ppb,
        "cycles_per_image_batched": batch_cycles / n,           # images back to back in hardware
        "hw_batched_ms": 1e3 * t_all / n,                       # throughput: wall time per image in batches
        "hw_total_s": t_all,                                    # all n images in batches
        "arm_single_ms": 1e3 * arm_single, "arm_batched_ms": 1e3 * arm_batched,
        "ovf_total": int(ovf), "passed": bool(ok),
    }
    (HERE / a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {a.out}")
    acc.close()
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()