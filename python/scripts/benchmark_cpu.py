"""Time the MLP on this computer's CPU, for comparison with the FPGA (dashboard "Python vs hardware").

    python -m scripts.benchmark_cpu            (from python/, or `make cpu-bench` from the repo root)

Measures, on the 10,000 MNIST test images:
  float  : the trained PyTorch model (float32)
  integer: the bit-exact integer reference (NumPy, int64) on Scala's export, the same
           numbers the FPGA computes
each as accuracy, time per image one at a time (latency) and time per image in one batch
(throughput). Writes results/cpu_benchmark.json. Without PyTorch only the integer part runs.
"""

import argparse
import json
import os
import pathlib
import platform
import time

import numpy as np

from .integer_reference import infer, load_network, quantize_input

# Same locations as train_model.py (not imported here, so the integer part runs without PyTorch)
PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PYTHON_DIR / "data"
RESULTS_DIR = PYTHON_DIR / "results"
QUANTIZED_DIR = PYTHON_DIR.parent / "generator/generated/quantized"


def read_idx(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    dims = [int.from_bytes(data[4 + 4 * k: 8 + 4 * k], "big") for k in range(data[3])]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * len(dims)).reshape(dims)


def per_image(fn, items, repeat=1):
    """Median seconds of fn(item) over items (one at a time)."""
    times = []
    for _ in range(repeat):
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantized-dir", type=pathlib.Path, default=QUANTIZED_DIR)
    ap.add_argument("--single", type=int, default=1000, help="images timed one at a time")
    ap.add_argument("--out", type=pathlib.Path, default=RESULTS_DIR / "cpu_benchmark.json")
    a = ap.parse_args()

    raw = DATA_DIR / "MNIST" / "raw"
    imgs = read_idx(raw / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    labels = read_idx(raw / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    x = imgs.astype(np.float32) / 255.0                       # what torchvision's ToTensor gives
    n, k = len(x), min(a.single, len(x))

    res = {"kind": "cpu_benchmark", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "host": platform.node(), "cpu": platform.processor() or platform.machine(),
           "cores": os.cpu_count(), "python": platform.python_version(), "images": n}

    # Integer reference (NumPy): the same arithmetic as the FPGA
    net = load_network(a.quantized_dir)
    q = quantize_input(x, net["input_bits"])
    t0 = time.perf_counter()
    scores = infer(q, net)
    res["int_batched_ms"] = 1e3 * (time.perf_counter() - t0) / n
    res["int_single_ms"] = 1e3 * per_image(lambda i: infer(q[i:i + 1], net), range(k))
    res["int_accuracy"] = 100.0 * float((scores.argmax(1) == labels).mean())
    res["shift_policy"] = net.get("shift_policy", "?")
    res["bits"] = net["input_bits"]

    # Float model (PyTorch)
    try:
        import torch
        from .train_model import ACTIVATION, OUTPUT_WEIGHTS_FILE, build_model
        sizes = [net["layers"][0]["in_features"]] + [l["out_features"] for l in net["layers"]]
        model = build_model(sizes, ACTIVATION)
        model.load_state_dict(torch.load(OUTPUT_WEIGHTS_FILE, weights_only=True, map_location="cpu"))
        model.eval()
        xt = torch.from_numpy(x)
        with torch.no_grad():
            model(xt[:256])                                   # warm-up
            t0 = time.perf_counter()
            logits = model(xt)
            res["float_batched_ms"] = 1e3 * (time.perf_counter() - t0) / n
            res["float_single_ms"] = 1e3 * per_image(lambda i: model(xt[i:i + 1]), range(k))
        res["float_accuracy"] = 100.0 * float((logits.argmax(1).numpy() == labels).mean())
        res["torch"] = torch.__version__
        res["torch_threads"] = torch.get_num_threads()
    except ImportError:
        print("PyTorch not installed: float model skipped")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    for key in ("float", "int"):
        if f"{key}_accuracy" in res:
            print(f"{key:>5}: {res[f'{key}_accuracy']:.2f} %   single {res[f'{key}_single_ms']:.4f} ms/image   "
                  f"batched {res[f'{key}_batched_ms']:.5f} ms/image")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()