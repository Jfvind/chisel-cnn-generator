"""Time the MLP on this computer's CPU, for comparison with the FPGA (dashboard "Python vs hardware").

    python -m scripts.benchmark_cpu --dataset fashion --hidden 64,32   (from python/, or `make cpu-bench DATASET=fashion HIDDEN=64,32`)

Measures, on the 10,000 test images of --dataset (mnist, fashion or kmnist):
  float  : the trained PyTorch model (float32)
  integer: the bit-exact integer reference (NumPy, int64) on Scala's export, the same
           numbers the FPGA computes
each as accuracy, time per image one at a time (latency) and time per image in one batch
(throughput). Writes results/cpu_benchmark.json (results/<dataset>/ for Fashion and KMNIST).
Without PyTorch only the integer part runs.
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
# --dataset -> the folder torchvision downloads it to, as in train_model.DATASETS
RAW_FOLDERS = {"mnist": "MNIST", "fashion": "FashionMNIST", "kmnist": "KMNIST"}


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
    ap.add_argument("--dataset", choices=RAW_FOLDERS, default="mnist")
    ap.add_argument("--hidden", default="128", help="hidden layer widths of the model, e.g. 64,32 or none (default 128)")
    ap.add_argument("--quantized-dir", type=pathlib.Path, help="default: the committed export of --dataset")
    ap.add_argument("--single", type=int, default=1000, help="images timed one at a time")
    ap.add_argument("--out", type=pathlib.Path, help="default: results/cpu_benchmark.json, results/<dataset>/... for the others")
    a = ap.parse_args()
    # MNIST keeps the committed paths; other hidden widths get hidden-<widths>/, as train_model.model_dir.
    # The folder is named from the text as the Makefile passes it (64,32 -> 64-32, none -> none)
    sub = pathlib.Path("" if a.dataset == "mnist" else a.dataset)
    if a.hidden != "128":
        sub = sub / ("hidden-" + a.hidden.replace(",", "-"))
    quantized_dir = a.quantized_dir or PYTHON_DIR.parent / "generator/generated" / sub / "quantized"
    out = a.out or RESULTS_DIR / sub / "cpu_benchmark.json"

    raw = DATA_DIR / RAW_FOLDERS[a.dataset] / "raw"
    imgs = read_idx(raw / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    labels = read_idx(raw / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    x = imgs.astype(np.float32) / 255.0                       # what torchvision's ToTensor gives
    n, k = len(x), min(a.single, len(x))

    res = {"kind": "cpu_benchmark", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "host": platform.node(), "cpu": platform.processor() or platform.machine(),
           "cores": os.cpu_count(), "python": platform.python_version(), "images": n, "dataset": a.dataset}

    # Integer reference (NumPy): the same arithmetic as the FPGA
    net = load_network(quantized_dir)
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
        from .train_model import ACTIVATION, build_model, model_dir
        sizes = [net["layers"][0]["in_features"]] + [l["out_features"] for l in net["layers"]]
        model = build_model(sizes, ACTIVATION)
        model.load_state_dict(torch.load(model_dir(a.dataset, [] if a.hidden == "none" else [int(w) for w in a.hidden.split(",")]) / "mlp_weights.pt", weights_only=True, map_location="cpu"))
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

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    for key in ("float", "int"):
        if f"{key}_accuracy" in res:
            print(f"{key:>5}: {res[f'{key}_accuracy']:.2f} %   single {res[f'{key}_single_ms']:.4f} ms/image   "
                  f"batched {res[f'{key}_batched_ms']:.5f} ms/image")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()