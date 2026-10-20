"""Compare float accuracy with Scala's integer export and write reference vectors."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .integer_reference import load_network, quantize_input, infer, write_test_vectors
from .train_model import ACTIVATION, CSV_DIR, DATA_DIR, OUTPUT_WEIGHTS_FILE, QUANTIZED_DIR, RESULTS_DIR, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized-dir", type=Path, default=QUANTIZED_DIR)
    parser.add_argument("--float-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_WEIGHTS_FILE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--test-vectors", type=int, default=64)
    args = parser.parse_args()
    if args.test_vectors < 1:
        parser.error("--test-vectors must be positive")
    network = load_network(args.quantized_dir)
    if ACTIVATION is not torch.nn.ReLU:
        raise ValueError("The exported numerical contract supports ReLU hidden layers only")
    sizes = [network["layers"][0]["in_features"]] + [layer["out_features"] for layer in network["layers"]]
    if sizes[0] != 784:
        raise ValueError("MNIST evaluation requires 784 inputs; integer_reference supports other shapes")
    model = build_model(sizes, ACTIVATION)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    for position, layer in enumerate(network["layers"]):
        linear = model[2 * position]
        for kind, values in (("weight", linear.weight), ("bias", linear.bias)):
            csv = np.loadtxt(args.float_dir / f"{layer['index']}_{kind}.csv", delimiter=",",
                             ndmin=2 if kind == "weight" else 1)
            if not np.array_equal(csv, values.detach().numpy().T):
                raise ValueError("CSV and float checkpoint differ")

    data = datasets.MNIST(args.data_dir, train=False, download=False, transform=transforms.ToTensor())
    float_correct = quantized_correct = 0
    vector_inputs = []
    vector_count = 0
    with torch.no_grad():
        for images, labels in DataLoader(data, batch_size=256, shuffle=False):
            x = images.flatten(1)
            float_correct += (model(x).argmax(1) == labels).sum().item()
            q = quantize_input(x.numpy(), network["input_bits"])
            scores = infer(q, network)
            quantized_correct += int((scores.argmax(1) == labels.numpy()).sum())
            take = min(len(q), args.test_vectors - vector_count)
            if take > 0:
                vector_inputs.append(q[:take].copy())
                vector_count += take

    float_accuracy = 100 * float_correct / len(data)
    quantized_accuracy = 100 * quantized_correct / len(data)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.results_dir / "quantization_results.csv",
               [[float_accuracy, quantized_accuracy, len(data), network["weight_bits"], network["activation_bits"], network["bias_bits"]]],
               delimiter=",", header="float_accuracy_percent,quantized_accuracy_percent,test_samples,weight_bits,activation_bits,bias_bits",
               comments="", fmt=["%.2f", "%.2f", "%d", "%d", "%d", "%d"])
    write_test_vectors(args.quantized_dir, network, np.concatenate(vector_inputs))
    print(f"Widths: weights={network['weight_bits']}, inputs/activations={network['activation_bits']}, bias={network['bias_bits']}")
    print(f"Float accuracy: {float_accuracy:.2f}% | Quantized accuracy: {quantized_accuracy:.2f}%")
    print(f"Difference: {quantized_accuracy - float_accuracy:+.2f} percentage points ({len(data)} images)")
    print(f"Exported {vector_count} deterministic test vectors to {args.quantized_dir}")


if __name__ == "__main__":
    main()
