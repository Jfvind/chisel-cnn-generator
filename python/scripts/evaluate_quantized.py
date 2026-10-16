# in: float checkpoint (.pt), float/integer CSVs, MNIST test images and labels
# out: accuracy report (.csv), exact NumPy/PyTorch integer output comparison
# Evaluates the exported integers alongside the float model.

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .integer_reference import infer
from .train_model import ACTIVATION, CSV_DIR, DATA_DIR, LAYER_SIZES, OUTPUT_WEIGHTS_FILE, QUANTIZED_DIR, RESULTS_DIR, build_model


def quantize_input(values):
    return torch.floor(values.to(torch.float64) * 255 + 0.5).clamp(0, 255).to(torch.int64)


def main():
    model = build_model(LAYER_SIZES, ACTIVATION)
    model.load_state_dict(torch.load(OUTPUT_WEIGHTS_FILE, weights_only=True, map_location="cpu"))
    model.eval()
    assert len(model) == 3 and isinstance(model[1], torch.nn.ReLU), "Expected Linear -> ReLU -> Linear"
    parameters = np.loadtxt(QUANTIZED_DIR / "layers.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    assert parameters[:, 0].tolist() == [0, 2], "Expected layers 0 and 2"
    layers = []
    for index, multiplier, shift in parameters:
        assert multiplier == 1 and 0 <= shift < 32, "Expected shift-only rescaling"
        linear = model[int(index)]
        assert isinstance(linear, torch.nn.Linear) and linear.bias is not None, "Expected Linear with bias"
        for kind, values in (("weight", linear.weight), ("bias", linear.bias)):
            csv = np.loadtxt(CSV_DIR / f"{index}_{kind}.csv", delimiter=",")
            assert np.array_equal(csv, values.detach().numpy().T), "CSV and float checkpoint differ"
        weights = np.loadtxt(QUANTIZED_DIR / f"{index}_weight.csv", delimiter=",", dtype=np.int64, ndmin=2)
        bias = np.loadtxt(QUANTIZED_DIR / f"{index}_bias.csv", dtype=np.int64, ndmin=1)
        assert weights.shape == linear.weight.T.shape and bias.shape == linear.bias.shape, "Invalid layer shapes"
        assert weights.min() >= -128 and weights.max() <= 127, "Weights must fit int8"
        assert bias.min() >= -(1 << 31) and bias.max() < (1 << 31), "Bias must fit int32"
        layers.append((weights, bias, int(shift)))
    torch_layers = [(torch.from_numpy(w), torch.from_numpy(b), s) for w, b, s in layers]

    data = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    float_correct = quantized_correct = 0
    with torch.no_grad():
        for images, labels in DataLoader(data, batch_size=256):
            x = images.flatten(1)
            float_correct += (model(x).argmax(1) == labels).sum().item()
            q = quantize_input(x)
            expected = infer(q.numpy(), layers)
            for position, (weights, bias, shift) in enumerate(torch_layers):
                acc = q @ weights + bias
                q = ((acc.clamp_min(0) >> shift) & 255) if position < len(layers) - 1 else acc
            np.testing.assert_array_equal(expected, q.numpy())
            quantized_correct += int((expected.argmax(1) == labels.numpy()).sum())

    float_accuracy = 100 * float_correct / len(data)
    quantized_accuracy = 100 * quantized_correct / len(data)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS_DIR / "quantization_results.csv", [[float_accuracy, quantized_accuracy, len(data)]],
               delimiter=",", header="float_accuracy_percent,quantized_accuracy_percent,test_samples", comments="", fmt=["%.2f", "%.2f", "%d"])
    print(f"Float accuracy: {float_accuracy:.2f}% | Quantized accuracy (wide logits): {quantized_accuracy:.2f}%")
    print(f"Difference: {quantized_accuracy - float_accuracy:+.2f} percentage points ({len(data)} images)")
    print(f"NumPy and PyTorch integer outputs match exactly on all {len(data)} images")


if __name__ == "__main__":
    main()
