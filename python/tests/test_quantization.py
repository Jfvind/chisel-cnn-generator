# in: fixed test cases, exported integer CSVs and MNIST test images
# out: unittest pass/fail results
# Checks input conversion, activation behavior, final scores and overflow.

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.evaluate_quantized import quantize_input
from scripts.integer_reference import infer, requantize
from scripts.train_model import DATA_DIR, QUANTIZED_DIR


class QuantizationTest(unittest.TestCase):
    def test_numpy_matches_pytorch_on_mnist(self):
        layers = []
        parameters = np.loadtxt(QUANTIZED_DIR / "layers.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
        for index, multiplier, shift in parameters:
            self.assertEqual(multiplier, 1)
            weights = np.loadtxt(QUANTIZED_DIR / f"{index}_weight.csv", delimiter=",", dtype=np.int64, ndmin=2)
            bias = np.loadtxt(QUANTIZED_DIR / f"{index}_bias.csv", dtype=np.int64, ndmin=1)
            layers.append((weights, bias, int(shift)))
        torch_layers = [(torch.from_numpy(w), torch.from_numpy(b), s) for w, b, s in layers]
        data = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
        for images, _ in DataLoader(data, batch_size=256):
            inputs = quantize_input(images.flatten(1))
            q = inputs
            for position, (weights, bias, shift) in enumerate(torch_layers):
                acc = q @ weights + bias
                q = ((acc.clamp_min(0) >> shift) & 255) if position < len(layers) - 1 else acc
            np.testing.assert_array_equal(infer(inputs.numpy(), layers), q.numpy())

    def test_unsigned_input(self):
        pixels = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
        self.assertEqual(quantize_input(pixels).tolist(), [0, 0, 128, 255, 255])

    def test_relu_before_truncating_shift(self):
        self.assertEqual(requantize([-3, -1, 0, 1, 3], 1).tolist(), [0, 0, 0, 0, 1])

    def test_lowest_eight_bits_match_activation(self):
        acc = [-1000, 0, 127, 255, 256, 300, 511, 512]
        self.assertEqual(requantize(acc, 0).tolist(), [0, 0, 127, 255, 0, 44, 255, 0])

    def test_int32_boundaries_and_overflow(self):
        inputs = np.zeros((1, 65536), dtype=np.uint8)  # 16 + log2(65536) = 32 MAC bits.
        weights = np.zeros((65536, 2), dtype=np.int64)
        bias = np.array([-(1 << 31), (1 << 31) - 1])
        self.assertEqual(infer(inputs, [(weights, bias, 0)]).tolist(), [bias.tolist()])
        for value in [-(1 << 31) - 1, 1 << 31]:
            with self.assertRaisesRegex(AssertionError, "MAC accumulator overflow"):
                infer(inputs, [(weights, np.array([value, 0]), 0)])

    def test_two_layers_with_signed_output(self):
        layers = [(np.array([[2, -1], [-1, 2]]), np.array([1, -3]), 1),
                  (np.array([[3, -2], [-1, 4]]), np.array([-2, 1]), 7)]
        self.assertEqual(infer(np.array([[2, 3], [0, 1]], dtype=np.uint8), layers).tolist(), [[1, -1], [-2, 1]])

    def test_final_scores_keep_sign_range_and_winning_class(self):
        layers = [(np.array([[100, 120, -100]]), np.array([100, 100, -100]), 11)]
        scores = infer([[1]], layers)
        self.assertEqual(scores.tolist(), [[200, 220, -200]])
        self.assertEqual(scores.argmax(1).tolist(), [1])

    def test_mac_overflow_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "MAC accumulator overflow"):
            infer([[255]], [(np.array([[127]]), np.array([32767]), 0)])

    def test_float_inputs_are_rejected(self):
        layer = (np.array([[1]]), np.array([0]), 0)
        with self.assertRaisesRegex(AssertionError, "integer inputs"):
            infer([[1.5]], [layer])


if __name__ == "__main__":
    unittest.main()
