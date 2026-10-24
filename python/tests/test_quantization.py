"""Contract, arithmetic, and exported-vector regression tests."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.integer_reference import infer, load_network, parameter_digest, quantize_input, requantize, write_test_vectors
from scripts.train_model import DATA_DIR, QUANTIZED_DIR


class QuantizationTest(unittest.TestCase):
    def network(self, bits=4, layers=None):
        return {"schema_version": 1, "input_bits": bits, "activation_bits": bits,
                "weight_bits": bits, "bias_bits": 32, "weight_signed": True,
                "activation_signed": False, "parameter_rounding": "nearest_ties_positive",
                "shift_rounding": "truncate", "activation_overflow": "saturate", "accumulator_overflow": "reject",
                "layers": layers or [
                    {"index": 0, "in_features": 2, "out_features": 2, "accumulator_bits": 2 * bits + 1,
                     "multiplier": 1, "shift": 1, "activation": "relu", "weights": [[2, -1], [-1, 2]], "bias": [1, -3]},
                    {"index": 2, "in_features": 2, "out_features": 2, "accumulator_bits": 2 * bits + 1,
                     "multiplier": 1, "shift": 7, "activation": "identity", "weights": [[3, -2], [-1, 4]], "bias": [-2, 1]}]}

    def write_network(self, directory, network):
        metadata = copy.deepcopy(network)
        for layer in metadata["layers"]:
            np.savetxt(directory / f"{layer['index']}_weight.csv", layer.pop("weights"), delimiter=",", fmt="%d")
            np.savetxt(directory / f"{layer['index']}_bias.csv", layer.pop("bias"), fmt="%d")
        (directory / "network.json").write_text(json.dumps(metadata))

    def test_metadata_drives_input_activation_and_signed_output(self):
        for bits in (4, 8, 12, 16):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.write_network(directory, self.network(bits))
                network = load_network(directory)
                maximum = (1 << bits) - 1
                self.assertEqual(quantize_input([-0.1, 0, 0.5, 1, 1.1], network["input_bits"]).tolist(),
                                 [0, 0, (maximum + 1) // 2, maximum, maximum])
                self.assertEqual(requantize([-3, 0, maximum, maximum + 1, maximum + 3], 0, bits).tolist(),
                                 [0, 0, maximum, maximum, maximum])  # saturates, does not wrap
                self.assertEqual(infer([[2, 3], [0, 1]], network).tolist(), [[1, -1], [-2, 1]])
                with self.assertRaisesRegex(ValueError, "configured width"):
                    infer([[maximum + 1, 0]], network)

    def test_relu_before_truncating_shift(self):
        self.assertEqual(requantize([-3, -1, 0, 1, 3], 1, 4).tolist(), [0, 0, 0, 0, 1])

    def test_layer_count_comes_from_metadata(self):
        network = self.network()
        network["layers"][1].update(activation="relu", shift=0)
        network["layers"].append({"index": 4, "in_features": 2, "out_features": 2, "accumulator_bits": 9,
                                  "multiplier": 1, "shift": 3, "activation": "identity",
                                  "weights": [[2, -1], [-1, 2]], "bias": [-1, 0]})
        with tempfile.TemporaryDirectory() as temporary:
            self.write_network(Path(temporary), network)
            loaded = load_network(temporary)
            self.assertEqual(infer([[2, 3], [0, 1]], loaded).tolist(), [[1, -1], [-2, 2]])

    def test_final_scores_keep_sign_range_and_winning_class(self):
        network = self.network(8, [{"in_features": 1, "weights": np.array([[100, 120, -100]]),
                                   "bias": np.array([100, 100, -100]), "accumulator_bits": 16,
                                   "shift": 11, "activation": "identity"}])
        scores = infer([[1]], network)
        self.assertEqual(scores.tolist(), [[200, 220, -200]])
        self.assertEqual(scores.argmax(1).tolist(), [1])

    def test_int32_boundaries_and_overflow(self):
        for bias in (-(1 << 31), (1 << 31) - 1):
            network = self.network(8, [{"in_features": 1, "weights": np.array([[0]]),
                                       "bias": np.array([bias]), "accumulator_bits": 32,
                                       "shift": 0, "activation": "identity"}])
            self.assertEqual(infer([[0]], network).tolist(), [[bias]])
        for value in (-(1 << 31) - 1, 1 << 31):
            network["layers"][0]["bias"] = np.array([value])
            with self.assertRaisesRegex(ValueError, "MAC accumulator overflow"):
                infer([[0]], network)

    def test_invalid_contracts_are_rejected(self):
        cases = [("schema_version", 2), ("weight_bits", 0), ("bias_bits", 64),
                 ("activation_signed", True), ("shift_rounding", "nearest"), ("activation_overflow", "wrap")]
        for key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                network = self.network()
                network[key] = value
                self.write_network(Path(temporary), network)
                with self.assertRaises(ValueError):
                    load_network(temporary)
        for kind, value in (("weight", 8), ("bias", 1 << 31)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.write_network(directory, self.network())
                filename = directory / f"0_{kind}.csv"
                values = np.loadtxt(filename, delimiter=",", ndmin=2 if kind == "weight" else 1)
                values.flat[0] = value
                np.savetxt(filename, values, delimiter=",", fmt="%d")
                with self.assertRaises(ValueError):
                    load_network(directory)

    def test_noninteger_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer inputs"):
            infer([[1.5, 0]], self.network())
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_input([np.nan], 4)

    def test_exported_vectors_round_trip_and_identify_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_network(directory, self.network())
            network = load_network(directory)
            inputs = np.array([[2, 3], [0, 1]])
            write_test_vectors(directory, network, inputs)
            metadata = json.loads((directory / "test_vectors.json").read_text())
            self.assertEqual(metadata["parameter_sha256"], parameter_digest(directory, network))
            np.testing.assert_array_equal(np.loadtxt(directory / "expected_outputs.csv", delimiter=","), [[1, -1], [-2, 1]])
            np.testing.assert_array_equal(np.loadtxt(directory / "test_inputs.csv", delimiter=","), inputs)
            previous = metadata["parameter_sha256"]
            (directory / "0_bias.csv").write_text("2\n-3\n")
            self.assertNotEqual(previous, parameter_digest(directory, network))

    def test_scala_export_matches_pytorch_on_mnist(self):
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        directory = Path(os.environ.get("QUANTIZED_DIR", QUANTIZED_DIR))
        network = load_network(directory)
        data = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
        for images, _ in DataLoader(data, batch_size=256):
            inputs = quantize_input(images.flatten(1).numpy(), network["input_bits"])
            q = torch.from_numpy(inputs)
            for layer in network["layers"]:
                acc = q @ torch.from_numpy(layer["weights"]) + torch.from_numpy(layer["bias"])
                q = (acc.clamp_min(0) >> layer["shift"]).clamp_max((1 << network["activation_bits"]) - 1) if layer["activation"] == "relu" else acc
            np.testing.assert_array_equal(infer(inputs, network), q.numpy())
        metadata = json.loads((directory / "test_vectors.json").read_text())
        self.assertEqual(metadata["parameter_sha256"], parameter_digest(directory, network))
        inputs = np.loadtxt(directory / "test_inputs.csv", delimiter=",", dtype=np.int64, ndmin=2)
        expected = np.loadtxt(directory / "expected_outputs.csv", delimiter=",", dtype=np.int64, ndmin=2)
        self.assertEqual(len(inputs), metadata["samples"])
        np.testing.assert_array_equal(infer(inputs, network), expected)


if __name__ == "__main__":
    unittest.main()
