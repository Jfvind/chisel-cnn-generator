"""Integer inference using Scala's exported parameters and numerical contract."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_network(directory):
    directory = Path(directory)
    network = json.loads((directory / "network.json").read_text())
    _require(isinstance(network, dict), "Expected a network metadata object")
    rules = {
        "schema_version": 1, "weight_signed": True, "activation_signed": False,
        "parameter_rounding": "nearest_ties_positive", "shift_rounding": "truncate",
        "activation_overflow": "wrap", "accumulator_overflow": "reject",
    }
    for key, expected in rules.items():
        _require(type(network.get(key)) is type(expected) and network[key] == expected,
                 f"Unsupported or missing {key}: {network.get(key)!r}")
    for key, lower, upper in (("input_bits", 1, 16), ("activation_bits", 1, 16),
                              ("weight_bits", 2, 16), ("bias_bits", 2, 32)):
        value = network.get(key)
        _require(type(value) is int and lower <= value <= upper, f"Invalid {key}")
    layers = network.get("layers")
    _require(isinstance(layers, list) and len(layers) > 0, "Expected a nonempty layer list")
    previous_outputs = None
    previous_index = -1
    for position, layer in enumerate(layers):
        _require(isinstance(layer, dict), "Expected a layer metadata object")
        for key in ("index", "in_features", "out_features", "accumulator_bits", "multiplier", "shift"):
            _require(type(layer.get(key)) is int, f"Invalid layer {key}")
        index = layer["index"]
        _require(index > previous_index, "Layer indices must be nonnegative and strictly increasing")
        _require(layer["in_features"] > 0 and layer["out_features"] > 0, "Invalid layer shape")
        _require(previous_outputs is None or previous_outputs == layer["in_features"], "Layer sizes do not chain")
        _require(1 <= layer["accumulator_bits"] <= 63 and 0 <= layer["shift"] < 63, "Invalid accumulator width or shift")
        _require(layer["multiplier"] == 1, "Only shift-only rescaling is supported")
        expected_activation = "identity" if position == len(layers) - 1 else "relu"
        _require(layer.get("activation") == expected_activation, "Unsupported layer activation")
        weights = np.loadtxt(directory / f"{index}_weight.csv", delimiter=",", dtype=np.int64,
                             ndmin=2, converters=int)
        bias = np.loadtxt(directory / f"{index}_bias.csv", dtype=np.int64, ndmin=1, converters=int)
        _require(weights.shape == (layer["in_features"], layer["out_features"])
                 and bias.shape == (layer["out_features"],), "Invalid parameter shapes")
        weight_limit = 1 << (network["weight_bits"] - 1)
        bias_limit = 1 << (network["bias_bits"] - 1)
        _require(weights.min() >= -weight_limit and weights.max() < weight_limit, "Weights exceed configured width")
        _require(bias.min() >= -bias_limit and bias.max() < bias_limit, "Bias exceeds configured width")
        input_bits = network["input_bits"] if position == 0 else network["activation_bits"]
        for j in range(layer["out_features"]):
            bound = ((1 << input_bits) - 1) * sum(abs(int(w)) for w in weights[:, j]) + abs(int(bias[j]))
            _require(bound < (1 << (layer["accumulator_bits"] - 1)), "MAC accumulator can overflow")
        layer.update(weights=weights, bias=bias)
        previous_index, previous_outputs = index, layer["out_features"]
    return network


def quantize_input(values, bits):
    values = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(values).all(), "Input pixels must be finite")
    maximum = (1 << bits) - 1
    return np.floor(values * maximum + 0.5).clip(0, maximum).astype(np.int64)


def requantize(accumulator, shift, bits):
    return (np.maximum(accumulator, 0) >> shift) & ((1 << bits) - 1)


def infer(inputs, network):
    q = np.asarray(inputs)
    _require(q.dtype.kind in "iu" and q.ndim == 2 and q.size > 0, "Expected a nonempty matrix of integer inputs")
    _require(q.shape[1] == network["layers"][0]["in_features"], "Invalid input feature count")
    _require(q.min() >= 0 and q.max() < (1 << network["input_bits"]), "Inputs exceed configured width")
    q = q.astype(np.int64)
    for layer in network["layers"]:
        acc = q @ layer["weights"] + layer["bias"]
        limit = 1 << (layer["accumulator_bits"] - 1)
        _require(acc.min() >= -limit and acc.max() < limit, "MAC accumulator overflow")
        q = requantize(acc, layer["shift"], network["activation_bits"]) if layer["activation"] == "relu" else acc
    return q


def parameter_digest(directory, network):
    directory = Path(directory)
    digest = hashlib.sha256((directory / "network.json").read_bytes())
    for layer in network["layers"]:
        for kind in ("weight", "bias"):
            digest.update((directory / f"{layer['index']}_{kind}.csv").read_bytes())
    return digest.hexdigest()


def write_test_vectors(directory, network, inputs):
    directory = Path(directory)
    scores = infer(inputs, network)
    np.savetxt(directory / "test_inputs.csv", inputs, delimiter=",", fmt="%d")
    np.savetxt(directory / "expected_outputs.csv", scores, delimiter=",", fmt="%d")
    metadata = {"schema_version": 1, "parameter_sha256": parameter_digest(directory, network),
                "samples": len(inputs), "input_features": inputs.shape[1], "output_features": scores.shape[1]}
    (directory / "test_vectors.json").write_text(json.dumps(metadata, indent=2) + "\n")
