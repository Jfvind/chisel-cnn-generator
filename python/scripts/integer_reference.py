# in: inputs: integer ndarray[batch, features], layers: list[(weights, bias, shift)]
# out: scores: integer ndarray[batch, classes]
# Runs integer MLP inference with uint8 hidden activations and wide signed scores.

import numpy as np


def requantize(accumulator, shift):
    """ReLU, right shift, then the lowest eight bits; inputs are already checked."""
    shifted = np.maximum(accumulator, 0) >> shift
    return shifted & 255


def infer(inputs, layers):
    """Run integer inputs through layers validated when loading the CSV files."""
    q = np.asarray(inputs)
    assert q.dtype.kind in "iu" and q.ndim == 2, "Expected a matrix of integer inputs"
    assert q.min() >= 0 and q.max() <= 255, "Inputs must fit uint8"
    q = q.astype(np.int64)
    for position, (weights, bias, shift) in enumerate(layers):
        acc = q @ weights + bias
        bits = min(32, 16 + (weights.shape[0] - 1).bit_length())  # MAC width, within int32.
        assert acc.min() >= -(1 << (bits - 1)) and acc.max() < (1 << (bits - 1)), "MAC accumulator overflow"
        q = requantize(acc, shift) if position < len(layers) - 1 else acc
    return q
