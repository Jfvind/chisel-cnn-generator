# Quantized network export, version 1

Run `WeightExporter` from `generator/`, or `make reference` from the repository
root. By default it reads `python/models/csv/` and writes
`generator/generated/quantized/`.

## Scala-produced files

- `<index>_weight.csv`: signed decimal integers, one input per row and one output
  per column. Dimensions are `in_features` by `out_features`.
- `<index>_bias.csv`: signed decimal integers, one output bias per line.
- `network.json`: the authoritative numerical contract. Replaces `layers.csv`.

The JSON contains `schema_version: 1`; `input_bits`, `activation_bits`,
`weight_bits`, `bias_bits`; signedness; parameter rounding, shift rounding,
activation-overflow and accumulator-overflow policies. `layers` is an ordered
nonempty list of `{index,in_features,out_features,accumulator_bits,multiplier,
shift,activation}`. Indices identify the parameter filenames and must be strictly
increasing. Layer sizes must chain. Multiplier is currently 1; hidden activations
are `relu` and the final activation is `identity`. See [the rules](quantization.md).

The Python loader validates supported policies, integer CSV values, parameter
shapes/ranges, layer chaining and conservative MAC bounds. It reads values from
Scala; it does not quantize weights or biases. NumPy int64 is a calculation
container, not an assertion that the hardware uses 64-bit weights.

## Python-produced reference files

`evaluate_quantized` compares the float checkpoint with the integer reference on
MNIST and writes:

- `python/results/quantization_results.csv`: float/integer accuracy, number of
  test images, and the selected weight, activation and bias widths.
- `test_inputs.csv` in the export directory: the first 64 MNIST test images in
  dataset order, flattened and quantized according to `input_bits`.
- `expected_outputs.csv`: corresponding wide signed integer scores, one row per
  input and one column per output class. Final-layer shifts are not applied.
- `test_vectors.json`: format version, dimensions, sample count and
  `parameter_sha256`. The digest covers the exact `network.json` bytes followed
  by each layer's weight and bias CSV bytes, in layer order (weight then bias).

Use `--test-vectors N` to select a different fixed prefix. There is no shuffling
or random training in this command. Regenerating Scala parameters invalidates
and removes prior reference vectors; run evaluation again before comparing.
Readers must check the parameter digest so vectors from another width/model
cannot be treated as the current reference.

A future ChiselTest can feed `test_inputs.csv` into `MlpCore` and compare every
score to `expected_outputs.csv`. The board test can use the same files. The
shell smoke test in #38 exercises the addition stub, so it cannot perform that
comparison yet. Signed score representation on AXI must be agreed when adding
the core: an 8-bit stream beat cannot hold a wide score in one beat, and the
generic PYNQ driver in #38 returns unsigned raw words.
