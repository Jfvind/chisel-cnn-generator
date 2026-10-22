# Numerical contract

Scala owns parameter quantization. Python reads its exported integers and contract;
it never requantizes float weights or biases. The original float checkpoint remains
the accuracy baseline. The integer reference is the future hardware's bit-exact target.

## Configuration

`model.QuantizationConfig(weightBits, activationBits, biasBits)` is passed to
`WeightLoader.load`. The exporter accepts `--weight-bits`, `--activation-bits`,
`--bias-bits`, `--input` and `--output`. Defaults are 8, 8 and 32 bits.

- Signed weights: 2..16 bits; unsigned inputs/hidden activations: 1..16 bits.
- Input and hidden activation widths are tied in the current Scala scheme.
- Signed bias: 2..32 bits. Biases outside the selected range fail before conversion.
- Each layer's signed MAC width is `weightBits + activationBits + ceil(log2(inputs))`.
  It must be at most 63 bits for the int64 reference. A wider bias setting does not
  widen the MAC; the exporter rejects parameters whose conservative sum bound
  does not fit the MAC width.

The present hardware primitives use a shared weight/activation width. Different
weight and activation widths are supported by the parameter export/reference;
the future MLP core must explicitly support their combination before synthesis.
No `MlpCore` is implemented here. The AXI shell and board workflow are tracked
separately in #38.

## Rules

For weight scale `W = 2^(weightBits-1)` and unsigned activation maximum
`A = 2^activationBits-1`:

1. Input pixels use `floor(x * A + 0.5)`, clipped to `[0,A]`.
2. Scala rejects nonfinite weights or weights outside `[-1,1]`, then computes
   `min(round(w * W), W-1)`. Parameter ties round toward positive infinity.
3. Bias is `round(b * activationScale * W)`, where the first activation scale
   is `A`. CSV matrices are `[input][output]`.
4. Scala picks the smallest nonnegative shift that makes the worst-case positive
   sum fit `[0,A]`. After the hidden layer, the activation scale becomes
   `activationScale * W / 2^shift`.
5. Hidden layers perform ReLU, truncating right shift, and keep the lowest
   `activationBits` bits. This is wrapping bit extraction, **not saturation**.
   The shift is chosen to avoid wrapping for valid exported parameters.
6. The final layer is identity: keep the wide signed MAC scores without ReLU
   or shift. Its exported shift is unused. Argmax ties select the lowest index.
7. Export checks `A * sum(abs(weights)) + abs(bias)` for each neuron. Inference
   uses int64 temporaries and rejects values outside the exported MAC range.

`network.json` records these policies explicitly. Python rejects unsupported
versions/policies rather than silently substituting 8-bit or saturating rules.
New rounding or saturation policies need matching generator, reference and tests;
they are not selectable in this change.

## Repeat the pipeline

With Python dependencies installed and the checked-in MNIST test data available:

```sh
make reference-test WEIGHT_BITS=4 ACTIVATION_BITS=4 BIAS_BITS=32
make reference-test WEIGHT_BITS=8 ACTIVATION_BITS=8 BIAS_BITS=32
make reference-test WEIGHT_BITS=12 ACTIVATION_BITS=12 BIAS_BITS=32
```

The same Python code consumes each Scala export. No retraining is performed.
`make reference` exports parameters, evaluates both models on 10,000 images and
writes the first 64 test inputs plus expected integer outputs. `reference-test`
also checks the exported arithmetic and vector round trip. The report records
all three configured widths. Override `PYTHON`/`SBT` for your environment.
To retain separate exports, use the exporter `--output` option and pass the
same directory to evaluation with `--quantized-dir`.

These are software-reference tests, not a full MLP hardware simulation or a board
accuracy test. Those require the MLP core and comparisons from issues #14/#15/#21.
