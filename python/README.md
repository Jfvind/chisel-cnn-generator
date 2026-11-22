# Training

Python trains the float model and exports its weights. Everything after that
runs in Scala: quantization, the integer reference, float/integer accuracy and
the test vectors (`generator/src/main/scala/reference/`). Python's integer
reference stays as an independent check: `python/tests` recompute the vectors
Scala wrote, at every width CI builds.

```text
python/
  scripts/       Training, the independent integer reference, CPU benchmark
  models/        Float checkpoint and csv/ parameter exports
  results/       Accuracy reports, including the selected widths
  tests/         Contract, arithmetic and reference-vector tests
  data/          MNIST images and labels
```

## Setup

From this directory, with Python 3.10+:

```sh
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
python -m pip install -r requirements.txt
```

Optional retraining: `python -m scripts.train_model`. The supplied checkpoint,
float CSVs and test dataset are enough to repeat evaluation without retraining.

## Quantize and evaluate

From the repository root, using the activated Python environment:

```sh
make reference-test WEIGHT_BITS=4 ACTIVATION_BITS=4 BIAS_BITS=32
```

Change only the Scala configuration variables to repeat with another precision.
There are no bit-width edits in Python. Defaults are 8-bit weights/activations
and 32-bit bias. `make reference` omits the Python test run; both targets export
fresh parameters, evaluate float/integer accuracy and produce 64 deterministic
input/output vectors. `PYTHON=/path/to/python` selects another environment.

The equivalent commands are:

```sh
cd generator
sbt "runMain service.WeightExporter --weight-bits 4 --activation-bits 4 --bias-bits 32" \
    "runMain reference.Evaluate --float-dir ../python/models/csv"
cd ../python
python -m unittest discover -s tests -t .
```

The reference is driven by layer metadata rather than hardcoded layer counts.
The evaluation specifically uses MNIST, so it requires 784 inputs; the float
accuracy is computed in Scala from the float CSVs `train_model.py` exports.

With the supplied 8-bit configuration the baseline is float **96.61%**, integer
**96.69%**. `results/quantization_results.csv` records the actual widths for
each new run.

See [quantization rules](../docs/quantization.md) and
[export/test-vector format](../docs/export-format.md) for the contract, supported
width ranges and how to consume the same vectors in future hardware tests.
