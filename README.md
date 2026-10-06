# chisel-cnn-generator
Parameterized Chisel generator for CNN inference accelerators on FPGA. Starts with an MLP and is verified bit-exact against PyTorch.


# Python setup

Tested with 3.13.8.

## Setup

    cd python
    python3 -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

Run `deactivate` to leave the venv.