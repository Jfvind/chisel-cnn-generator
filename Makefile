# Variables can be overwritten: make board CORE=mlp STREAM_W=16
CORE ?= stub

# --- Per-core defaults: <core>_<VAR>. Anything given on the command line still wins ---
# stub: echoes every beat (+ARG0), so n_out follows n_in
stub_N_IN     := 784
stub_N_OUT     = $(N_IN)
stub_STREAM_W := 32
stub_RUN      := smoke_test.py
stub_EXTRA    :=

# mlp: 28x28 pixels in, 10 signed class scores out (wider than 8 bit -> 32-bit beats)
mlp_N_IN      := 784
mlp_N_OUT     := 10
mlp_STREAM_W  := 32
mlp_RUN       := mlp_test.py
# mlp_test.py needs the quantized export (same config as the bitstream), the Python vectors and MNIST
QUANT_DIR     := generator/generated/quantized
MNIST         := python/data/MNIST/raw/t10k-images-idx3-ubyte.gz python/data/MNIST/raw/t10k-labels-idx1-ubyte.gz
mlp_EXTRA      = $(QUANT_DIR)/network.json $(wildcard $(QUANT_DIR)/*.csv) $(MNIST)

# cnn: placeholders until the core exists
cnn_N_IN      := 784
cnn_N_OUT     := 10
cnn_STREAM_W  := 32
cnn_RUN       := cnn_test.py
cnn_EXTRA     :=

ifndef $(CORE)_STREAM_W
    $(error Unknown CORE '$(CORE)'. Known cores: stub mlp cnn)
endif

N_IN     ?= $($(CORE)_N_IN)
N_OUT    ?= $($(CORE)_N_OUT)
STREAM_W ?= $($(CORE)_STREAM_W)
RUN      ?= $($(CORE)_RUN)
EXTRA    ?= $($(CORE)_EXTRA)
ARGS     ?=
FCLK_MHZ ?= 100
NAME     ?= accel
JOBS     ?= 8
HOST     ?= pynq-direct,pynq

# --- Quantization (Scala WeightExporter + Python reference) ---
WEIGHT_BITS     ?= 8
ACTIVATION_BITS ?= 8
BIAS_BITS       ?= 32
SHIFT_POLICY    ?= worst
WEIGHTS         ?= python/models/csv

# --- Tools (windows detection) ---
ifeq ($(OS),Windows_NT)
    VIVADO_DEFAULT := vivado.bat
    SBT_DEFAULT    := sbt.bat
    PYTHON_DEFAULT := python
else
    VIVADO_DEFAULT := vivado
    SBT_DEFAULT    := sbt
    PYTHON_DEFAULT := python3
endif
VIVADO ?= $(VIVADO_DEFAULT)
SBT    ?= $(SBT_DEFAULT)
PYTHON ?= $(PYTHON_DEFAULT)

# --- Paths + commands ---
BUILD := hw/vivado
RTL_DIR := $(BUILD)/rtl
OVERLAY := $(BUILD)/overlay/$(NAME).bit

GEN_ARGS = --out ../$(RTL_DIR) --core $(CORE) --nIn $(N_IN) --nOut $(N_OUT) \
           --streamW $(STREAM_W) --fclkMhz $(FCLK_MHZ) \
           --weightBiasDir ../$(WEIGHTS) --weightBits $(WEIGHT_BITS) --actBits $(ACTIVATION_BITS) \
           --shiftPolicy $(SHIFT_POLICY)
VIVADO_ARGS = -mode batch -nojournal -log $(BUILD)/vivado.log \
              -source hw/scripts/build.tcl -tclargs $(RTL_DIR) $(BUILD) $(NAME) $(JOBS) 0
EXPORT_ARGS = --weight-bits $(WEIGHT_BITS) --activation-bits $(ACTIVATION_BITS) --bias-bits $(BIAS_BITS) \
              --shift-policy $(SHIFT_POLICY) --input ../$(WEIGHTS)

DEPLOY = $(PYTHON) hw/scripts/deploy.py --overlay $(OVERLAY) --host $(HOST) --run=$(RUN) \
         --args="$(ARGS)" --extra $(EXTRA)

# --- Demo (#18): always the MLP, whatever CORE defaults to ---
DEMO_DIR   := demo/mlp
DEMO_BUILD := hw/vivado/demo
DEMO_ARGS  ?= --demo 5
DEMO_LIVE  ?= --demo 5 --images 200 --out demo_run.json

.PHONY: rtl bitstream deploy board test reference reference-test report compare cpu-bench demo demo-build demo-freeze demo-report clean help	# targets that are not files

help:
	@echo Targets: test, reference, reference-test, rtl, bitstream, deploy, board, clean, report, compare, cpu-bench, demo, demo-build, demo-freeze, demo-report
	@echo Config:  CORE=$(CORE) STREAM_W=$(STREAM_W) N_IN=$(N_IN) N_OUT=$(N_OUT) FCLK_MHZ=$(FCLK_MHZ) RUN=$(RUN) ARGS=$(ARGS)
	@echo Quant:   WEIGHT_BITS=$(WEIGHT_BITS) ACTIVATION_BITS=$(ACTIVATION_BITS) BIAS_BITS=$(BIAS_BITS) SHIFT_POLICY=$(SHIFT_POLICY) WEIGHTS=$(WEIGHTS)
	@echo Tools:   VIVADO=$(VIVADO) SBT=$(SBT) PYTHON=$(PYTHON)

test:
	cd generator && $(SBT) test

# Scala quantizes + exports, Python evaluates float vs integer and writes test vectors
reference:
	cd generator && $(SBT) "runMain service.WeightExporter $(EXPORT_ARGS)"
	cd python && $(PYTHON) -m scripts.evaluate_quantized

reference-test: reference
	cd python && $(PYTHON) -m unittest discover -s tests -t .

rtl:
	cd generator && $(SBT) "runMain shell.ShellGen $(GEN_ARGS)"

# For several builds stored run "make bitstream CORE=stub BUILD=hw/vivado/stub" - new runs doesn't overide stub.
bitstream: rtl
	$(VIVADO) $(VIVADO_ARGS)
	$(PYTHON) hw/scripts/report.py $(BUILD)

deploy:
	$(DEPLOY)
	$(PYTHON) hw/scripts/report.py $(BUILD)

board: bitstream
	$(DEPLOY)
	$(PYTHON) hw/scripts/report.py $(BUILD)

# Vivado reports -> $(BUILD)/reports/dashboard.html   (make report OPEN=1 to open it)
report:
	$(PYTHON) hw/scripts/report.py $(BUILD) $(if $(OPEN),--open)

# Side by side: make compare BUILDS="hw/vivado/stub hw/vivado/mlp-mnc1 hw/vivado/mlp-mnc4"
compare:
	$(PYTHON) hw/scripts/report.py $(BUILDS) -o hw/vivado/compare.html --open

# Time PyTorch (float) and the NumPy integer reference on this PC -> python/results/cpu_benchmark.json
# (the dashboard's "Python vs hardware" card puts it next to the board results)
cpu-bench:
	cd python && $(PYTHON) -m scripts.benchmark_cpu

# Deploy the frozen, verified MLP overlay in demo/mlp and run the demo (no Vivado needed)
demo:
	$(PYTHON) hw/scripts/deploy.py --overlay $(DEMO_DIR)/overlay/$(NAME).bit --host $(HOST) --run=mlp_test.py \
	    --args="$(DEMO_LIVE)" --results=demo_run.json \
	    --extra $(DEMO_DIR)/quant/network.json $(wildcard $(DEMO_DIR)/quant/*.csv) $(MNIST)

# Rebuild the MLP from the current source in its own build dir and run the demo on it (all 10,000 images)
demo-build:
	"$(MAKE)" board CORE=mlp BUILD=$(DEMO_BUILD) ARGS="$(DEMO_ARGS)"

# After demo-build passed: freeze its overlay, reports and the matching export into demo/mlp, then commit demo/mlp
demo-freeze:
	$(PYTHON) hw/scripts/freeze_demo.py --build $(DEMO_BUILD) --quant $(QUANT_DIR) --out $(DEMO_DIR) --name $(NAME)
	$(PYTHON) hw/scripts/report.py $(DEMO_DIR)

# The frozen demo's dashboard (timing, resources, 10,000-image board results)
demo-report:
	$(PYTHON) hw/scripts/report.py $(DEMO_DIR) --open

clean:
	rm -rf $(BUILD)