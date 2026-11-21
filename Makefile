# Variables can be overwritten: make board CORE=mlp STREAM_W=16
CORE ?= stub

# ===========================================================
# Per-core defaults: <core>_<VAR>. Anything given on the command line still wins
# ===========================================================
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
# rtl runs `make board-reference` first: ShellGen builds the hardware from that export in the build
# folder, and mlp_test.py checks the board against the same export, its test vectors and MNIST
mlp_PREREQ    := board-reference
MNIST         := python/data/MNIST/raw/t10k-images-idx3-ubyte.gz python/data/MNIST/raw/t10k-labels-idx1-ubyte.gz
mlp_EXTRA      = $(BOARD_QUANT_DIR)/network.json $(wildcard $(BOARD_QUANT_DIR)/*.csv) $(MNIST)

ifndef $(CORE)_STREAM_W
    $(error Unknown CORE '$(CORE)'. Known cores: stub mlp)
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

# ===========================================================
# Quantization (Scala WeightExporter + reference.Evaluate)
# ===========================================================
WEIGHT_BITS     ?= 8
ACTIVATION_BITS ?= 8
BIAS_BITS       ?= 32
SHIFT_POLICY    ?= worst
WEIGHTS         ?= python/models/csv
# The committed export (make reference, sbt test, CI) and the one a board build makes from its own settings
QUANT_DIR       ?= generator/generated/quantized
BOARD_QUANT_DIR  = $(BUILD)/quantized

# ==========================================================
# Windows detection
# =========================================================
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


# =========================================================
# Paths + commands
# =========================================================
BUILD := hw/vivado
RTL_DIR := $(BUILD)/rtl
OVERLAY := $(BUILD)/overlay/$(NAME).bit

GEN_ARGS = --out ../$(RTL_DIR) --core $(CORE) --nIn $(N_IN) --nOut $(N_OUT) \
           --streamW $(STREAM_W) --fclkMhz $(FCLK_MHZ) --quantizedDir ../$(BOARD_QUANT_DIR)
VIVADO_ARGS = -mode batch -nojournal -log $(BUILD)/vivado.log \
              -source hw/scripts/build.tcl -tclargs $(RTL_DIR) $(BUILD) $(NAME) $(JOBS) 0
EXPORT_ARGS = --weight-bits $(WEIGHT_BITS) --activation-bits $(ACTIVATION_BITS) --bias-bits $(BIAS_BITS) \
              --shift-policy $(SHIFT_POLICY) --input ../$(WEIGHTS)
# $(call REFERENCE,export folder,results CSV), both relative to the repository root
REFERENCE = cd generator && $(SBT) "runMain service.WeightExporter $(EXPORT_ARGS) --output ../$(1)" \
            "runMain reference.Evaluate --quantized-dir ../$(1) --float-dir ../$(WEIGHTS) --results ../$(2)"

DEPLOY = $(PYTHON) hw/scripts/deploy.py --overlay $(OVERLAY) --host $(HOST) --run=$(RUN) \
         --args="$(ARGS)" --extra $(EXTRA)
# =========================================================


.PHONY: rtl bitstream deploy board test reference board-reference reference-test report compare cpu-bench camera clean help	# targets that are not files

help:
	@echo Targets: test, reference, reference-test, rtl, bitstream, deploy, board, clean, report, compare, cpu-bench, camera
	@echo Config:  CORE=$(CORE) STREAM_W=$(STREAM_W) N_IN=$(N_IN) N_OUT=$(N_OUT) FCLK_MHZ=$(FCLK_MHZ) RUN=$(RUN) ARGS=$(ARGS)
	@echo Quant:   WEIGHT_BITS=$(WEIGHT_BITS) ACTIVATION_BITS=$(ACTIVATION_BITS) BIAS_BITS=$(BIAS_BITS) SHIFT_POLICY=$(SHIFT_POLICY) WEIGHTS=$(WEIGHTS)
	@echo Tools:   VIVADO=$(VIVADO) SBT=$(SBT) PYTHON=$(PYTHON)


# ==========================================================
# Tests
# ==========================================================
test:
	cd generator && $(SBT) test

# Scala quantizes + exports, then evaluates float vs integer on MNIST and writes the test vectors
reference:
	$(call REFERENCE,$(QUANT_DIR),python/results/quantization_results.csv)

# The same for a board build, kept in $(BUILD) so other widths or shift policies leave the committed export alone
board-reference:
	$(call REFERENCE,$(BOARD_QUANT_DIR),$(BUILD)/quantization_results.csv)

# Python's integer reference recomputes the Scala vectors independently (python/tests)
reference-test: reference
	cd python && $(PYTHON) -m unittest discover -s tests -t .


# ==========================================================
# Board deploying
# ==========================================================
rtl: $($(CORE)_PREREQ)
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


# ==========================================================
# Reporting
# ==========================================================
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


# =========================================================
# Camera
# =========================================================
# Live digits from a USB webcam on the board (Ctrl-C stops)
camera:
	$(PYTHON) hw/scripts/deploy.py --overlay $(OVERLAY) --host $(HOST) --run=camera_demo.py \
	    --results=none.json --extra $(BOARD_QUANT_DIR)/network.json $(wildcard $(BOARD_QUANT_DIR)/*.csv)


# ========================================================
# Clean
# ========================================================
clean:
	rm -rf $(BUILD)