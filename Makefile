# Variables can be overwritten: make board CORE=mlp STREAM_W=16
CORE ?= stub

# --- Per-core defaults: <core>_<VAR>. Anything given on the command line still wins ---
# stub: echoes every beat (+ARG0), so n_out follows n_in
stub_N_IN     := 784
stub_N_OUT     = $(N_IN)
stub_STREAM_W := 32
stub_RUN      := smoke_test.py

# mlp: 28x28 pixels in, 10 signed class scores out (wider than 8 bit -> 32-bit beats)
mlp_N_IN      := 784
mlp_N_OUT     := 10
mlp_STREAM_W  := 32
mlp_RUN       := mlp_test.py

# cnn: placeholders until the core exists
cnn_N_IN      := 784
cnn_N_OUT     := 10
cnn_STREAM_W  := 32
cnn_RUN       := cnn_test.py

ifndef $(CORE)_STREAM_W
    $(error Unknown CORE '$(CORE)'. Known cores: stub mlp cnn)
endif

N_IN     ?= $($(CORE)_N_IN)
N_OUT    ?= $($(CORE)_N_OUT)
STREAM_W ?= $($(CORE)_STREAM_W)
RUN      ?= $($(CORE)_RUN)
FCLK_MHZ ?= 100
NAME     ?= accel
JOBS     ?= 8
HOST     ?= pynq-direct,pynq

# --- Quantization (Scala WeightExporter + Python reference) ---
WEIGHT_BITS     ?= 8
ACTIVATION_BITS ?= 8
BIAS_BITS       ?= 32

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
           --streamW $(STREAM_W) --fclkMhz $(FCLK_MHZ)
VIVADO_ARGS = -mode batch -nojournal -log $(BUILD)/vivado.log \
              -source hw/scripts/build.tcl -tclargs $(RTL_DIR) $(BUILD) $(NAME) $(JOBS) 0
EXPORT_ARGS = --weight-bits $(WEIGHT_BITS) --activation-bits $(ACTIVATION_BITS) --bias-bits $(BIAS_BITS)

DEPLOY = $(PYTHON) hw/scripts/deploy.py --overlay $(OVERLAY) --host $(HOST) --run=$(RUN)

.PHONY: rtl bitstream deploy board test reference reference-test clean help	# targets that are not files

help:
	@echo Targets: test, reference, reference-test, rtl, bitstream, deploy, board, clean
	@echo Config:  CORE=$(CORE) STREAM_W=$(STREAM_W) N_IN=$(N_IN) N_OUT=$(N_OUT) FCLK_MHZ=$(FCLK_MHZ) RUN=$(RUN)
	@echo Quant:   WEIGHT_BITS=$(WEIGHT_BITS) ACTIVATION_BITS=$(ACTIVATION_BITS) BIAS_BITS=$(BIAS_BITS)
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

bitstream: rtl
	$(VIVADO) $(VIVADO_ARGS)

deploy:
	$(DEPLOY)

board: bitstream
	$(DEPLOY)

clean:
	rm -rf $(BUILD)