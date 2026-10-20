WEIGHT_BITS ?= 8
ACTIVATION_BITS ?= 8
BIAS_BITS ?= 32

ifeq ($(OS),Windows_NT)
    SBT ?= sbt.bat
    PYTHON ?= python
else
    SBT ?= sbt
    PYTHON ?= python3
endif

.PHONY: help test reference reference-test

help:
	@echo Targets: test, reference, reference-test
	@echo Config: WEIGHT_BITS=$(WEIGHT_BITS) ACTIVATION_BITS=$(ACTIVATION_BITS) BIAS_BITS=$(BIAS_BITS)
	@echo Tools: SBT=$(SBT) PYTHON=$(PYTHON)

test:
	cd generator && $(SBT) test

reference:
	cd generator && $(SBT) "runMain service.WeightExporter --weight-bits $(WEIGHT_BITS) --activation-bits $(ACTIVATION_BITS) --bias-bits $(BIAS_BITS)"
	cd python && $(PYTHON) -m scripts.evaluate_quantized

reference-test: reference
	cd python && $(PYTHON) -m unittest discover -s tests -t .
