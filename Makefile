SHELL := bash

PYTHON ?= python
TESSERACT ?= tesseract
FIRE_IMAGE ?= fire_spread_torch
SMOKE_IMAGE ?= smoke_transport_cpp
PYTHON_EXECUTABLE := $(shell $(PYTHON) -c "import sys; print(sys.executable)")

.PHONY: build-fire docker-tag-fire build-smoke docker-tag-smoke build-c0 smoke-kernel tiny-direct tiny-gradient tiny-map test clean

build-fire:
	$(TESSERACT) build components/tesseracts/fire_spread_torch -t $(FIRE_IMAGE)

docker-tag-fire:
	docker tag $(FIRE_IMAGE):$(FIRE_IMAGE) $(FIRE_IMAGE):latest

build-smoke:
	$(TESSERACT) build components/tesseracts/smoke_transport_cpp -t $(SMOKE_IMAGE)

docker-tag-smoke:
	docker tag $(SMOKE_IMAGE):$(SMOKE_IMAGE) $(SMOKE_IMAGE):latest

build-c0: build-fire docker-tag-fire build-smoke docker-tag-smoke

smoke-kernel:
	$(PYTHON) -c "import pybind11" || { echo 'pybind11 is required; install app[dev]'; exit 1; }
	cmake -S components/tesseracts/smoke_transport_cpp -B components/tesseracts/smoke_transport_cpp/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(PYTHON_EXECUTABLE)
	cmake --build components/tesseracts/smoke_transport_cpp/build --config Release

tiny-direct:
	$(PYTHON) -m climacare.cli direct --output-dir results/tiny_direct

tiny-gradient:
	$(PYTHON) -m pytest tests/test_gradient_pipeline.py -q

tiny-map:
	$(PYTHON) -m climacare.cli map --output-dir results/tiny_map

test:
	$(PYTHON) -m pytest tests -q

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('components/tesseracts/smoke_transport_cpp/build', ignore_errors=True)"
	$(PYTHON) -c "import shutil; shutil.rmtree('results/tiny_direct', ignore_errors=True)"
