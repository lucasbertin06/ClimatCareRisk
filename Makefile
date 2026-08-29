SHELL := bash

PYTHON ?= python
TESSERACT ?= tesseract
FIRE_IMAGE ?= fire_spread_torch
SMOKE_IMAGE ?= smoke_transport_cpp

.PHONY: build-fire build-smoke build-c0 smoke-kernel tiny-direct tiny-gradient tiny-map test clean

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
	cmake -S components/tesseracts/smoke_transport_cpp -B components/tesseracts/smoke_transport_cpp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build components/tesseracts/smoke_transport_cpp/build --config Release

# Runs the reproducible direct C0 command and writes tiny_direct.json.
tiny-direct:
	$(PYTHON) -m climacare.cli direct --output-dir results/tiny_direct

# Runs the end-to-end gradient tests for the Tiny case.
tiny-gradient:
	$(PYTHON) -m pytest tests/test_gradient_pipeline.py -q

# Runs the MAP/inverse tests for the Tiny case.
tiny-map:
	$(PYTHON) -m climacare.cli map --output-dir results/tiny_map

test:
	$(PYTHON) -m pytest tests -q

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('components/tesseracts/smoke_transport_cpp/build', ignore_errors=True)"
	$(PYTHON) -c "import shutil; shutil.rmtree('results/tiny_direct', ignore_errors=True)"

# On Windows PowerShell, use the explicit Tesseract path if tesseract is not on PATH:
# make TESSERACT='C:/Users/Alexis/AppData/Local/Python/pythoncore-3.14-64/Scripts/tesseract.exe' build-c0
# On Windows, run make targets from Git Bash if GNU make is installed.
