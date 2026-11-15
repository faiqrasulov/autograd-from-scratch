PYTHON ?= python
.PHONY: install install-locked data run test lint clean

install:  ## install the package with dev extras (PyTorch CPU is used only as the reference)
	$(PYTHON) -m pip install -e ".[dev]"

# Exact package versions that produced the committed reports/ (see requirements.lock).
install-locked:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

data:  ## load_digits ships inside scikit-learn: nothing to download, just print the split
	$(PYTHON) -m tensorgrad.data

run:  ## full reproduction: op checks, MLP + CNN training, PyTorch parity, metrics + figures
	$(PYTHON) -m tensorgrad.cli --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
