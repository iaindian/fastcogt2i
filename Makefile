PYTHON_3_10_BIN_PATH := $(shell which python3.10)

VENV_PY_3_10 := .venv-3.10/bin/python
VENV_PIP_3_10 := .venv-3.10/bin/pip

check-python-3.10:
	@if [ -z "$(PYTHON_3_10_BIN_PATH)" ]; then \
		echo "Error: Python 3.10 is not installed or not in PATH."; \
		echo "Please install Python 3.10 and make sure it's available in your PATH."; \
		exit 1; \
	elif ! $(PYTHON_3_10_BIN_PATH) -c "import sys; sys.exit(0)"; then \
		echo "Error: Python 3.10 is not functioning correctly."; \
		echo "Please ensure Python 3.10 is properly installed and configured."; \
		exit 1; \
	fi

init: check-python-3.10
	${PYTHON_3_10_BIN_PATH} -m pip install --upgrade pip
	${PYTHON_3_10_BIN_PATH} -m venv .venv-3.10
	${VENV_PIP_3_10} install --upgrade pip
	${VENV_PIP_3_10} install -r requirements-fal-dev.txt

run:
	${VENV_PY_3_10} -m fal run falapp.py

deploy:
	${VENV_PY_3_10} -m fal deploy falapp.py
