PY ?= python
.PHONY: install install-dev data train train-indobert-smoke evaluate test lint format run demo bot docker purge all

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e ".[train,ocr]"

data:
	$(PY) scripts/prepare_data.py --config configs/data.yaml

train: data
	$(PY) scripts/train_baseline.py --config configs/baseline.yaml

train-indobert-smoke:
	$(PY) scripts/train_indobert.py --config configs/indobert.yaml --smoke

evaluate:
	$(PY) scripts/evaluate_ood.py --model-dir models/baseline

test:
	$(PY) -m pytest --cov --cov-report=term-missing

lint:
	ruff check src tests scripts demo && ruff format --check src tests scripts

format:
	ruff format src tests scripts demo && ruff check --fix src tests scripts demo

run:
	uvicorn cekdulu.api.app:create_app --factory --reload --port 8000

demo:
	$(PY) demo/app.py

bot:
	$(PY) -m cekdulu.bot

docker:
	docker compose up -d --build

purge:
	$(PY) -m cekdulu.maintenance --days 30

all: install-dev train evaluate lint test
