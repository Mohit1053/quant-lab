.PHONY: install install-dev lint format test test-unit test-integration ingest features pipeline pretrain train train-rl detect-regimes backtest report mlflow clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

test:
	pytest tests/ -v --cov=quant_lab --cov-report=term-missing

test-unit:
	pytest tests/ -v --ignore=tests/test_integration -m "not slow"

test-integration:
	pytest tests/test_integration/ -v

ingest:
	python scripts/ingest_data.py

features:
	python scripts/compute_features.py

pipeline:
	python scripts/run_pipeline.py

pretrain:
	python scripts/pretrain.py

train:
	python scripts/train_forecaster.py

train-rl:
	python scripts/train_rl.py

detect-regimes:
	python scripts/detect_regimes.py

backtest:
	python scripts/run_backtest.py

report:
	python scripts/generate_report.py

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
