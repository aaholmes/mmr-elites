.PHONY: install build test lint format clean demo benchmark

# Installation
install:
	pip install -e ".[dev]"

build:
	maturin develop --release

# Testing
test:
	PYTHONPATH=. pytest tests/ -v

test-cov:
	PYTHONPATH=. pytest tests/ --cov=mmr_elites --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Code quality
lint:
	black --check mmr_elites/ tests/ experiments/ examples/
	isort --check mmr_elites/ tests/ experiments/ examples/

format:
	black mmr_elites/ tests/ experiments/ examples/
	isort mmr_elites/ tests/ experiments/ examples/

# Running
demo:
	streamlit run demo/app.py

benchmark:
	mmr-elites benchmark --quick

benchmark-full:
	mmr-elites benchmark --full

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/
	rm -rf results/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Docker
docker-build:
	docker build -t mmr-elites .

docker-run:
	docker run -it mmr-elites mmr-elites benchmark --quick
