.PHONY: install lint format test demo

install:
	pip install -e ".[dev,xgboost]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest

demo:
	vibnoise synth demo/data
	vibnoise evaluate demo/data --config demo/data/synth_config.yaml \
	    --models logreg svm --leakage-audit --out reports/demo
