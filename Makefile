.PHONY: install test bench serve lint

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test: lint
	pytest

bench:
	python scripts/run_benchmark.py --write-readme

# Opt-in. Downloads the local NLI checkpoint on first use. Not run by CI.
compare:
	python scripts/compare_claim_support.py

serve:
	pocket-qa serve --host 127.0.0.1 --port 8741
