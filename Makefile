.PHONY: install test bench serve lint

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test: lint
	pytest

bench:
	python scripts/run_benchmark.py --write-readme

serve:
	pocket-qa serve --host 127.0.0.1 --port 8741
