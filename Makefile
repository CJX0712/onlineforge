.PHONY: install test demo lint fmt lock

install:
	pip install -e .
	pip install pytest ruff river || true

test:
	python -m pytest -q -W ignore::UserWarning

demo:
	python onlineforge/examples/run_demo.py

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

lock:
	python -m pip freeze > requirements.lock.txt
