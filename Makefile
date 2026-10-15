.PHONY: install install-all lint format typecheck test eval serve ui docker

install:
	pip install -e ".[dev,api,ui]"

install-all:
	pip install -e ".[all,dev]"

lint:
	ruff check src tests eval
	ruff format --check src tests eval

format:
	ruff format src tests eval
	ruff check --fix src tests eval

typecheck:
	mypy src

test:
	pytest --cov=ragdoc --cov-report=term-missing

eval:
	python eval/run_eval.py

serve:
	ragdoc serve --host 0.0.0.0 --port 8000

ui:
	ragdoc ui

docker:
	docker compose -f docker/docker-compose.yml up --build
