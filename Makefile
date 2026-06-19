.PHONY: setup format lint test clean

setup:
pip install -r requirements-dev.txt
pre-commit install

format:
black src tests
isort src tests

lint:
flake8 src tests

test:
pytest tests

clean:
find . -type d -name __pycache__ -exec rm -rf {} +
