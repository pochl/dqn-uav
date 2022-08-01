help:
	@echo "format - format Python code with isort/Black"
	@echo "lint - check style with pylint"
	@echo "mypy - run the static type checker"
	@echo "check - run all static checks and analyzers"
	@echo "pytest - run the tests and measure the code coverage"
	@echo "test - run the code formatter, linter, type checker, tests and coverage"
	@echo "ci-test - run the Continuous Integration (CI) pipeline (check-only)"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"

format:
	poetry run isort src tests
	poetry run black src tests

lint:
	mkdir -p reports
	poetry run pylint src tests

mypy:
	poetry run mypy src

check: lint mypy
	poetry run isort --check src tests
	poetry run black --check src tests

pytest:
	poetry run pytest

test: format lint mypy pytest

ci-test: check
	make pytest

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.pytest_cache' -exec rm -fr {} +
	find . -name '.mypy_cache' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f coverage.xml
	rm -fr reports/
	rm -fr htmlcov
