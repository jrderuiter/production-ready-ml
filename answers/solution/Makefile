
.PHONY: black
black:  # Formats our code.
	python -m black src

.PHONY: clean
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY: dist
dist:  # Build a distribution of our package
	python -m pep517.build .

.PHONY: docker-build
docker-build: update-reqs
	docker build -t titanic .

.PHONY: docker-run
docker-run:
	# TODO: Change command line for training, etc
	# and mounting data/output volumes.
	docker run --rm titanic --help

.PHONY: lint
lint:  # Lints our code.
	python -m pylint src

.PHONY: test
test:  # Runs tests.
	python -m pytest tests

.PHONY: update-reqs
update-reqs:
	pip-compile requirements.in
	sed -i .bak '/-e/d' requirements.txt && rm requirements.txt.bak
	pip-compile requirements.dev.in
	sed -i .bak '/-e/d' requirements.dev.txt && rm requirements.dev.txt.bak
