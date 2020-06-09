
.PHONY: black
black:  # Formats our code.
	python -m black src

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
	pip-compile
