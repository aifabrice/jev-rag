.PHONY: test lint compile security build check

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check .

compile:
	python3 -m compileall -q local_kb.py jev_test.py tests

security:
	python3 scripts/check_release.py

build:
	python3 -m build
	python3 -m twine check dist/*

check: security compile lint test build
