.PHONY: build check
PY ?= python3


check:
	python -m pytest xemc3/

format:
	black .

install: check
	pip3 install . --user

install-without-check:
	pip3 install . --user

publish: check
	rm -rf dist
	python3 -m pip install --user --upgrade setuptools wheel twine
	python3 setup.py sdist
	#python3 -m twine upload dist/xemc3*.tar.gz
	python -m twine upload --repository testpypi dist/*

doc:
	sphinx-build docs/ html/
	@echo Documentation is in file://$$(pwd)/html/index.html


coverage:
	coverage run -m pytest
	coverage html --include=./* --omit=xemc3/test/*
	@echo Report is in file://$$(pwd)/htmlcov/index.html
