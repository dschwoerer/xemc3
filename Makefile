.PHONY: build check
PY ?= python3

check: flake mypy
	python3 -m pytest xemc3/

recheck:
	python3 -m pytest xemc3 --last-failed --new-first

debug-failure:
	python3 -m pytest xemc3 --last-failed --new-first --pdb

flake:
	flake8 xemc3 --count --select=E9,F63,F7,F82 --show-source --statistics
# W503 : line break before binary operator
# E203 : whitespace before ':'
	flake8 xemc3 --count --exit-zero --max-complexity=20 --ignore E203,W503 --exclude test --max-line-length=127 --statistics

mypy:
	mypy xemc3
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
	python3 -m twine upload --repository testpypi dist/*

docs/cli.rst: docs/cli.rst.in.py setup.cfg
	python $< > $@.tmp
	mv $@.tmp $@

doc: docs/cli.rst
	sphinx-build docs/ html/
	@echo Documentation is in file://$$(pwd)/html/index.html


coverage:
	coverage run -m pytest xemc3
	coverage html --include=./* --omit=xemc3/test/*
	@echo Report is in file://$$(pwd)/htmlcov/index.html

coverage-all:
	for i in {6..9} ; do coverage-3.$i run -p -m pytest xemc3; done
	coverage html --include=./* --omit=xemc3/test/*
	@echo Report is in file://$$(pwd)/htmlcov/index.html
