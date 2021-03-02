.PHONY: build check
PY ?= python3

check: flake
	python -m pytest xemc3/

recheck: flake
	python -m pytest xemc3 --last-failed --new-first

flake:
	flake8 xemc3 --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 xemc3 --count --exit-zero --max-complexity=10 --ignore E203 --max-line-length=127 --statistics
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
