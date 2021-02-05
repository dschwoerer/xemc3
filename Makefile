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
	#python3 -m twine upload dist/eudist*.tar.gz
	python -m twine upload --repository testpypi dist/*

doc:
	sphinx-build docs/ html/


coverage:
	coverage run -m pytest
	coverage html --include=./*,xemc3/*,xemc3/*/*,xemc3/*/*/*
