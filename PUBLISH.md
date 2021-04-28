# How to publish

## Run all tests
```bash
make check
```
ensure GHA succeed


## Tag commit
```bash
git tag 0.0.x
```

## Install dependencies
```bash
python3 -m pip install --user --upgrade setuptools wheel twine
```

## Do release

```bash
rm -rf dist
python3 setup.py sdist
python3 -m pip wheel . -w dist/ --no-deps
# maybe without testpypi repositroy
python -m twine upload --repository testpypi dist/*
```
