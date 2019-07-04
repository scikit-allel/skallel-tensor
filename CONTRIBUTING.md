# Contributing

## Fork and clone the repo

Go to https://github.com/scikit-allel/skallel-tensor and click the
"Fork" button to fork the repository to your own GitHub account.

Clone your fork of the skallel-tensor repo to your local computer:

```bash
git clone git@github.com:username/skallel-tensor.git
cd skallel-tensor
```

...replacing "username" with your own GitHub username.

The rest of this guide assumes your current working directory is your
local clone of the skallel-tensor repo.

## Install development environment

Make sure you have Python 3.7 installed on your local system somehow.

Create a virtual environment for development work on the
skallel-tensor distribution. E.g., install
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/),
then run:

```bash
mkvirtualenv --python=/usr/bin/python3.7 skallel-tensor
pip install -r requirements.txt
pip install -e .
```

...replacing "/usr/bin/python3.7" with whatever is the path to your
Python 3.7 binary.

Whenever you want to do development work on skallel-tensor,
remember to activate the virtual environment, e.g.:

```bash
workon skallel-tensor
```

## Run tests

To run the unit tests manually:

```bash
pytest -v
```

To run the unit tests via tox, with coverage report:

```bash
tox -e py37-cov
```

## Code style

Please use [black](https://black.readthedocs.io/en/stable/index.html) to format all 
Python files.

Python files should also pass [flake8](http://flake8.pycqa.org/en/latest/) checks. 
E.g., run:

```bash
flake8 --max-line-length=88
```

## Running benchmarks

To run the [ASV](https://asv.readthedocs.org/) benchmarks against the
current state of the source code:

```
asv dev
```

...or:

```
asv run --python=same
```

To run the benchmarks against the latest commit in the master branch:

```
asv run
```

To test a range of commits, e.g., on a particular branch since
branching off master:

```
asv run master..mybranch
```

To view benchmarking results:

```
asv publish
asv preview
```

## Release

To release a new version, create a release on GitHub. Use a tag like
"v0.1.0" for normal releases, or "v0.1.0a1" for an alpha pre-release,
or "v0.1.0b1" for a beta pre-release. Follow [semantic
versioning(https://semver.org/). The release will automatically be
deployed to PyPI by Travis CI.
