# Contributing

## Fork and clone the repo

Go to https://github.com/scikit-allel/scikit-allel-model and click the
"Fork" button to fork the repository to your own GitHub account.

Clone your fork of the scikit-allel-model repo to your local computer:

```bash
git clone git@github.com:username/scikit-allel-model.git
cd scikit-allel-model
```

...replacing "username" with your own GitHub username.

The rest of this guide assumes your current working directory is your
local clone of the scikit-allel-model repo.

## Install development environment

Make sure you have Python 3.7 installed on your local system somehow.

Create a virtual environment for development work on the
scikit-allel-model distribution. E.g., install
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/),
then run:

```bash
mkvirtualenv --python=/usr/bin/python3.7 scikit-allel-model
pip install -r requirements.txt
pip install -e .
```

...replacing "/usr/bin/python3.7" with whatever is the path to your
Python 3.7 binary.

Whenever you want to do development work on scikit-allel-model,
remember to activate the virtual environment, e.g.:

```bash
workon scikit-allel-model
```

## Run tests

To run the unit tests manually:

```bash
pytest -v
```

To run the unit tests via tox, with coverage report:

```bash
tox -e py37-nojit
```

## Code style

Please use [black](https://black.readthedocs.io/en/stable/index.html) to format all 
Python files.

Python files should also pass [flake8](http://flake8.pycqa.org/en/latest/) checks. 
E.g., run:

```bash
flake8 --max-line-length=88
```
