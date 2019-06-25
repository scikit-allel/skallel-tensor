from setuptools import setup, find_packages

setup(
    name="scikit-allel-model",
    description="Array functions and classes for genetic variation data.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=["setuptools>18.0", "setuptools-scm>1.5.4"],
    install_requires=["numpy", "numba", "dask[array]"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "src/skallel/model/version.py",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    maintainer="Alistair Miles",
    maintainer_email="alimanfoo@googlemail.com",
    url="https://github.com/scikit-allel/scikit-allel-model",
    license="MIT",
)
