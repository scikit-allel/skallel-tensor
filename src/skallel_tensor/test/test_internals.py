import numpy as np
import dask.array as da
import pytest
from skallel_tensor import functions, methods_numpy, methods_dask


def test_get_methods():
    a = np.arange(100)
    m = functions.get_methods(a)
    assert m is methods_numpy
    d = da.from_array(a)
    m = functions.get_methods(d)
    assert m is methods_dask
    with pytest.raises(TypeError):
        functions.get_methods("foo")


def test_accepts():
    a = np.arange(100)
    assert methods_numpy.accepts(a)
    assert methods_dask.accepts(a)
    d = da.from_array(a)
    assert not methods_numpy.accepts(d)
    assert methods_dask.accepts(d)
