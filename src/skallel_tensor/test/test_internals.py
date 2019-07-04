import numpy as np
import dask.array as da
import pytest
from skallel_tensor import api, numpy_backend, dask_backend


def test_get_backend():
    a = np.arange(100)
    m = api.get_backend(a)
    assert m is numpy_backend
    d = da.from_array(a)
    m = api.get_backend(d)
    assert m is dask_backend
    with pytest.raises(TypeError):
        api.get_backend("foo")


def test_accepts():
    a = np.arange(100)
    assert numpy_backend.accepts(a)
    assert dask_backend.accepts(a)
    d = da.from_array(a)
    assert not numpy_backend.accepts(d)
    assert dask_backend.accepts(d)
