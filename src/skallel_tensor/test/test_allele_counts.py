import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import dask.array as da
import zarr
import pytest


from skallel_tensor import (
    allele_counts_to_frequencies,
    allele_counts_allelism,
    allele_counts_max_allele,
)


def _test_ac_func(f, ac, expect, compare):

    # 3D tests.
    assert ac.ndim == 3

    # Test numpy array.
    actual = f(ac)
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)
    assert expect.dtype == actual.dtype

    # Test numpy array, Fortran order.
    actual = f(np.asfortranarray(ac))
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(1, 2, -1))
    actual = f(ac_dask)
    assert isinstance(actual, da.Array)
    compare(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # Test zarr array.
    ac_zarr = zarr.array(data=ac)
    actual = f(ac_zarr)
    assert isinstance(actual, da.Array)
    compare(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # Reshape to test as 2D.
    ac = ac.reshape((-1, ac.shape[2]))
    if expect.ndim == 3:
        expect = expect.reshape(ac.shape)
    elif expect.ndim == 2:
        expect = expect.reshape(-1)

    # Test numpy array.
    actual = f(ac)
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = f(ac_dask)
    assert isinstance(actual, da.Array)
    compare(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # Test zarr array.
    ac_zarr = zarr.array(data=ac)
    actual = f(ac_zarr)
    assert isinstance(actual, da.Array)
    compare(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # Test errors.
    with pytest.raises(TypeError):
        # Wrong type.
        f("foo")
    with pytest.raises(TypeError):
        # Wrong dtype.
        f(ac.astype("f4"))
    with pytest.raises(ValueError):
        # Wrong ndim.
        f(ac[0])


def test_to_frequencies():
    ac = np.array(
        [
            [[3, 0, 0], [0, 3, 0], [2, 1, 0], [0, 1, 2], [1, 1, 1]],
            [[2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, -1], [-1, -1, -1]],
        ],
        dtype=np.int32,
    )
    expect = np.array(
        [
            [
                [3 / 3, 0 / 3, 0 / 3],
                [0 / 3, 3 / 3, 0 / 3],
                [2 / 3, 1 / 3, 0 / 3],
                [0 / 3, 1 / 3, 2 / 3],
                [1 / 3, 1 / 3, 1 / 3],
            ],
            [
                [2 / 2, 0 / 2, 0 / 2],
                [0 / 2, 2 / 2, 0 / 2],
                [1 / 2, 1 / 2, 0 / 2],
                [0 / 2, 1 / 2, 1 / 2],
                [1 / 2, 0 / 2, 1 / 2],
            ],
            [
                [1 / 1, 0 / 1, 0 / 1],
                [0 / 1, 1 / 1, 0 / 1],
                [np.nan, np.nan, np.nan],
                [0 / 1, 1 / 1, np.nan],
                [np.nan, np.nan, np.nan],
            ],
        ],
        dtype=np.float32,
    )
    _test_ac_func(allele_counts_to_frequencies, ac, expect, assert_allclose)


def test_allelism():
    ac = np.array(
        [
            [[3, 0, 0], [0, 3, 0], [2, 1, 0], [0, 1, 2], [1, 1, 1]],
            [[2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [-1, -1, -1]],
        ],
        dtype=np.int32,
    )
    expect = np.array(
        [[1, 1, 2, 2, 3], [1, 1, 2, 2, 2], [1, 1, 1, 0, 0]], dtype=np.int8
    )
    _test_ac_func(allele_counts_allelism, ac, expect, assert_array_equal)


def test_max_allele():
    ac = np.array(
        [
            [[3, 0, 0], [0, 3, 0], [2, 1, 0], [0, 1, 2], [1, 1, 1]],
            [[2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [-1, -1, -1]],
        ],
        dtype=np.int32,
    )
    expect = np.array(
        [[0, 1, 1, 2, 2], [0, 1, 1, 2, 2], [0, 1, 2, -1, -1]], dtype=np.int8
    )
    _test_ac_func(allele_counts_max_allele, ac, expect, assert_array_equal)
