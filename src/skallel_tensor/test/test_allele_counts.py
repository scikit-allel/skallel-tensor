import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import dask.array as da


from skallel_tensor import (
    allele_counts_to_frequencies,
    allele_counts_allelism,
    allele_counts_max_allele,
    allele_counts_locate_variant,
    allele_counts_locate_non_variant,
    allele_counts_locate_segregating,
)


def test_2d_to_frequencies():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array(
        [
            [3 / 4, 1 / 4, 0 / 4],
            [1 / 4, 2 / 4, 1 / 4],
            [1 / 4, 2 / 4, 1 / 4],
            [0 / 2, 0 / 2, 2 / 2],
            [np.nan, np.nan, np.nan],
            [0 / 3, 1 / 3, 2 / 3],
        ],
        dtype=np.float32,
    )

    # Test numpy array.
    actual = allele_counts_to_frequencies(data)
    assert isinstance(actual, np.ndarray)
    assert_allclose(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_to_frequencies(data_dask)
    assert isinstance(actual, da.Array)
    assert_allclose(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # TODO Test errors.


def test_2d_allelism():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([2, 3, 3, 1, 0, 2], np.int8)

    # Test numpy array.
    actual = allele_counts_allelism(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_allelism(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_max_allele():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([1, 2, 2, 2, -1, 2], dtype=np.int8)

    # Test numpy array.
    actual = allele_counts_max_allele(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_max_allele(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_variant():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([1, 1, 1, 1, 0, 1], dtype=np.bool_)

    # Test numpy array.
    actual = allele_counts_locate_variant(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_locate_variant(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_non_variant():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([0, 0, 0, 0, 1, 0], dtype=np.bool_)

    # Test numpy array.
    actual = allele_counts_locate_non_variant(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_locate_non_variant(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_segregating():
    data = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )

    # Test numpy array.
    actual = allele_counts_locate_segregating(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    data_dask = da.from_array(data, chunks=(2, -1))
    actual = allele_counts_locate_segregating(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype
