import numpy as np
import dask.array as da
import pytest
from numpy.testing import assert_array_equal
import zarr


from skallel.model.fn import (
    genotype_array_check,
    genotype_array_is_called,
    genotype_array_is_missing,
    genotype_array_is_hom,
    genotype_array_is_het,
    genotype_array_count_alleles,
)


def test_init():

    # valid data - numpy array, passed through
    data = np.array([[[0, 1], [2, 3], [4, 5]], [[4, 5], [6, 7], [-1, -1]]], dtype="i1")
    gt = genotype_array_check(data)
    assert data is gt

    # valid data - dask array, passed through
    data_dask = da.from_array(data, chunks=(1, 1, 2))
    gt = genotype_array_check(data_dask)
    assert data_dask is gt

    # valid data - zarr array, gets converted to dask array
    data_zarr = zarr.array(data)
    gt = genotype_array_check(data_zarr)
    assert data_zarr is not gt
    assert isinstance(gt, da.Array)

    # valid data (triploid)
    data_triploid = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[-1, -1, -1], [12, 13, 14]],
        ],
        dtype="i1",
    )
    gt = genotype_array_check(data_triploid)
    assert gt is data_triploid

    # bad type
    data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotype_array_check(data)

    # bad dtype
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        with pytest.raises(TypeError):
            genotype_array_check(data)

    # bad ndim
    data = np.array([[0, 1], [2, 3]], dtype="i1")
    with pytest.raises(ValueError):
        genotype_array_check(data)

    # bad ndim
    data = np.array([0, 1], dtype="i1")
    with pytest.raises(ValueError):
        genotype_array_check(data)


def test_is_called():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, True, True], [False, False, False]], dtype=bool)

    # test numpy array
    actual = genotype_array_is_called(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_is_called(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_missing():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, False], [True, True, True]], dtype=bool)

    # test numpy array
    actual = genotype_array_is_missing(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_is_missing(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_hom():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # test numpy array
    actual = genotype_array_is_hom(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_is_hom(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_het():

    data = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)

    # test numpy array
    actual = genotype_array_is_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_is_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_het_triploid():

    data = np.array(
        [[[0, 0, 0], [0, 0, 1], [0, 1, 2]], [[0, 0, -1], [0, 1, -1], [0, -1, -1]]],
        dtype="i1",
    )
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)

    # test numpy array
    actual = genotype_array_is_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_is_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_count_alleles():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[3, 1, 2], [2, 0, 0]], dtype="i4")

    # test numpy array
    actual = genotype_array_count_alleles(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_array_count_alleles(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
