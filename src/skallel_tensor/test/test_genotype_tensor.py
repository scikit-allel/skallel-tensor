import numpy as np
import dask.array as da
import pytest
from numpy.testing import assert_array_equal
import zarr


from skallel_tensor.functions import (
    genotype_tensor_check,
    genotype_tensor_is_called,
    genotype_tensor_is_missing,
    genotype_tensor_is_hom,
    genotype_tensor_is_het,
    genotype_tensor_is_call,
    genotype_tensor_count_alleles,
    genotype_tensor_to_allele_counts,
    genotype_tensor_to_allele_counts_melt,
)


def test_tensor_check():

    # Valid data - numpy array, passed through.
    data = np.array(
        [[[0, 1], [2, 3], [4, 5]], [[4, 5], [6, 7], [-1, -1]]], dtype="i1"
    )
    genotype_tensor_check(data)

    # Valid data - dask array, passed through.
    data_dask = da.from_array(data, chunks=(1, 1, 2))
    genotype_tensor_check(data_dask)

    # Valid data - zarr array, gets converted to dask array.
    data_zarr = zarr.array(data)
    genotype_tensor_check(data_zarr)

    # Valid data (triploid).
    data_triploid = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[-1, -1, -1], [12, 13, 14]],
        ],
        dtype="i1",
    )
    genotype_tensor_check(data_triploid)

    # Bad type.
    data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotype_tensor_check(data)

    # Bad dtype.
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        with pytest.raises(TypeError):
            genotype_tensor_check(data)

    # Bad ndim.
    data = np.array([[0, 1], [2, 3]], dtype="i1")
    with pytest.raises(ValueError):
        genotype_tensor_check(data)

    # Bad ndim.
    data = np.array([0, 1], dtype="i1")
    with pytest.raises(ValueError):
        genotype_tensor_check(data)


def test_is_called():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_called(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_called(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_missing():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, False], [True, True, True]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_missing(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_missing(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_hom():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_hom(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_hom(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_het():

    data = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_het_triploid():

    data = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, -1], [0, 1, -1], [0, -1, -1]],
        ],
        dtype="i1",
    )
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_is_call():

    data = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotype_tensor_is_call(data, (1, 0))
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_is_call(data_dask, (1, 0))
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotype_tensor_is_call(data, "foo")
    with pytest.raises(TypeError):
        genotype_tensor_is_call(data, [[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        genotype_tensor_is_call(data, (0, 1, 2))


def test_count_alleles():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[3, 1, 2], [1, 1, 0]], dtype="i4")

    # Test numpy array.
    actual = genotype_tensor_count_alleles(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_count_alleles(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotype_tensor_count_alleles(data, max_allele="foo")
    with pytest.raises(ValueError):
        genotype_tensor_count_alleles(data, max_allele=128)


def test_to_allele_counts():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array(
        [[[2, 0, 0], [1, 1, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotype_tensor_to_allele_counts(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_to_allele_counts(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotype_tensor_to_allele_counts(data, max_allele="foo")
    with pytest.raises(ValueError):
        genotype_tensor_to_allele_counts(data, max_allele=128)


def test_to_allele_counts_melt():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array(
        [[2, 1, 0], [0, 1, 0], [0, 0, 2], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotype_tensor_to_allele_counts_melt(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotype_tensor_to_allele_counts_melt(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotype_tensor_to_allele_counts_melt(data, max_allele="foo")
    with pytest.raises(ValueError):
        genotype_tensor_to_allele_counts_melt(data, max_allele=128)
