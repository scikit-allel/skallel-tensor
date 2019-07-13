import numpy as np
import dask.array as da
from numpy.testing import assert_array_equal
import pytest
import zarr


from skallel_tensor import (
    genotypes_locate_called,
    genotypes_locate_missing,
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
)


def test_locate_called():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_called(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_called(data[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_called(data[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_called(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    data_zarr = zarr.array(data)
    actual = genotypes_locate_called(data_zarr)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Bad type.
    data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotypes_locate_called(data)

    # Bad dtype.
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        with pytest.raises(TypeError):
            genotypes_locate_called(data)

    # Bad ndim.
    data = np.array([0, 1], dtype="i1")
    with pytest.raises(ValueError):
        genotypes_locate_called(data)


def test_locate_missing():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, False], [True, True, True]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_missing(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_missing(data[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_missing(data[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_missing(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_hom():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_hom(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_hom(data[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_hom(data[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_hom(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_het():

    data = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_het(data[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_het(data[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_het_triploid():

    data = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, -1], [0, 1, -1], [0, -1, -1]],
        ],
        dtype="i1",
    )
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_het(data)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_het(data_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_call():

    data = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    for call in (1, 0), [1, 0], np.array([1, 0]):
        actual = genotypes_locate_call(data, call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)
        # Test row/column.
        actual = genotypes_locate_call(data[0], call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect[0], actual)
        actual = genotypes_locate_call(data[:, 0], call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_locate_call(data_dask, call=np.array([1, 0]))
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # # Test exceptions.
    # with pytest.raises(TypeError):
    #     genotypes_locate_call(data, "foo")
    # with pytest.raises(TypeError):
    #     genotypes_locate_call(data, [[0, 1], [2, 3]])
    # with pytest.raises(ValueError):
    #     genotypes_locate_call(data, (0, 1, 2))


def test_count_alleles():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[3, 1, 2], [1, 1, 0]], dtype="i4")

    # Test numpy array.
    actual = genotypes_count_alleles(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_count_alleles(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_count_alleles(data, max_allele="foo")
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_count_alleles(data, max_allele=[1])
    with pytest.raises(ValueError):
        genotypes_count_alleles(data, max_allele=128)


def test_to_allele_counts():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array(
        [[[2, 0, 0], [1, 1, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotypes_to_allele_counts(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_to_allele_counts(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotypes_to_allele_counts(data, max_allele="foo")
    with pytest.raises(TypeError):
        genotypes_to_allele_counts(data, max_allele=[1])
    with pytest.raises(ValueError):
        genotypes_to_allele_counts(data, max_allele=128)


def test_to_allele_counts_melt():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array(
        [[2, 1, 0], [0, 1, 0], [0, 0, 2], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotypes_to_allele_counts_melt(data, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    actual = genotypes_to_allele_counts_melt(data_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotypes_to_allele_counts_melt(data, max_allele="foo")
    with pytest.raises(ValueError):
        genotypes_to_allele_counts_melt(data, max_allele=128)
