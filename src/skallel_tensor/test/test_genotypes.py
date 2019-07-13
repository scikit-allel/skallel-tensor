import numpy as np
import dask.array as da
from numpy.testing import assert_array_equal
import pytest
import zarr


from skallel_tensor import (
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_called_allele_counts,
    genotypes_to_missing_allele_counts,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
)


def test_locate_hom():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_hom(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_hom(gt[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_hom(gt[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_locate_hom(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_het():

    gt = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_het(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_locate_het(gt[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_locate_het(gt[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_locate_het(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_het_triploid():

    gt = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, -1], [0, 1, -1], [0, -1, -1]],
        ],
        dtype=np.int8,
    )
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)

    # Test numpy array.
    actual = genotypes_locate_het(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_locate_het(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_locate_call():

    gt = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[False, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    for call in (1, 0), [1, 0], np.array([1, 0]):
        actual = genotypes_locate_call(gt, call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)
        # Test row/column.
        actual = genotypes_locate_call(gt[0], call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect[0], actual)
        actual = genotypes_locate_call(gt[:, 0], call=call)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_locate_call(gt_dask, call=np.array([1, 0]))
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(ValueError):
        genotypes_locate_call(gt, call="foo")
    with pytest.raises(ValueError):
        genotypes_locate_call(gt, call=[[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        genotypes_locate_call(gt, call=(0, 1, 2))


def test_count_alleles():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[3, 1, 2], [1, 1, 0]], dtype="i4")

    # Test numpy array.
    actual = genotypes_count_alleles(gt, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_count_alleles(gt_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_count_alleles(gt, max_allele="foo")
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_count_alleles(gt, max_allele=[1])
    with pytest.raises(ValueError):
        genotypes_count_alleles(gt, max_allele=128)


def test_to_called_allele_counts():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[2, 2, 2], [1, 1, 0]], dtype=np.int8)

    # Test numpy array.
    actual = genotypes_to_called_allele_counts(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_to_called_allele_counts(gt[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_to_called_allele_counts(gt[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_to_called_allele_counts(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    gt_zarr = zarr.array(gt)
    actual = genotypes_to_called_allele_counts(gt_zarr)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Bad type.
    gt = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotypes_to_called_allele_counts(gt)

    # TODO revisit dtype restrictions and jitting
    # # Bad dtype.
    # for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
    #     gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
    #     with pytest.raises(TypeError):
    #         genotypes_to_called_allele_counts(gt)

    # Bad ndim.
    gt = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError):
        genotypes_to_called_allele_counts(gt)


def test_to_missing_allele_counts():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[0, 0, 0], [1, 1, 2]], dtype=np.int8)

    # Test numpy array.
    actual = genotypes_to_missing_allele_counts(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = genotypes_to_missing_allele_counts(gt[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = genotypes_to_missing_allele_counts(gt[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_to_missing_allele_counts(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_to_allele_counts():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array(
        [[[2, 0, 0], [1, 1, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotypes_to_allele_counts(gt, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_to_allele_counts(gt_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_to_allele_counts(gt, max_allele="foo")
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_to_allele_counts(gt, max_allele=[1])
    with pytest.raises(ValueError):
        genotypes_to_allele_counts(gt, max_allele=128)


def test_to_allele_counts_melt():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array(
        [[2, 1, 0], [0, 1, 0], [0, 0, 2], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype="i4",
    )

    # Test numpy array.
    actual = genotypes_to_allele_counts_melt(gt, max_allele=2)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 1, -1))
    actual = genotypes_to_allele_counts_melt(gt_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        genotypes_to_allele_counts_melt(gt, max_allele="foo")
    with pytest.raises(ValueError):
        genotypes_to_allele_counts_melt(gt, max_allele=128)
