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
    allele_counts_locate_hom,
    allele_counts_locate_het,
    genotypes_to_allele_counts,
)


def test_2d_to_frequencies():
    ac = np.array(
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
    actual = allele_counts_to_frequencies(ac)
    assert isinstance(actual, np.ndarray)
    assert_allclose(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_to_frequencies(ac_dask)
    assert isinstance(actual, da.Array)
    assert_allclose(expect, actual.compute())
    assert expect.dtype == actual.dtype

    # TODO Test errors.


def test_2d_allelism():
    ac = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([2, 3, 3, 1, 0, 2], np.int8)

    # Test numpy array.
    actual = allele_counts_allelism(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_allelism(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_max_allele():
    ac = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([1, 2, 2, 2, -1, 2], dtype=np.int8)

    # Test numpy array.
    actual = allele_counts_max_allele(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_max_allele(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_variant():
    ac = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([1, 1, 1, 1, 0, 1], dtype=np.bool_)

    # Test numpy array.
    actual = allele_counts_locate_variant(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_locate_variant(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_non_variant():
    ac = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([0, 0, 0, 0, 1, 0], dtype=np.bool_)

    # Test numpy array.
    actual = allele_counts_locate_non_variant(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_locate_non_variant(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_2d_locate_segregating():
    ac = np.array(
        [[3, 1, 0], [1, 2, 1], [1, 2, 1], [0, 0, 2], [0, 0, 0], [0, 1, 2]],
        dtype=np.int32,
    )
    expect = np.array([1, 1, 1, 0, 0, 1], dtype=np.bool_)

    # Test numpy array.
    actual = allele_counts_locate_segregating(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)
    assert expect.dtype == actual.dtype

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(2, -1))
    actual = allele_counts_locate_segregating(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
    assert expect.dtype == actual.dtype


def test_3d_locate_hom():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    ac = genotypes_to_allele_counts(gt, max_allele=3)
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = allele_counts_locate_hom(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = allele_counts_locate_hom(ac[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = allele_counts_locate_hom(ac[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(1, 1, -1))
    actual = allele_counts_locate_hom(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_3d_locate_het():

    gt = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    ac = genotypes_to_allele_counts(gt, max_allele=3)
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)

    # Test numpy array.
    actual = allele_counts_locate_het(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test row/column.
    actual = allele_counts_locate_het(ac[0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[0], actual)
    actual = allele_counts_locate_het(ac[:, 0])
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect[:, 0], actual)

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(1, 1, -1))
    actual = allele_counts_locate_het(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())


def test_3d_locate_het_triploid():

    gt = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, -1], [0, 1, -1], [0, -1, -1]],
        ],
        dtype=np.int8,
    )
    ac = genotypes_to_allele_counts(gt, max_allele=2)
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)

    # Test numpy array.
    actual = allele_counts_locate_het(ac)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    ac_dask = da.from_array(ac, chunks=(1, 1, -1))
    actual = allele_counts_locate_het(ac_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())
