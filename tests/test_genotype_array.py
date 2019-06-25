import numpy as np
import dask.array as da
import pytest
from numpy.testing import assert_array_equal


from skallel.model.oo import GenotypeArray


def test_init():

    # valid data - numpy array
    data = np.array([[[0, 1], [2, 3], [4, 5]], [[4, 5], [6, 7], [-1, -1]]], dtype="i1")
    gt = GenotypeArray(data)
    assert data is gt.data
    assert data is gt.values
    assert 2 == gt.n_variants
    assert 3 == gt.n_samples
    assert 2 == gt.ploidy

    # valid data - dask array
    data_dask = da.from_array(data, chunks=(1, 1, 2))
    gt = GenotypeArray(data_dask)
    assert data_dask is gt.data
    assert data_dask is gt.values
    assert 2 == gt.n_variants
    assert 3 == gt.n_samples
    assert 2 == gt.ploidy

    # valid data (triploid)
    data_triploid = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[-1, -1, -1], [12, 13, 14]],
        ],
        dtype="i1",
    )
    gt = GenotypeArray(data_triploid)
    assert data_triploid is gt.data
    assert data_triploid is gt.values
    assert 3 == gt.n_variants
    assert 2 == gt.n_samples
    assert 3 == gt.ploidy

    # bad type
    data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        GenotypeArray(data)

    # bad dtype
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        with pytest.raises(TypeError):
            GenotypeArray(data)

    # bad ndim
    data = np.array([[0, 1], [2, 3]], dtype="i1")
    with pytest.raises(ValueError):
        GenotypeArray(data)

    # bad ndim
    data = np.array([0, 1], dtype="i1")
    with pytest.raises(ValueError):
        GenotypeArray(data)


def test_is_called():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, True, True], [False, False, False]], dtype=bool)

    # test numpy array
    gt = GenotypeArray(data)
    actual = gt.is_called()
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    gt = GenotypeArray(data_dask)
    actual = gt.is_called().compute()
    assert_array_equal(expect, actual)


def test_is_missing():

    data = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[False, False, False], [True, True, True]], dtype=bool)

    # test numpy array
    gt = GenotypeArray(data)
    actual = gt.is_missing()
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    gt = GenotypeArray(data_dask)
    actual = gt.is_missing().compute()
    assert_array_equal(expect, actual)


def test_is_hom():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)

    # test numpy array
    gt = GenotypeArray(data)
    actual = gt.is_hom()
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    gt = GenotypeArray(data_dask)
    actual = gt.is_hom().compute()
    assert_array_equal(expect, actual)


def test_count_alleles():

    data = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    expect = np.array([[3, 1, 2], [2, 0, 0]], dtype="i4")

    # test numpy array
    gt = GenotypeArray(data)
    actual = gt.count_alleles(max_allele=2)
    assert_array_equal(expect, actual)

    # test dask array
    data_dask = da.from_array(data, chunks=(1, 1, -1))
    gt = GenotypeArray(data_dask)
    actual = gt.count_alleles(max_allele=2).compute()
    assert_array_equal(expect, actual)
