import pytest
import numpy as np
from numpy.testing import assert_array_equal
from skallel.model.fn_numpy import genotype_array_check, genotype_array_is_called


def test_genotype_array_check():

    # valid data - diploid
    gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="i1")
    genotype_array_check(gt)

    # valid data (triploid)
    gt = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype="i1")
    genotype_array_check(gt)

    # bad type
    gt = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotype_array_check(gt)

    # bad dtype
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        with pytest.raises(TypeError):
            genotype_array_check(gt)

    # bad ndim
    gt = np.array([[0, 1], [2, 3]], dtype="i1")
    with pytest.raises(ValueError):
        genotype_array_check(gt)

    # bad ndim
    gt = np.array([0, 1], dtype="i1")
    with pytest.raises(ValueError):
        genotype_array_check(gt)


def test_genotype_array_is_called():
    gt = np.array([[[0, 0], [0, 1]], [[0, -1], [-1, -1]]], dtype="i1")
    expect = np.array([[True, True], [False, False]], dtype=bool)
    actual = genotype_array_is_called(gt)
    assert_array_equal(expect, actual)
