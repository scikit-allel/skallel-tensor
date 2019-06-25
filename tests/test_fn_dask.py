import pytest
import numpy as np
import dask.array as da
from skallel.model.fn_dask import genotype_array_check


def test_genotype_array_check():

    # valid data - diploid
    gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="i1")
    gt = da.from_array(gt)
    genotype_array_check(gt)

    # valid data (triploid)
    gt = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype="i1")
    gt = da.from_array(gt)
    genotype_array_check(gt)

    # bad type
    gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="i1")
    with pytest.raises(TypeError):
        genotype_array_check(gt)

    # bad type
    gt = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with pytest.raises(TypeError):
        genotype_array_check(gt)

    # bad dtype
    for dtype in "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8":
        gt = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=dtype)
        gt = da.from_array(gt)
        with pytest.raises(TypeError):
            genotype_array_check(gt)

    # bad ndim
    gt = np.array([[0, 1], [2, 3]], dtype="i1")
    gt = da.from_array(gt)
    with pytest.raises(ValueError):
        genotype_array_check(gt)

    # bad ndim
    gt = np.array([0, 1], dtype="i1")
    gt = da.from_array(gt)
    with pytest.raises(ValueError):
        genotype_array_check(gt)
