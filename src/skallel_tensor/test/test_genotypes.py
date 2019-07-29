import numpy as np
import dask.array as da
from numpy.testing import assert_array_equal
import pytest
import zarr
from numba import cuda


from skallel_tensor import (
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_called_allele_counts,
    genotypes_to_missing_allele_counts,
    genotypes_to_major_allele_counts,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
    genotypes_to_haplotypes,
)


def _test_gt_func(f, gt, expect, compare, **kwargs):

    # 3D tests.
    assert gt.ndim == 3

    # Test numpy array.
    actual = f(gt, **kwargs)
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)

    # Test numpy array, Fortran order.
    actual = f(np.asfortranarray(gt), **kwargs)
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 2, -1))
    actual = f(gt_dask, **kwargs)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    gt_zarr = zarr.array(data=gt, chunks=(1, 2, None))
    actual = f(gt_zarr, **kwargs)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Reshape to test as 2D.
    gt = gt.reshape((-1, gt.shape[2]))
    if expect.ndim == 3:
        expect = expect.reshape((gt.shape[0], -1))
    elif expect.ndim == 2:
        expect = expect.reshape(-1)

    # Test numpy array.
    actual = f(gt, **kwargs)
    assert isinstance(actual, np.ndarray)
    compare(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(2, -1))
    actual = f(gt_dask, **kwargs)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    gt_zarr = zarr.array(data=gt)
    actual = f(gt_zarr, **kwargs)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # Wrong type.
        f("foo", **kwargs)
    with pytest.raises(TypeError):
        # Wrong dtype.
        f(gt.astype("f4"), **kwargs)
    with pytest.raises(ValueError):
        # Wrong ndim.
        f(gt[0], **kwargs)


def test_locate_hom():
    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[True, False, True], [False, False, False]], dtype=bool)
    _test_gt_func(genotypes_locate_hom, gt, expect, compare=assert_array_equal)


def test_locate_het():
    gt = np.array(
        [[[0, 0], [0, 1], [1, 2]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[False, True, True], [False, False, False]], dtype=bool)
    _test_gt_func(genotypes_locate_het, gt, expect, compare=assert_array_equal)


def test_locate_het_triploid():
    gt = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, -1], [0, 1, -1], [0, -1, -1]],
        ],
        dtype=np.int8,
    )
    expect = np.array([[False, True, True], [False, True, False]], dtype=bool)
    _test_gt_func(genotypes_locate_het, gt, expect, compare=assert_array_equal)


def test_locate_call():
    gt = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[False, False, True], [False, False, False]], dtype=bool)
    for call in (1, 0), [1, 0], np.array([1, 0]):
        _test_gt_func(
            genotypes_locate_call,
            gt,
            expect,
            compare=assert_array_equal,
            call=call,
        )

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

    # Test cuda array.
    gt_cuda = cuda.to_device(gt)
    actual = genotypes_count_alleles(gt_cuda, max_allele=2)
    assert isinstance(actual, type(gt_cuda))
    assert_array_equal(expect, actual.copy_to_host())

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 2, -1))
    actual = genotypes_count_alleles(gt_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    gt_zarr = zarr.array(gt, chunks=(1, 2, None))
    actual = genotypes_count_alleles(gt_zarr, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test dask cuda array.
    gt_dask_cuda = gt_dask.map_blocks(cuda.to_device)
    actual = genotypes_count_alleles(gt_dask_cuda, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute(scheduler="single-threaded"))

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
    _test_gt_func(
        genotypes_to_called_allele_counts,
        gt,
        expect,
        compare=assert_array_equal,
    )


def test_to_missing_allele_counts():
    gt = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 0], [0, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array([[0, 0, 0], [1, 1, 2]], dtype=np.int8)
    _test_gt_func(
        genotypes_to_missing_allele_counts,
        gt,
        expect,
        compare=assert_array_equal,
    )


def test_to_allele_counts():
    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array(
        [[[2, 0, 0], [1, 1, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]],
        dtype="i4",
    )
    _test_gt_func(
        genotypes_to_allele_counts,
        gt,
        expect,
        compare=assert_array_equal,
        max_allele=2,
    )

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
    gt_dask = da.from_array(gt, chunks=(1, 2, -1))
    actual = genotypes_to_allele_counts_melt(gt_dask, max_allele=2)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_to_allele_counts_melt(gt, max_allele="foo")
    with pytest.raises(ValueError):
        genotypes_to_allele_counts_melt(gt, max_allele=128)


def test_to_major_allele_counts():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 3]], [[-1, 1], [1, -1], [-1, -1]]], dtype=np.int8
    )

    expect = np.array([[2, 1, 0], [1, 1, 0]], dtype=np.int8)

    # Test numpy array.
    actual = genotypes_to_major_allele_counts(gt, max_allele=3)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 2, -1))
    actual = genotypes_to_major_allele_counts(gt_dask, max_allele=3)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        genotypes_to_major_allele_counts(gt, max_allele="foo")
    with pytest.raises(ValueError):
        genotypes_to_major_allele_counts(gt, max_allele=128)


def test_to_haplotypes():

    gt = np.array(
        [[[0, 0], [0, 1], [2, 2]], [[-1, 0], [1, -1], [-1, -1]]], dtype=np.int8
    )
    expect = np.array(
        [[0, 0, 0, 1, 2, 2], [-1, 0, 1, -1, -1, -1]], dtype=np.int8
    )

    # Test numpy array.
    actual = genotypes_to_haplotypes(gt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test numpy array, F order.
    actual = genotypes_to_haplotypes(np.asfortranarray(gt))
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Test dask array.
    gt_dask = da.from_array(gt, chunks=(1, 2, -1))
    actual = genotypes_to_haplotypes(gt_dask)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test zarr array.
    gt_zarr = zarr.array(gt, chunks=(1, 2, 2))
    actual = genotypes_to_haplotypes(gt_zarr)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual.compute())

    # Test exceptions.
    with pytest.raises(TypeError):
        # Wrong type.
        genotypes_to_haplotypes("foo")
    with pytest.raises(TypeError):
        # Wrong dtype.
        genotypes_to_haplotypes(gt.astype("f4"))
    with pytest.raises(ValueError):
        # Wrong ndim.
        genotypes_to_haplotypes(gt[0])
