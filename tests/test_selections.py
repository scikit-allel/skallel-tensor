import numpy as np
from numpy.testing import assert_array_equal
import dask.array as da
import zarr


from skallel.model.functions import Selection, slice_variants, take, take_variants


def test_slice_variants():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))

    # numpy array
    for a in pos, gt:
        expect = a[10:20:2]
        actual = slice_variants(a, 10, 20, 2)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a[10:20:2]
        d = da.from_array(a)
        actual = slice_variants(d, 10, 20, 2)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants/POS": pos, "calldata/GT": gt}
    actual = slice_variants(g, 10, 20, 2)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants/POS"], np.ndarray)
    assert isinstance(actual["calldata/GT"], np.ndarray)
    assert_array_equal(pos[10:20:2], actual["variants/POS"])
    assert_array_equal(gt[10:20:2], actual["calldata/GT"])

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)

    # test on non-nested groups
    actual = slice_variants(g["variants"], 10, 20, 2)
    assert isinstance(actual, Selection)
    assert isinstance(actual["POS"], da.Array)
    assert_array_equal(pos[10:20:2], actual["POS"].compute())
    actual = slice_variants(g["calldata"], 10, 20, 2)
    assert isinstance(actual, Selection)
    assert isinstance(actual["GT"], da.Array)
    assert_array_equal(gt[10:20:2], actual["GT"].compute())

    # test on nested groups
    actual = slice_variants(g, 10, 20, 2)
    assert isinstance(actual, Selection)
    # access via full path
    assert isinstance(actual["variants/POS"], da.Array)
    assert isinstance(actual["calldata/GT"], da.Array)
    assert_array_equal(pos[10:20:2], actual["variants/POS"].compute())
    assert_array_equal(gt[10:20:2], actual["calldata/GT"].compute())
    # access as nested groups
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos[10:20:2], actual["variants"]["POS"].compute())
    assert_array_equal(gt[10:20:2], actual["calldata"]["GT"].compute())


def test_take():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    indices = np.arange(1, 99, 3)

    # numpy array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        actual = take(a, indices, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        d = da.from_array(a)
        actual = take(d, indices, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants/POS": pos, "calldata/GT": gt}
    actual = take(g, indices, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants/POS"], np.ndarray)
    assert isinstance(actual["calldata/GT"], np.ndarray)
    assert_array_equal(pos.take(indices, axis=0), actual["variants/POS"])
    assert_array_equal(gt.take(indices, axis=0), actual["calldata/GT"])

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)

    # test on non-nested groups
    actual = take(g["variants"], indices, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["POS"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["POS"].compute())
    actual = take(g["calldata"], indices, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["GT"], da.Array)
    assert_array_equal(gt.take(indices, axis=0), actual["GT"].compute())

    # test on nested groups
    actual = take_variants(g, indices)
    assert isinstance(actual, Selection)
    # access via full path
    assert isinstance(actual["variants/POS"], da.Array)
    assert isinstance(actual["calldata/GT"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["variants/POS"].compute())
    assert_array_equal(gt.take(indices, axis=0), actual["calldata/GT"].compute())
    # access as nested groups
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["variants"]["POS"].compute())
    assert_array_equal(gt.take(indices, axis=0), actual["calldata"]["GT"].compute())


def test_take_variants():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    indices = np.arange(1, 99, 3)

    # numpy array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        actual = take_variants(a, indices)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        d = da.from_array(a)
        actual = take_variants(d, indices)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants/POS": pos, "calldata/GT": gt}
    actual = take_variants(g, indices)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants/POS"], np.ndarray)
    assert isinstance(actual["calldata/GT"], np.ndarray)
    assert_array_equal(pos.take(indices, axis=0), actual["variants/POS"])
    assert_array_equal(gt.take(indices, axis=0), actual["calldata/GT"])

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)

    # test on non-nested groups
    actual = take_variants(g["variants"], indices)
    assert isinstance(actual, Selection)
    assert isinstance(actual["POS"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["POS"].compute())
    actual = take_variants(g["calldata"], indices)
    assert isinstance(actual, Selection)
    assert isinstance(actual["GT"], da.Array)
    assert_array_equal(gt.take(indices, axis=0), actual["GT"].compute())

    # test on nested groups
    actual = take_variants(g, indices)
    assert isinstance(actual, Selection)
    # access via full path
    assert isinstance(actual["variants/POS"], da.Array)
    assert isinstance(actual["calldata/GT"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["variants/POS"].compute())
    assert_array_equal(gt.take(indices, axis=0), actual["calldata/GT"].compute())
    # access as nested groups
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["variants"]["POS"].compute())
    assert_array_equal(gt.take(indices, axis=0), actual["calldata"]["GT"].compute())
