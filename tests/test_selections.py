import numpy as np
from numpy.testing import assert_array_equal
import dask.array as da
import zarr


from skallel.model.functions import Selection, select_slice, select_indices, select_mask


def test_select_slice():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))

    # numpy array
    for a in pos, gt:
        expect = a[10:20:2]
        actual = select_slice(a, 10, 20, 2, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a[10:20:2]
        d = da.from_array(a)
        actual = select_slice(d, 10, 20, 2, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants": {"POS": pos}, "calldata": {"GT": gt}}
    actual = select_slice(g, 10, 20, 2, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos[10:20:2], actual["variants"]["POS"])
    assert_array_equal(gt[10:20:2], actual["calldata"]["GT"])

    # check mapping methods work
    assert len(g) == len(actual)
    assert sorted(g) == sorted(actual)
    assert sorted(g.keys()) == sorted(actual.keys())
    for v in actual.values():
        assert isinstance(v, (np.ndarray, Selection))
    for k, v in actual.items():
        assert k in g
        assert isinstance(v, (np.ndarray, Selection))
    for k in g:
        assert k in actual

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_slice(g, 10, 20, 2, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos[10:20:2], actual["variants"]["POS"].compute())
    assert_array_equal(gt[10:20:2], actual["calldata"]["GT"].compute())


def test_select_indices():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    indices = np.arange(1, 99, 3)

    # numpy array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        actual = select_indices(a, indices, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        d = da.from_array(a)
        actual = select_indices(d, indices, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants": {"POS": pos}, "calldata": {"GT": gt}}
    actual = select_indices(g, indices, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos.take(indices, axis=0), actual["variants"]["POS"])
    assert_array_equal(gt.take(indices, axis=0), actual["calldata"]["GT"])

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_indices(g, indices, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos.take(indices, axis=0), actual["variants"]["POS"].compute())
    assert_array_equal(gt.take(indices, axis=0), actual["calldata"]["GT"].compute())


def test_select_mask():

    # setup
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    mask = np.zeros(100, dtype=bool)
    mask[1:99:3] = True

    # numpy array
    for a in pos, gt:
        expect = a.compress(mask, axis=0)
        actual = select_mask(a, mask, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # dask array
    for a in pos, gt:
        expect = a.compress(mask, axis=0)
        d = da.from_array(a)
        actual = select_mask(d, mask, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # numpy group
    g = {"variants": {"POS": pos}, "calldata": {"GT": gt}}
    actual = select_mask(g, mask, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos.compress(mask, axis=0), actual["variants"]["POS"])
    assert_array_equal(gt.compress(mask, axis=0), actual["calldata"]["GT"])

    # zarr group
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_mask(g, mask, axis=0)
    assert isinstance(actual, Selection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos.compress(mask, axis=0), actual["variants"]["POS"].compute())
    assert_array_equal(gt.compress(mask, axis=0), actual["calldata"]["GT"].compute())
