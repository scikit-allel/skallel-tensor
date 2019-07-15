import numpy as np
from numpy.testing import assert_array_equal
import pytest
import dask.array as da
import zarr


from skallel_tensor.utils import DictGroup
from skallel_tensor.api import GroupSelection, GroupConcatenation
from skallel_tensor import (
    select_slice,
    select_indices,
    select_mask,
    select_range,
    select_values,
    concatenate,
)


def test_select_slice():

    # Setup.
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))

    # Numpy array.
    for a in pos, gt:
        expect = a[10:20:2]
        actual = select_slice(a, start=10, stop=20, step=2, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:
        expect = a[10:20:2]
        d = da.from_array(a)
        actual = select_slice(d, start=10, stop=20, step=2, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = select_slice(g, start=10, stop=20, step=2, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos[10:20:2], actual["variants"]["POS"])
    assert_array_equal(gt[10:20:2], actual["calldata"]["GT"])

    # Check mapping methods work.
    assert len(g) == len(actual)
    assert sorted(g) == sorted(actual)
    assert sorted(g.keys()) == sorted(actual.keys())
    for v in actual.values():
        assert isinstance(v, (np.ndarray, GroupSelection))
    for k, v in actual.items():
        assert k in g
        assert isinstance(v, (np.ndarray, GroupSelection))
    for k in g:
        assert k in actual

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_slice(g, start=10, stop=20, step=2, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos[10:20:2], actual["variants"]["POS"].compute())
    assert_array_equal(gt[10:20:2], actual["calldata"]["GT"].compute())


def test_select_indices():

    # Setup.
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    indices = np.arange(1, 99, 3)

    # Numpy array.
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        actual = select_indices(a, indices, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:
        expect = a.take(indices, axis=0)
        d = da.from_array(a)
        actual = select_indices(d, indices, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = select_indices(g, indices, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos.take(indices, axis=0), actual["variants"]["POS"])
    assert_array_equal(gt.take(indices, axis=0), actual["calldata"]["GT"])

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_indices(g, indices, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(
        pos.take(indices, axis=0), actual["variants"]["POS"].compute()
    )
    assert_array_equal(
        gt.take(indices, axis=0), actual["calldata"]["GT"].compute()
    )


def test_select_mask():

    # Setup.
    pos = np.arange(100)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    mask = np.zeros(100, dtype=bool)
    mask[1:99:3] = True

    # Numpy array.
    for a in pos, gt:
        expect = a.compress(mask, axis=0)
        actual = select_mask(a, mask, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:
        expect = a.compress(mask, axis=0)
        d = da.from_array(a)
        actual = select_mask(d, mask, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())
        # With mask as dask array.
        actual = select_mask(d, da.from_array(mask), axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = select_mask(g, mask, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos.compress(mask, axis=0), actual["variants"]["POS"])
    assert_array_equal(gt.compress(mask, axis=0), actual["calldata"]["GT"])

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_mask(g, mask, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(
        pos.compress(mask, axis=0), actual["variants"]["POS"].compute()
    )
    assert_array_equal(
        gt.compress(mask, axis=0), actual["calldata"]["GT"].compute()
    )


def test_select_range():

    # Setup.
    pos = np.arange(1, 300, 3)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))

    # Numpy array.
    for a in pos, gt:
        expect = a[10:20]
        actual = select_range(a, pos, begin=30, end=60, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:
        expect = a[10:20]
        d = da.from_array(a)
        actual = select_range(d, pos, begin=30, end=60, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = select_range(g, "variants/POS", begin=30, end=60, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos[10:20], actual["variants"]["POS"])
    assert_array_equal(gt[10:20], actual["calldata"]["GT"])

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_range(g, "variants/POS", begin=30, end=60, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos[10:20], actual["variants"]["POS"].compute())
    assert_array_equal(gt[10:20], actual["calldata"]["GT"].compute())


def test_select_values():

    # Setup.
    pos = np.arange(1, 300, 3)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))
    query = [31, 61]

    # Numpy array.
    for a in pos, gt:
        expect = a[[10, 20]]
        actual = select_values(a, pos, query, axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:
        expect = a[[10, 20]]
        d = da.from_array(a)
        actual = select_values(d, pos, query, axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = select_values(g, "variants/POS", query, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(pos[[10, 20]], actual["variants"]["POS"])
    assert_array_equal(gt[[10, 20]], actual["calldata"]["GT"])

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = select_values(g, "variants/POS", query, axis=0)
    assert isinstance(actual, GroupSelection)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(pos[[10, 20]], actual["variants"]["POS"].compute())
    assert_array_equal(gt[[10, 20]], actual["calldata"]["GT"].compute())

    # Errors.
    with pytest.raises(KeyError):
        select_values(gt, pos, query=[1, 999], axis=0)


def test_concatenate():

    # Setup.
    pos = np.arange(1, 300, 3)
    gt = np.random.randint(low=-1, high=4, size=(100, 10))

    # Numpy array.
    for a in pos, gt:

        # Concatenate dim0.
        expect = np.concatenate([a, a], axis=0)
        actual = concatenate([a, a], axis=0)
        assert isinstance(actual, np.ndarray)
        assert_array_equal(expect, actual)

    # Concatenate dim1.
    expect = np.concatenate([gt, gt], axis=1)
    actual = concatenate([gt, gt], axis=1)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(expect, actual)

    # Dask array.
    for a in pos, gt:

        # Concatenate dim0.
        expect = np.concatenate([a, a], axis=0)
        d = da.from_array(a)
        actual = concatenate([d, d], axis=0)
        assert isinstance(actual, da.Array)
        assert_array_equal(expect, actual.compute())

    # Concatenate dim1.
    expect = np.concatenate([gt, gt], axis=1)
    d = da.from_array(gt)
    actual = concatenate([d, d], axis=1)
    assert isinstance(actual, da.Array)
    assert_array_equal(expect, actual)

    # Numpy group.
    g = DictGroup({"variants": {"POS": pos}, "calldata": {"GT": gt}})
    actual = concatenate([g, g], axis=0)
    assert isinstance(actual, GroupConcatenation)
    assert isinstance(actual["variants"]["POS"], np.ndarray)
    assert isinstance(actual["calldata"]["GT"], np.ndarray)
    assert_array_equal(
        np.concatenate([pos, pos], axis=0), actual["variants"]["POS"]
    )
    assert_array_equal(
        np.concatenate([gt, gt], axis=0), actual["calldata"]["GT"]
    )

    # Check mapping methods work.
    assert len(g) == len(actual)
    assert sorted(g) == sorted(actual)
    assert sorted(g.keys()) == sorted(actual.keys())
    for v in actual.values():
        assert isinstance(v, (np.ndarray, GroupConcatenation))
    for k, v in actual.items():
        assert k in g
        assert isinstance(v, (np.ndarray, GroupConcatenation))
    for k in g:
        assert k in actual

    # Zarr group.
    g = zarr.group()
    g.create_dataset("variants/POS", data=pos)
    g.create_dataset("calldata/GT", data=gt)
    actual = concatenate([g, g], axis=0)
    assert isinstance(actual["variants"]["POS"], da.Array)
    assert isinstance(actual["calldata"]["GT"], da.Array)
    assert_array_equal(
        np.concatenate([pos, pos], axis=0), actual["variants"]["POS"].compute()
    )
    assert_array_equal(
        np.concatenate([gt, gt], axis=0), actual["calldata"]["GT"].compute()
    )

    # Test errors.
    with pytest.raises(TypeError):
        concatenate({"gt": gt}, axis=0)
    with pytest.raises(ValueError):
        concatenate([gt], axis=0)
    with pytest.raises(NotImplementedError):
        x = [1, 2, 3, 4]
        concatenate([x, x], axis=0)
