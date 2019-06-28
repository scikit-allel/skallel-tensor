import zarr
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import dask.dataframe as dd
import pytest
from skallel.model.oo import ContigCallset


def setup_callset_data_zarr():
    data = zarr.group()
    data.create_dataset("variants/ID", data=np.array(["RS1", "RS3", "RS9", "RS11"]))
    data.create_dataset("variants/POS", data=np.array([3, 6, 18, 42]))
    data.create_dataset("variants/REF", data=np.array(["A", "C", "T", "G"]))
    data.create_dataset(
        "variants/ALT", data=np.array([["C", ""], ["T", "G"], ["G", ""], ["CAA", "T"]])
    )
    data.create_dataset("variants/QUAL", data=np.array([3.4, 6.7, 18.1, 42.0]))
    data.create_dataset(
        "variants/FILTER_PASS", data=np.array([True, True, False, True])
    )
    data.create_dataset("variants/DP", data=np.array([12, 23, 34, 45]))
    data.create_dataset(
        "variants/AC", data=np.array([[12, 23], [34, 45], [56, 67], [78, 89]])
    )
    return data


def test_init():

    # test zarr data
    data = setup_callset_data_zarr()

    # instantiate
    callset = ContigCallset(data)

    # test properties
    assert data is callset.data


def test_variants_to_dataframe_default():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframe
    df = callset.variants_to_dataframe()
    assert isinstance(df, pd.DataFrame)

    # check default column ordering - expect VCF fixed fields first if present,
    # then other fields ordered alphabetically
    assert [
        "ID",
        "REF",
        "ALT_1",
        "ALT_2",
        "QUAL",
        "FILTER_PASS",
        "AC_1",
        "AC_2",
        "DP",
    ] == df.columns.tolist()

    # check POS used as index
    assert "POS" == df.index.name

    # check data
    assert_array_equal(data["variants/POS"][:], df.index.values)
    assert_array_equal(data["variants/ID"][:], df["ID"].values)
    assert_array_equal(data["variants/REF"][:], df["REF"].values)
    assert_array_equal(data["variants/ALT"][:, 0], df["ALT_1"].values)
    assert_array_equal(data["variants/ALT"][:, 1], df["ALT_2"].values)
    assert_array_equal(data["variants/QUAL"][:], df["QUAL"].values)
    assert_array_equal(data["variants/FILTER_PASS"][:], df["FILTER_PASS"].values)
    assert_array_equal(data["variants/AC"][:, 0], df["AC_1"].values)
    assert_array_equal(data["variants/AC"][:, 1], df["AC_2"].values)
    assert_array_equal(data["variants/DP"][:], df["DP"].values)


def test_variants_to_dataframe_noindex():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframe
    df = callset.variants_to_dataframe(index=None)
    assert isinstance(df, pd.DataFrame)

    assert [
        "POS",
        "ID",
        "REF",
        "ALT_1",
        "ALT_2",
        "QUAL",
        "FILTER_PASS",
        "AC_1",
        "AC_2",
        "DP",
    ] == df.columns.tolist()

    # check POS used as index
    assert None is df.index.name

    # check data
    assert_array_equal(data["variants/POS"][:], df["POS"].values)


def test_variants_to_dataframe_columns():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframe
    df = callset.variants_to_dataframe(columns=["REF", "POS", "DP", "AC"])
    assert isinstance(df, pd.DataFrame)

    # check default column ordering - expect VCF fixed fields first if present,
    # then other fields ordered alphabetically
    assert ["REF", "DP", "AC_1", "AC_2"] == df.columns.tolist()

    # check POS used as index
    assert "POS" == df.index.name

    # check data
    assert_array_equal(data["variants/POS"][:], df.index.values)
    assert_array_equal(data["variants/REF"][:], df["REF"].values)
    assert_array_equal(data["variants/AC"][:, 0], df["AC_1"].values)
    assert_array_equal(data["variants/AC"][:, 1], df["AC_2"].values)
    assert_array_equal(data["variants/DP"][:], df["DP"].values)


def test_variants_to_dataframe_columns_noindex():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframe
    df = callset.variants_to_dataframe(columns=["REF", "POS", "DP", "AC"], index=None)
    assert isinstance(df, pd.DataFrame)

    # check default column ordering - expect VCF fixed fields first if present,
    # then other fields ordered alphabetically
    assert ["REF", "POS", "DP", "AC_1", "AC_2"] == df.columns.tolist()

    # check POS used as index
    assert None is df.index.name

    # check data
    assert_array_equal(data["variants/POS"][:], df["POS"].values)
    assert_array_equal(data["variants/REF"][:], df["REF"].values)
    assert_array_equal(data["variants/AC"][:, 0], df["AC_1"].values)
    assert_array_equal(data["variants/AC"][:, 1], df["AC_2"].values)
    assert_array_equal(data["variants/DP"][:], df["DP"].values)


def test_variants_to_dataframe_exceptions():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # field not present in data
    with pytest.raises(ValueError):
        callset.variants_to_dataframe(columns=["foo"])

    # array has too many dimensions
    data.create_dataset("variants/bar", data=np.arange(1000).reshape(10, 10, 10))
    with pytest.warns(UserWarning):
        callset.variants_to_dataframe()


def test_variants_to_dask_dataframe_default():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframe
    df = callset.variants_to_dask_dataframe()
    assert isinstance(df, dd.DataFrame)

    # check default column ordering - expect VCF fixed fields first if present,
    # then other fields ordered alphabetically
    assert [
        "POS",
        "ID",
        "REF",
        "ALT_1",
        "ALT_2",
        "QUAL",
        "FILTER_PASS",
        "AC_1",
        "AC_2",
        "DP",
    ] == df.columns.tolist()

    # check data
    assert_array_equal(data["variants/POS"][:], df["POS"].compute())
    assert_array_equal(data["variants/ID"][:], df["ID"].compute())
    assert_array_equal(data["variants/REF"][:], df["REF"].compute())
    assert_array_equal(data["variants/ALT"][:, 0], df["ALT_1"].compute())
    assert_array_equal(data["variants/ALT"][:, 1], df["ALT_2"].compute())
    assert_array_equal(data["variants/QUAL"][:], df["QUAL"].compute())
    assert_array_equal(data["variants/FILTER_PASS"][:], df["FILTER_PASS"].compute())
    assert_array_equal(data["variants/AC"][:, 0], df["AC_1"].compute())
    assert_array_equal(data["variants/AC"][:, 1], df["AC_2"].compute())
    assert_array_equal(data["variants/DP"][:], df["DP"].compute())
