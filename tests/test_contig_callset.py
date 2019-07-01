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

    # construct dataframes
    pdf = callset.variants_to_dataframe()
    ddf = callset.variants_to_dask_dataframe()

    # check return types
    assert isinstance(pdf, pd.DataFrame)
    assert isinstance(ddf, dd.DataFrame)

    for df in pdf, ddf:

        # check default column ordering - expect VCF fixed fields first if present,
        # then other fields ordered alphabetically
        expected_cols = [
            "ID",
            "REF",
            "ALT_1",
            "ALT_2",
            "QUAL",
            "FILTER_PASS",
            "AC_1",
            "AC_2",
            "DP",
        ]
        assert expected_cols == df.columns.tolist()

        # check POS used as index
        assert "POS" == df.index.name

        # check data
        assert_array_equal(data["variants/POS"][:], np.asarray(df.index))
        assert_array_equal(data["variants/ID"][:], np.asarray(df["ID"]))
        assert_array_equal(data["variants/REF"][:], np.asarray(df["REF"]))
        assert_array_equal(data["variants/ALT"][:, 0], np.asarray(df["ALT_1"]))
        assert_array_equal(data["variants/ALT"][:, 1], np.asarray(df["ALT_2"]))
        assert_array_equal(data["variants/QUAL"][:], np.asarray(df["QUAL"]))
        assert_array_equal(
            data["variants/FILTER_PASS"][:], np.asarray(df["FILTER_PASS"])
        )
        assert_array_equal(data["variants/AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(data["variants/AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(data["variants/DP"][:], np.asarray(df["DP"]))


def test_variants_to_dataframe_index():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframes
    pdf = callset.variants_to_dataframe(index="ID")
    ddf = callset.variants_to_dask_dataframe(index="ID")

    for df in pdf, ddf:

        # check default column ordering - expect VCF fixed fields first if present,
        # then other fields ordered alphabetically
        expected_cols = [
            "POS",
            "REF",
            "ALT_1",
            "ALT_2",
            "QUAL",
            "FILTER_PASS",
            "AC_1",
            "AC_2",
            "DP",
        ]
        assert expected_cols == df.columns.tolist()

        # check ID used as index
        assert "ID" == df.index.name

    # N.B., different behaviour here between pandas and dask, because dask forces
    # the dataframe to be sorted by the index, whereas pandas does not.

    assert_array_equal(data["variants/POS"][:], np.asarray(pdf["POS"]))
    assert_array_equal(data["variants/ID"][:], np.asarray(pdf.index))

    pdf_sorted = pdf.sort_index()
    assert_array_equal(np.asarray(pdf_sorted["POS"]), np.asarray(ddf["POS"]))
    assert_array_equal(np.asarray(pdf_sorted.index), np.asarray(ddf.index))


def test_variants_to_dataframe_noindex():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframes
    pdf = callset.variants_to_dataframe(index=None)
    ddf = callset.variants_to_dask_dataframe(index=None)

    for df in pdf, ddf:

        expected_cols = [
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
        ]
        assert expected_cols == df.columns.tolist()

        # check POS used as index
        assert None is df.index.name

        # check data
        assert_array_equal(data["variants/POS"][:], np.asarray(df["POS"]))


def test_variants_to_dataframe_columns():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframes
    cols = ["REF", "POS", "DP", "AC"]
    pdf = callset.variants_to_dataframe(columns=cols)
    ddf = callset.variants_to_dask_dataframe(columns=cols)

    for df in pdf, ddf:

        # check column ordering, should be as requested
        assert ["REF", "DP", "AC_1", "AC_2"] == df.columns.tolist()

        # check POS used as index
        assert "POS" == df.index.name

        # check data
        assert_array_equal(data["variants/POS"][:], np.asarray(df.index))
        assert_array_equal(data["variants/REF"][:], np.asarray(df["REF"]))
        assert_array_equal(data["variants/AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(data["variants/AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(data["variants/DP"][:], np.asarray(df["DP"]))


def test_variants_to_dataframe_columns_noindex():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # construct dataframes
    cols = ["REF", "POS", "DP", "AC"]
    pdf = callset.variants_to_dataframe(columns=cols, index=None)
    ddf = callset.variants_to_dask_dataframe(columns=cols, index=None)

    for df in pdf, ddf:

        # check column ordering, should be as requested
        assert ["REF", "POS", "DP", "AC_1", "AC_2"] == df.columns.tolist()

        # check POS not used as index
        assert None is df.index.name

        # check data
        assert_array_equal(data["variants/POS"][:], np.asarray(df["POS"]))
        assert_array_equal(data["variants/REF"][:], np.asarray(df["REF"]))
        assert_array_equal(data["variants/AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(data["variants/AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(data["variants/DP"][:], np.asarray(df["DP"]))


def test_variants_to_dataframe_exceptions():

    # setup
    data = setup_callset_data_zarr()
    callset = ContigCallset(data)

    # field not present in data
    with pytest.raises(ValueError):
        callset.variants_to_dataframe(columns=["foo"])
    with pytest.raises(ValueError):
        callset.variants_to_dask_dataframe(columns=["foo"])

    # array has too many dimensions
    data.create_dataset("variants/bar", data=np.arange(1000).reshape((10, 10, 10)))
    with pytest.warns(UserWarning):
        callset.variants_to_dataframe()
    with pytest.warns(UserWarning):
        callset.variants_to_dask_dataframe()
