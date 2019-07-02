import zarr
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import dask.dataframe as dd
import pytest
from skallel.model.functions import variants_to_dataframe


def setup_variants_numpy():
    variants = dict()
    variants["ID"] = np.array(["RS1", "RS3", "RS9", "RS11"])
    variants["POS"] = np.array([3, 6, 18, 42])
    variants["REF"] = np.array(["A", "C", "T", "G"])
    variants["ALT"] = np.array([["C", ""], ["T", "G"], ["G", ""], ["CAA", "T"]])
    variants["QUAL"] = np.array([3.4, 6.7, 18.1, 42.0])
    variants["FILTER_PASS"] = np.array([True, True, False, True])
    variants["DP"] = np.array([12, 23, 34, 45])
    variants["AC"] = np.array([[12, 23], [34, 45], [56, 67], [78, 89]])
    return variants


def setup_variants_zarr():
    variants_numpy = setup_variants_numpy()
    variants_zarr = zarr.group()
    for k, v in variants_numpy.items():
        variants_zarr.create_dataset(k, data=v)
    return variants_zarr


def test_variants_to_dataframe_index_pos():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # construct dataframes
    pdf = variants_to_dataframe(variants_numpy, index="POS")
    ddf = variants_to_dataframe(variants_zarr, index="POS")

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
        assert_array_equal(variants_numpy["POS"], np.asarray(df.index))
        assert_array_equal(variants_numpy["ID"], np.asarray(df["ID"]))
        assert_array_equal(variants_numpy["REF"], np.asarray(df["REF"]))
        assert_array_equal(variants_numpy["ALT"][:, 0], np.asarray(df["ALT_1"]))
        assert_array_equal(variants_numpy["ALT"][:, 1], np.asarray(df["ALT_2"]))
        assert_array_equal(variants_numpy["QUAL"], np.asarray(df["QUAL"]))
        assert_array_equal(variants_numpy["FILTER_PASS"], np.asarray(df["FILTER_PASS"]))
        assert_array_equal(variants_numpy["AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(variants_numpy["AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(variants_numpy["DP"], np.asarray(df["DP"]))


def test_variants_to_dataframe_index_id():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # construct dataframes
    pdf = variants_to_dataframe(variants_numpy, index="ID")
    ddf = variants_to_dataframe(variants_zarr, index="ID")

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

    assert_array_equal(variants_numpy["POS"], np.asarray(pdf["POS"]))
    assert_array_equal(variants_numpy["ID"], np.asarray(pdf.index))

    pdf_sorted = pdf.sort_index()
    assert_array_equal(np.asarray(pdf_sorted["POS"]), np.asarray(ddf["POS"]))
    assert_array_equal(np.asarray(pdf_sorted.index), np.asarray(ddf.index))


def test_variants_to_dataframe_noindex():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # construct dataframes
    pdf = variants_to_dataframe(variants_numpy)
    ddf = variants_to_dataframe(variants_zarr)

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
        assert_array_equal(variants_numpy["POS"], np.asarray(df["POS"]))


def test_variants_to_dataframe_columns_index_pos():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # construct dataframes
    cols = ["REF", "POS", "DP", "AC"]
    pdf = variants_to_dataframe(variants_numpy, columns=cols, index="POS")
    ddf = variants_to_dataframe(variants_zarr, columns=cols, index="POS")

    for df in pdf, ddf:

        # check column ordering, should be as requested
        assert ["REF", "DP", "AC_1", "AC_2"] == df.columns.tolist()

        # check POS used as index
        assert "POS" == df.index.name

        # check data
        assert_array_equal(variants_numpy["POS"], np.asarray(df.index))
        assert_array_equal(variants_numpy["REF"], np.asarray(df["REF"]))
        assert_array_equal(variants_numpy["AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(variants_numpy["AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(variants_numpy["DP"], np.asarray(df["DP"]))


def test_variants_to_dataframe_columns_noindex():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # construct dataframes
    cols = ["REF", "POS", "DP", "AC"]
    pdf = variants_to_dataframe(variants_numpy, columns=cols)
    ddf = variants_to_dataframe(variants_zarr, columns=cols)

    for df in pdf, ddf:

        # check column ordering, should be as requested
        assert ["REF", "POS", "DP", "AC_1", "AC_2"] == df.columns.tolist()

        # check POS not used as index
        assert None is df.index.name

        # check data
        assert_array_equal(variants_numpy["POS"], np.asarray(df["POS"]))
        assert_array_equal(variants_numpy["REF"], np.asarray(df["REF"]))
        assert_array_equal(variants_numpy["AC"][:, 0], np.asarray(df["AC_1"]))
        assert_array_equal(variants_numpy["AC"][:, 1], np.asarray(df["AC_2"]))
        assert_array_equal(variants_numpy["DP"], np.asarray(df["DP"]))


def test_variants_to_dataframe_exceptions():

    # setup
    variants_numpy = setup_variants_numpy()
    variants_zarr = setup_variants_zarr()

    # field not present in data
    with pytest.raises(ValueError):
        variants_to_dataframe(variants_numpy, columns=["foo"])
    with pytest.raises(ValueError):
        variants_to_dataframe(variants_zarr, columns=["foo"])

    # array has too many dimensions
    variants_numpy["bar"] = np.arange(1000).reshape((10, 10, 10))
    variants_zarr.create_dataset("bar", data=np.arange(1000).reshape((10, 10, 10)))
    with pytest.warns(UserWarning):
        variants_to_dataframe(variants_numpy)
    with pytest.warns(UserWarning):
        variants_to_dataframe(variants_zarr)
