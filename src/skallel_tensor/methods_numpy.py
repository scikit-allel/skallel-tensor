import warnings
import numpy as np
from numpy import take, compress, concatenate  # noqa
import numba
import pandas as pd


def accepts(a):
    if isinstance(a, np.ndarray):
        return True


def getitem(a, item):
    return a[item]


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_tensor_is_called(gt):
    out = np.ones(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_tensor_is_missing(gt):
    out = np.zeros(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = True
                    # No need to check other alleles.
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_tensor_is_hom(gt):
    out = np.ones(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            first_allele = gt[i, j, 0]
            if first_allele < 0:
                out[i, j] = False
            else:
                for k in range(1, gt.shape[2]):
                    if gt[i, j, k] != first_allele:
                        out[i, j] = False
                        # No need to check other alleles.
                        break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_tensor_is_het(gt):
    out = np.zeros(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            first_allele = gt[i, j, 0]
            if first_allele >= 0:
                for k in range(1, gt.shape[2]):
                    allele = gt[i, j, k]
                    if allele >= 0 and allele != first_allele:
                        out[i, j] = True
                        # No need to check other alleles.
                        break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :], numba.int8[:]), nogil=True)
def genotype_tensor_is_call(gt, call):
    out = np.ones(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] != call[k]:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


@numba.njit(numba.int32[:, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotype_tensor_count_alleles(gt, max_allele):
    out = np.zeros((gt.shape[0], max_allele + 1), dtype=np.int32)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
    return out


@numba.njit(numba.int8[:, :, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotype_tensor_to_allele_counts(gt, max_allele):
    out = np.zeros((gt.shape[0], gt.shape[1], max_allele + 1), dtype=np.int8)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, j, allele] += 1
    return out


@numba.njit(numba.int8[:, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotype_tensor_to_allele_counts_melt(gt, max_allele):
    out = np.zeros((gt.shape[0] * (max_allele + 1), gt.shape[1]), dtype=np.int8)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[(i * (max_allele + 1)) + allele, j] += 1
    return out


def variants_to_dataframe(variants, columns):

    # Build dataframe.
    df_cols = {}
    for c in columns:

        # Obtain values.
        a = variants[c]

        # Ensure numpy array.
        a = np.asarray(a)

        # Check number of dimensions.
        if a.ndim == 1:
            df_cols[c] = a
        elif a.ndim == 2:
            # Split columns.
            for i in range(a.shape[1]):
                df_cols["{}_{}".format(c, i + 1)] = a[:, i]
        else:
            warnings.warn(
                "Ignoring {!r} because it has an unsupported number of "
                "dimensions.".format(c)
            )

    df = pd.DataFrame(df_cols)

    return df
