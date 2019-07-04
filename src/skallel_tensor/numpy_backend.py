import warnings
import numbers
import numpy as np
import numba
import pandas as pd
from . import api, utils


def select_slice(a, start=None, stop=None, step=None, axis=0):
    item = utils.expand_slice(
        start=start, stop=stop, step=step, axis=axis, ndim=a.ndim
    )
    return a[item]


api.select_slice.add((np.ndarray,), select_slice)


def select_indices(a, indices, axis=0):
    return np.take(a, indices, axis=axis)


api.select_indices.add((np.ndarray, np.ndarray), select_indices)


def select_mask(a, mask, axis=0):
    return np.compress(mask, a, axis=axis)


api.select_mask.add((np.ndarray, np.ndarray), select_mask)


def concatenate(seq, axis=0):
    return np.concatenate(seq, axis=axis)


api.concatenate_dispatcher.add((np.ndarray,), concatenate)


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


api.genotype_tensor_is_called.add((np.ndarray,), genotype_tensor_is_called)


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


api.genotype_tensor_is_missing.add((np.ndarray,), genotype_tensor_is_missing)


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


api.genotype_tensor_is_hom.add((np.ndarray,), genotype_tensor_is_hom)


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


api.genotype_tensor_is_het.add((np.ndarray,), genotype_tensor_is_het)


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


api.genotype_tensor_is_call.add(
    (np.ndarray, np.ndarray), genotype_tensor_is_call
)


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


api.genotype_tensor_count_alleles.add(
    (np.ndarray, numbers.Integral), genotype_tensor_count_alleles
)


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


api.genotype_tensor_to_allele_counts.add(
    (np.ndarray, numbers.Integral), genotype_tensor_to_allele_counts
)


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


api.genotype_tensor_to_allele_counts_melt.add(
    (np.ndarray, numbers.Integral), genotype_tensor_to_allele_counts_melt
)


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


api.variants_to_dataframe_dispatcher.add((np.ndarray,), variants_to_dataframe)
