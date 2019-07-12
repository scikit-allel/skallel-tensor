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


api.select_slice_dispatcher.add((np.ndarray,), select_slice)


def select_indices(a, indices, axis=0):
    return np.take(a, indices, axis=axis)


api.select_indices_dispatcher.add((np.ndarray, np.ndarray), select_indices)


def select_mask(a, mask, axis=0):
    return np.compress(mask, a, axis=axis)


api.select_mask_dispatcher.add((np.ndarray, np.ndarray), select_mask)


def concatenate(seq, axis=0):
    return np.concatenate(seq, axis=axis)


api.concatenate_dispatcher.add((np.ndarray,), concatenate)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True, parallel=True)
def genotype_tensor_is_called(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in numba.prange(m):
        for j in range(n):
            for k in range(p):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


api.genotype_tensor_is_called_dispatcher.add(
    (np.ndarray,), genotype_tensor_is_called
)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True, parallel=True)
def genotype_tensor_is_missing(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in numba.prange(m):
        for j in range(n):
            for k in range(gt.shape[p]):
                if gt[i, j, k] < 0:
                    out[i, j] = True
                    # No need to check other alleles.
                    break
    return out


api.genotype_tensor_is_missing_dispatcher.add(
    (np.ndarray,), genotype_tensor_is_missing
)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True, parallel=True)
def genotype_tensor_is_hom(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in numba.prange(m):
        for j in range(n):
            first_allele = gt[i, j, 0]
            if first_allele < 0:
                out[i, j] = False
            else:
                for k in range(1, p):
                    if gt[i, j, k] != first_allele:
                        out[i, j] = False
                        # No need to check other alleles.
                        break
    return out


api.genotype_tensor_is_hom_dispatcher.add((np.ndarray,), genotype_tensor_is_hom)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True, parallel=True)
def genotype_tensor_is_het(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in numba.prange(m):
        for j in range(n):
            first_allele = gt[i, j, 0]
            if first_allele >= 0:
                for k in range(1, p):
                    allele = gt[i, j, k]
                    if allele >= 0 and allele != first_allele:
                        out[i, j] = True
                        # No need to check other alleles.
                        break
    return out


api.genotype_tensor_is_het_dispatcher.add((np.ndarray,), genotype_tensor_is_het)


@numba.njit(
    numba.boolean[:, :](numba.int8[:, :, :], numba.int8[:]),
    nogil=True,
    parallel=True,
)
def genotype_tensor_is_call(gt, call):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in numba.prange(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] != call[k]:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


api.genotype_tensor_is_call_dispatcher.add(
    (np.ndarray, np.ndarray), genotype_tensor_is_call
)


@numba.njit(
    numba.int32[:, :](numba.int8[:, :, :], numba.int8),
    nogil=True,
    parallel=True,
)
def genotype_tensor_count_alleles(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, max_allele + 1), dtype=np.int32)
    for i in numba.prange(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
    return out


api.genotype_tensor_count_alleles_dispatcher.add(
    (np.ndarray, numbers.Integral), genotype_tensor_count_alleles
)


@numba.njit(
    numba.int8[:, :, :](numba.int8[:, :, :], numba.int8),
    parallel=True,
    nogil=True,
)
def genotype_tensor_to_allele_counts(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n, max_allele + 1), dtype=np.int8)
    for i in numba.prange(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, j, allele] += 1
    return out


api.genotype_tensor_to_allele_counts_dispatcher.add(
    (np.ndarray, numbers.Integral), genotype_tensor_to_allele_counts
)


@numba.njit(
    numba.int8[:, :](numba.int8[:, :, :], numba.int8), nogil=True, parallel=True
)
def genotype_tensor_to_allele_counts_melt(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m * (max_allele + 1), n), dtype=np.int8)
    for i in numba.prange(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[(i * (max_allele + 1)) + allele, j] += 1
    return out


api.genotype_tensor_to_allele_counts_melt_dispatcher.add(
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


@numba.njit(numba.float32[:, :](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_to_frequencies(ac):
    out = np.empty(ac.shape, dtype=np.float32)
    for i in numba.prange(ac.shape[0]):
        n = 0
        for j in range(ac.shape[1]):
            n += ac[i, j]
        if n > 0:
            for j in range(ac.shape[1]):
                out[i, j] = ac[i, j] / n
        else:
            for j in range(ac.shape[1]):
                out[i, j] = np.nan
    return out


@numba.njit(numba.int8[:](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_allelism(ac):
    out = np.zeros(ac.shape[0], dtype=np.int8)
    for i in numba.prange(ac.shape[0]):
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                out[i] += 1
    return out


@numba.njit(numba.int8[:](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_max_allele(ac):
    out = np.empty(ac.shape[0], dtype=np.int8)
    for i in numba.prange(ac.shape[0]):
        m = -1
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                m = j
        out[i] = m
    return out


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_is_segregating(ac):
    out = np.zeros(ac.shape[0], dtype=np.bool_)
    for i in numba.prange(ac.shape[0]):
        n = 0
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                n += 1
        if n > 1:
            out[i] = True
    return out


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_is_variant(ac):
    out = np.zeros(ac.shape[0], dtype=np.bool_)
    for i in numba.prange(ac.shape[0]):
        for j in range(1, ac.shape[1]):
            if ac[i, j] > 0:
                out[i] = True
                break
    return out


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True, parallel=True)
def allele_counts_is_non_variant(ac):
    out = np.ones(ac.shape[0], dtype=np.bool_)
    for i in numba.prange(ac.shape[0]):
        for j in range(1, ac.shape[1]):
            if ac[i, j] > 0:
                out[i] = False
                break
    return out
