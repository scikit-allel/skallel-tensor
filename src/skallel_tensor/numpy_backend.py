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


api.dispatch_select_slice.add((np.ndarray,), select_slice)


def select_indices(a, indices, axis=0):
    return np.take(a, indices, axis=axis)


api.dispatch_select_indices.add((np.ndarray, np.ndarray), select_indices)


def select_mask(a, mask, axis=0):
    return np.compress(mask, a, axis=axis)


api.dispatch_select_mask.add((np.ndarray, np.ndarray), select_mask)


def concatenate(seq, axis=0):
    return np.concatenate(seq, axis=axis)


api.dispatch_concatenate.add((np.ndarray,), concatenate)


@numba.njit(numba.boolean[:](numba.int8[:, :]), nogil=True)
def genotypes_2d_locate_called(gt):
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.ones(n, dtype=np.bool_)
    for i in range(n):
        for j in range(p):
            if gt[i, j] < 0:
                out[i] = False
                # No need to check other alleles.
                break
    return out


api.dispatch_genotypes_2d_locate_called.add(
    (np.ndarray,), genotypes_2d_locate_called
)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotypes_3d_locate_called(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


api.dispatch_genotypes_3d_locate_called.add(
    (np.ndarray,), genotypes_3d_locate_called
)


@numba.njit(numba.boolean[:](numba.int8[:, :]), nogil=True)
def genotypes_2d_locate_missing(gt):
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        for j in range(p):
            if gt[i, j] < 0:
                out[i] = True
                # No need to check other alleles.
                break
    return out


api.dispatch_genotypes_2d_locate_missing.add(
    (np.ndarray,), genotypes_2d_locate_missing
)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotypes_3d_locate_missing(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            for k in range(gt.shape[p]):
                if gt[i, j, k] < 0:
                    out[i, j] = True
                    # No need to check other alleles.
                    break
    return out


api.dispatch_genotypes_3d_locate_missing.add(
    (np.ndarray,), genotypes_3d_locate_missing
)


@numba.njit(numba.boolean[:](numba.int8[:, :]), nogil=True)
def genotypes_2d_locate_hom(gt):
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.ones(n, dtype=np.bool_)
    for i in range(n):
        first_allele = gt[i, 0]
        if first_allele < 0:
            out[i] = False
        else:
            for j in range(1, p):
                if gt[i, j] != first_allele:
                    out[i] = False
                    # No need to check other alleles.
                    break
    return out


api.dispatch_genotypes_2d_locate_hom.add((np.ndarray,), genotypes_2d_locate_hom)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotypes_3d_locate_hom(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in range(m):
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


api.dispatch_genotypes_3d_locate_hom.add((np.ndarray,), genotypes_3d_locate_hom)


@numba.njit(numba.boolean[:](numba.int8[:, :]), nogil=True)
def genotypes_2d_locate_het(gt):
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        first_allele = gt[i, 0]
        if first_allele >= 0:
            for j in range(1, p):
                allele = gt[i, j]
                if allele >= 0 and allele != first_allele:
                    out[i] = True
                    # No need to check other alleles.
                    break
    return out


api.dispatch_genotypes_2d_locate_het.add((np.ndarray,), genotypes_2d_locate_het)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotypes_3d_locate_het(gt):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in range(m):
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


api.dispatch_genotypes_3d_locate_het.add((np.ndarray,), genotypes_3d_locate_het)


@numba.njit(numba.boolean[:](numba.int8[:, :], numba.int8[:]), nogil=True)
def genotypes_2d_locate_call(gt, call):
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.ones(n, dtype=np.bool_)
    for i in range(n):
        for j in range(p):
            if gt[i, j] != call[j]:
                out[i] = False
                # No need to check other alleles.
                break
    return out


api.dispatch_genotypes_2d_locate_call.add(
    (np.ndarray, np.ndarray), genotypes_2d_locate_call
)


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :], numba.int8[:]), nogil=True)
def genotypes_3d_locate_call(gt, call):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if gt[i, j, k] != call[k]:
                    out[i, j] = False
                    # No need to check other alleles.
                    break
    return out


api.dispatch_genotypes_3d_locate_call.add(
    (np.ndarray, np.ndarray), genotypes_3d_locate_call
)


@numba.njit(numba.int32[:, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotypes_3d_count_alleles(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, max_allele + 1), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
    return out


api.dispatch_genotypes_3d_count_alleles.add(
    (np.ndarray,), genotypes_3d_count_alleles
)


@numba.njit(numba.int8[:, :, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotypes_3d_to_allele_counts(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n, max_allele + 1), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, j, allele] += 1
    return out


api.dispatch_genotypes_3d_to_allele_counts.add(
    (np.ndarray, numbers.Integral), genotypes_3d_to_allele_counts
)


@numba.njit(numba.int8[:, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotypes_3d_to_allele_counts_melt(gt, max_allele):
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m * (max_allele + 1), n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[(i * (max_allele + 1)) + allele, j] += 1
    return out


api.dispatch_genotypes_3d_to_allele_counts_melt.add(
    (np.ndarray, numbers.Integral), genotypes_3d_to_allele_counts_melt
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


api.dispatch_variants_to_dataframe.add((np.ndarray,), variants_to_dataframe)


@numba.njit(numba.float32[:, :](numba.int32[:, :]), nogil=True)
def allele_counts_2d_to_frequencies(ac):
    out = np.empty(ac.shape, dtype=np.float32)
    for i in range(ac.shape[0]):
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


api.dispatch_allele_counts_2d_to_frequencies.add(
    (np.ndarray,), allele_counts_2d_to_frequencies
)


@numba.njit(numba.int8[:](numba.int32[:, :]), nogil=True)
def allele_counts_2d_allelism(ac):
    out = np.zeros(ac.shape[0], dtype=np.int8)
    for i in range(ac.shape[0]):
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                out[i] += 1
    return out


api.dispatch_allele_counts_2d_allelism.add(
    (np.ndarray,), allele_counts_2d_allelism
)


@numba.njit(numba.int8[:](numba.int32[:, :]), nogil=True)
def allele_counts_2d_max_allele(ac):
    out = np.empty(ac.shape[0], dtype=np.int8)
    for i in range(ac.shape[0]):
        m = -1
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                m = j
        out[i] = m
    return out


api.dispatch_allele_counts_2d_max_allele.add(
    (np.ndarray,), allele_counts_2d_max_allele
)


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True)
def allele_counts_2d_locate_segregating(ac):
    out = np.zeros(ac.shape[0], dtype=np.bool_)
    for i in range(ac.shape[0]):
        n = 0
        for j in range(ac.shape[1]):
            if ac[i, j] > 0:
                n += 1
        if n > 1:
            out[i] = True
    return out


api.dispatch_allele_counts_2d_locate_segregating.add(
    (np.ndarray,), allele_counts_2d_locate_segregating
)


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True)
def allele_counts_2d_locate_variant(ac):
    out = np.zeros(ac.shape[0], dtype=np.bool_)
    for i in range(ac.shape[0]):
        for j in range(1, ac.shape[1]):
            if ac[i, j] > 0:
                out[i] = True
                break
    return out


api.dispatch_allele_counts_2d_locate_variant.add(
    (np.ndarray,), allele_counts_2d_locate_variant
)


@numba.njit(numba.boolean[:](numba.int32[:, :]), nogil=True)
def allele_counts_2d_locate_non_variant(ac):
    out = np.ones(ac.shape[0], dtype=np.bool_)
    for i in range(ac.shape[0]):
        for j in range(1, ac.shape[1]):
            if ac[i, j] > 0:
                out[i] = False
                break
    return out


api.dispatch_allele_counts_2d_locate_non_variant.add(
    (np.ndarray,), allele_counts_2d_locate_non_variant
)


@numba.njit(numba.int8[:, :](numba.int8[:, :, :]), nogil=True)
def allele_counts_3d_allelism(ac):
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            allelism = np.int8(0)
            for k in range(p):
                if ac[i, j, k] > 0:
                    allelism += 1
            out[i, j] = allelism
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def allele_counts_3d_locate_called(ac):
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if ac[i, j, k] > 0:
                    out[i, j] = True
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def allele_counts_3d_locate_missing(ac):
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.ones((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if ac[i, j, k] > 0:
                    out[i, j] = False
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def allele_counts_3d_locate_hom(ac):
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            allelism = 0
            for k in range(p):
                if ac[i, j, k] > 0:
                    allelism += 1
            if allelism == 1:
                out[i, j] = True
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def allele_counts_3d_locate_het(ac):
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.zeros((m, n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            allelism = 0
            for k in range(p):
                if ac[i, j, k] > 0:
                    allelism += 1
            if allelism > 1:
                out[i, j] = True
    return out
