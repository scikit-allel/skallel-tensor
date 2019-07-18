import warnings
import numpy as np
import numba
import pandas as pd
from . import api, utils


def select_slice(a, *, start=None, stop=None, step=None, axis=0):
    item = utils.expand_slice(
        start=start, stop=stop, step=step, axis=axis, ndim=a.ndim
    )
    return a[item]


api.dispatch_select_slice.add((np.ndarray,), select_slice)


def select_indices(a, indices, *, axis=0):
    return np.take(a, indices, axis=axis)


api.dispatch_select_indices.add((np.ndarray, np.ndarray), select_indices)


def select_mask(a, mask, *, axis=0):
    return np.compress(mask, a, axis=axis)


api.dispatch_select_mask.add((np.ndarray, np.ndarray), select_mask)


def concatenate(seq, *, axis=0):
    return np.concatenate(seq, axis=axis)


api.dispatch_concatenate.add((np.ndarray,), concatenate)


@numba.njit(nogil=True)
def genotypes_2d_to_called_allele_counts(gt):
    assert gt.ndim == 2
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        for j in range(p):
            if gt[i, j] >= 0:
                out[i] += 1
    return out


api.dispatch_genotypes_2d_to_called_allele_counts.add(
    (np.ndarray,), genotypes_2d_to_called_allele_counts
)


@numba.njit(nogil=True)
def genotypes_3d_to_called_allele_counts(gt):
    assert gt.ndim == 3
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if gt[i, j, k] >= 0:
                    out[i, j] += 1
    return out


api.dispatch_genotypes_3d_to_called_allele_counts.add(
    (np.ndarray,), genotypes_3d_to_called_allele_counts
)


@numba.njit(nogil=True)
def genotypes_2d_to_missing_allele_counts(gt):
    assert gt.ndim == 2
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        for j in range(p):
            if gt[i, j] < 0:
                out[i] += 1
    return out


api.dispatch_genotypes_2d_to_missing_allele_counts.add(
    (np.ndarray,), genotypes_2d_to_missing_allele_counts
)


@numba.njit(nogil=True)
def genotypes_3d_to_missing_allele_counts(gt):
    assert gt.ndim == 3
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    out = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            for k in range(gt.shape[p]):
                if gt[i, j, k] < 0:
                    out[i, j] += 1
    return out


api.dispatch_genotypes_3d_to_missing_allele_counts.add(
    (np.ndarray,), genotypes_3d_to_missing_allele_counts
)


@numba.njit(nogil=True)
def genotypes_2d_locate_hom(gt):
    assert gt.ndim == 2
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


@numba.njit(nogil=True)
def genotypes_3d_locate_hom(gt):
    assert gt.ndim == 3
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


@numba.njit(nogil=True)
def genotypes_2d_locate_het(gt):
    assert gt.ndim == 2
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


@numba.njit(nogil=True)
def genotypes_3d_locate_het(gt):
    assert gt.ndim == 3
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


@numba.njit(nogil=True)
def genotypes_2d_locate_call(gt, *, call):
    assert gt.ndim == 2
    assert call.ndim == 1
    assert gt.shape[1] == call.shape[0]
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
    (np.ndarray,), genotypes_2d_locate_call
)


@numba.njit(nogil=True)
def genotypes_3d_locate_call(gt, *, call):
    assert gt.ndim == 3
    assert call.ndim == 1
    assert gt.shape[2] == call.shape[0]
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
    (np.ndarray,), genotypes_3d_locate_call
)


@numba.njit(nogil=True)
def genotypes_3d_count_alleles(gt, *, max_allele):
    assert gt.ndim == 3
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


@numba.njit(nogil=True)
def genotypes_2d_to_allele_counts(gt, *, max_allele):
    assert gt.ndim == 2
    n = gt.shape[0]
    p = gt.shape[1]
    out = np.zeros((n, max_allele + 1), dtype=np.int8)
    for i in range(n):
        for j in range(p):
            allele = gt[i, j]
            if 0 <= allele <= max_allele:
                out[i, allele] += 1
    return out


api.dispatch_genotypes_2d_to_allele_counts.add(
    (np.ndarray,), genotypes_2d_to_allele_counts
)


@numba.njit(nogil=True)
def genotypes_3d_to_allele_counts(gt, *, max_allele):
    assert gt.ndim == 3
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
    (np.ndarray,), genotypes_3d_to_allele_counts
)


@numba.njit(nogil=True)
def genotypes_3d_to_allele_counts_melt(gt, *, max_allele):
    assert gt.ndim == 3
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
    (np.ndarray,), genotypes_3d_to_allele_counts_melt
)


@numba.njit(nogil=True)
def genotypes_3d_to_major_allele_counts(gt, *, max_allele):
    assert gt.ndim == 3
    m = gt.shape[0]
    n = gt.shape[1]
    p = gt.shape[2]
    ac = np.zeros(max_allele + 1, dtype=np.int32)
    out = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        # First count alleles, needed to find major allele.
        ac[:] = 0
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    ac[allele] += 1
        # Now find major allele.
        major_allele = -1
        major_allele_count = 0
        for j in range(max_allele + 1):
            c = ac[j]
            if c > major_allele_count:
                major_allele = j
                major_allele_count = c
        # Now recode genotypes as major allele counts.
        if major_allele >= 0:
            for j in range(n):
                for k in range(p):
                    if gt[i, j, k] == major_allele:
                        out[i, j] += 1
    return out


api.dispatch_genotypes_3d_to_major_allele_counts.add(
    (np.ndarray,), genotypes_3d_to_major_allele_counts
)


def genotypes_3d_to_haplotypes(gt):
    assert gt.ndim == 3
    m = gt.shape[0]
    return gt.reshape((m, -1))


api.dispatch_genotypes_3d_to_haplotypes.add(
    (np.ndarray,), genotypes_3d_to_haplotypes
)


def variants_to_dataframe(variants, *, columns):

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


@numba.njit(nogil=True)
def allele_counts_2d_to_frequencies(ac):
    assert ac.ndim == 2
    n = ac.shape[0]
    p = ac.shape[1]
    out = np.empty((n, p), dtype=np.float32)
    for i in range(n):
        s = 0
        for j in range(p):
            c = ac[i, j]
            if c > 0:
                s += c
        if s > 0:
            for j in range(p):
                c = ac[i, j]
                if c >= 0:
                    out[i, j] = np.float32(c) / np.float32(s)
                else:
                    out[i, j] = np.nan
        else:
            for j in range(p):
                out[i, j] = np.nan
    return out


api.dispatch_allele_counts_2d_to_frequencies.add(
    (np.ndarray,), allele_counts_2d_to_frequencies
)


@numba.njit(nogil=True)
def allele_counts_3d_to_frequencies(ac):
    assert ac.ndim == 3
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.empty((m, n, p), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(p):
                c = ac[i, j, k]
                if c > 0:
                    s += c
            if s > 0:
                for k in range(p):
                    c = ac[i, j, k]
                    if c >= 0:
                        out[i, j, k] = np.float32(c) / np.float32(s)
                    else:
                        out[i, j, k] = np.nan
            else:
                for k in range(p):
                    out[i, j, k] = np.nan
    return out


api.dispatch_allele_counts_3d_to_frequencies.add(
    (np.ndarray,), allele_counts_3d_to_frequencies
)


@numba.njit(nogil=True)
def allele_counts_2d_allelism(ac):
    assert ac.ndim == 2
    n = ac.shape[0]
    p = ac.shape[1]
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        for j in range(p):
            if ac[i, j] > 0:
                out[i] += 1
    return out


api.dispatch_allele_counts_2d_allelism.add(
    (np.ndarray,), allele_counts_2d_allelism
)


@numba.njit(nogil=True)
def allele_counts_3d_allelism(ac):
    assert ac.ndim == 3
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


api.dispatch_allele_counts_3d_allelism.add(
    (np.ndarray,), allele_counts_3d_allelism
)


@numba.njit(nogil=True)
def allele_counts_2d_max_allele(ac):
    assert ac.ndim == 2
    n = ac.shape[0]
    p = ac.shape[1]
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        x = -1
        for j in range(p):
            if ac[i, j] > 0:
                x = j
        out[i] = x
    return out


api.dispatch_allele_counts_2d_max_allele.add(
    (np.ndarray,), allele_counts_2d_max_allele
)


@numba.njit(nogil=True)
def allele_counts_3d_max_allele(ac):
    assert ac.ndim == 3
    m = ac.shape[0]
    n = ac.shape[1]
    p = ac.shape[2]
    out = np.empty((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            x = -1
            for k in range(p):
                if ac[i, j, k] > 0:
                    x = k
            out[i, j] = x
    return out


api.dispatch_allele_counts_3d_max_allele.add(
    (np.ndarray,), allele_counts_3d_max_allele
)
