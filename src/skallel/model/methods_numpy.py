import numpy as np
import numba


ARRAY_TYPE = np.ndarray


def array_check(a):
    if isinstance(a, np.ndarray):
        return a
    raise TypeError


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_array_is_called(gt):
    out = np.ones(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # no need to check other alleles
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_array_is_missing(gt):
    out = np.zeros(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = True
                    # no need to check other alleles
                    break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_array_is_hom(gt):
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
                        # no need to check other alleles
                        break
    return out


@numba.njit(numba.boolean[:, :](numba.int8[:, :, :]), nogil=True)
def genotype_array_is_het(gt):
    out = np.zeros(gt.shape[:2], dtype=np.bool_)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            first_allele = gt[i, j, 0]
            if first_allele >= 0:
                for k in range(1, gt.shape[2]):
                    allele = gt[i, j, k]
                    if allele >= 0 and allele != first_allele:
                        out[i, j] = True
                        # no need to check other alleles
                        break
    return out


@numba.njit(numba.int32[:, :](numba.int8[:, :, :], numba.int8), nogil=True)
def genotype_array_count_alleles(gt, max_allele):
    out = np.zeros((gt.shape[0], max_allele + 1), dtype=np.int32)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
    return out
