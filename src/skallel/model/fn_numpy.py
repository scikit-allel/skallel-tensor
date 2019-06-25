import numpy as np
import numba


def genotype_array_is_called(gt):
    out = np.ones(gt.shape[:2], dtype=bool)
    _genotype_array_is_called(gt, out)
    return out


@numba.njit(numba.void(numba.int8[:, :, :], numba.boolean[:, :]), nogil=True)
def _genotype_array_is_called(gt, out):
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # no need to check other alleles
                    break


def genotype_array_is_missing(gt):
    out = np.zeros(gt.shape[:2], dtype=bool)
    _genotype_array_is_missing(gt, out)
    return out


@numba.njit(numba.void(numba.int8[:, :, :], numba.boolean[:, :]), nogil=True)
def _genotype_array_is_missing(gt, out):
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            for k in range(gt.shape[2]):
                if gt[i, j, k] < 0:
                    out[i, j] = True
                    # no need to check other alleles
                    break


def genotype_array_is_hom(gt):
    out = np.ones(gt.shape[:2], dtype=bool)
    _genotype_array_is_hom(gt, out)
    return out


@numba.njit(numba.void(numba.int8[:, :, :], numba.boolean[:, :]), nogil=True)
def _genotype_array_is_hom(g, out):
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            first_allele = g[i, j, 0]
            if first_allele < 0:
                out[i, j] = False
            else:
                for k in range(1, g.shape[2]):
                    if g[i, j, k] != first_allele:
                        out[i, j] = False
                        # no need to check other alleles
                        break


def genotype_array_count_alleles(gt, max_allele):
    out = np.zeros((gt.shape[0], max_allele + 1), dtype="i4")
    _genotype_array_count_alleles(gt, max_allele, out)
    return out


@numba.njit(numba.void(numba.int8[:, :, :], numba.int8, numba.int32[:, :]), nogil=True)
def _genotype_array_count_alleles(g, max_allele, out):
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for k in range(g.shape[2]):
                allele = g[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
