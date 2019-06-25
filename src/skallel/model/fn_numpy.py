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
