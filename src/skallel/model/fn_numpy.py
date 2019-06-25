import numpy as np
import numba


def genotype_array_is_called(gt):
    out = np.ones(gt.shape[:2], dtype=bool)
    _genotype_array_is_called(gt, out)
    return out


@numba.njit(numba.void(numba.int8[:, :, :], numba.boolean[:, :]), nogil=True)
def _genotype_array_is_called(gt, out):
    n_variants = gt.shape[0]
    n_samples = gt.shape[1]
    ploidy = gt.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                if gt[i, j, k] < 0:
                    out[i, j] = False
                    # no need to check other alleles
                    break
