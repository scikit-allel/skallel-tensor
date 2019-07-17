from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
import math
from skallel_tensor import api


@cuda.jit
def kernel_genotypes_3d_count_alleles(gt, max_allele, out):
    m = gt.shape[0]
    i = cuda.grid(1)
    if i < m:
        n = gt.shape[1]
        p = gt.shape[2]
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1


def genotypes_3d_count_alleles(gt, *, max_allele):
    assert gt.ndim == 3
    assert cuda.is_cuda_array(gt)
    gt = cuda.as_cuda_array(gt)
    m = gt.shape[0]
    out = cuda.device_array((m, max_allele + 1), dtype=np.int32)
    threads = 32
    blocks = math.ceil(m / threads)
    kernel_genotypes_3d_count_alleles[blocks, threads](gt, max_allele, out)
    # TODO return cupy array to allow for inserting axes
    return out


api.dispatch_genotypes_3d_count_alleles.add(
    (DeviceNDArray,), genotypes_3d_count_alleles
)
