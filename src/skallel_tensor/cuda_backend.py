from numba import cuda
import numpy as np
from skallel_tensor import api

# Simulated CUDA arrays.
from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray

cuda_array_types = (FakeCUDAArray,)
try:  # pragma: no cover
    # noinspection PyUnresolvedReferences
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    cuda_array_types += (DeviceNDArray,)
except ImportError:
    # Not available when using CUDA simulator.
    pass


@cuda.jit
def kernel_genotypes_3d_count_alleles(gt, max_allele, out):
    m = gt.shape[0]
    i = cuda.grid(1)
    if i < m:
        # Initialize to zero.
        for j in range(max_allele + 1):
            out[i, j] = 0
        n = gt.shape[1]
        p = gt.shape[2]
        for j in range(n):
            for k in range(p):
                allele = gt[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1


def genotypes_3d_count_alleles(gt, *, max_allele):
    assert gt.ndim == 3
    m = gt.shape[0]
    out = cuda.device_array((m, max_allele + 1), dtype=np.int32)
    # Let numba decide number of threads and blocks.
    kernel = kernel_genotypes_3d_count_alleles.forall(m)
    kernel(gt, max_allele, out)
    return out


api.dispatch_genotypes_3d_count_alleles.add(
    (cuda_array_types,), genotypes_3d_count_alleles
)
