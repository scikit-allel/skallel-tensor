import numpy as np
import dask.array as da
from numba import cuda
import os
from skallel_tensor import numpy_backend, dask_backend, cuda_backend


cudasim = False
if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    cudasim = True


class TimeGenotypes3D:
    """Timing benchmarks for genotypes 3D functions."""

    def setup(self):
        self.data = np.random.randint(-1, 4, size=(20000, 1000, 2), dtype="i1")
        self.data_dask = da.from_array(self.data, chunks=(2000, 1000, 2))
        if not cudasim:
            self.data_cuda = cuda.to_device(self.data)
            self.data_dask_cuda = self.data_dask.map_blocks(cuda.to_device)

    def time_locate_hom_numpy(self):
        numpy_backend.genotypes_3d_locate_hom(self.data)

    def time_locate_hom_dask(self):
        dask_backend.genotypes_3d_locate_hom(self.data_dask).compute()

    def time_locate_het_numpy(self):
        numpy_backend.genotypes_3d_locate_het(self.data)

    def time_locate_het_dask(self):
        dask_backend.genotypes_3d_locate_het(self.data_dask).compute()

    def time_locate_call_numpy(self):
        numpy_backend.genotypes_3d_locate_call(
            self.data, np.array([0, 1], dtype="i1")
        )

    def time_locate_call_dask(self):
        dask_backend.genotypes_3d_locate_call(
            self.data_dask, call=np.array([0, 1], dtype="i1")
        ).compute()

    def time_count_alleles_numpy(self):
        numpy_backend.genotypes_3d_count_alleles(self.data, max_allele=3)

    def time_count_alleles_cuda(self):
        if not cudasim:
            cuda_backend.genotypes_3d_count_alleles(
                self.data_cuda, max_allele=3
            )
            cuda.synchronize()

    def time_count_alleles_dask(self):
        dask_backend.genotypes_3d_count_alleles(
            self.data_dask, max_allele=3
        ).compute()

    def time_count_alleles_dask_cuda(self):
        if not cudasim:
            dask_backend.genotypes_3d_count_alleles(
                self.data_dask_cuda, max_allele=3
            ).compute(scheduler="single-threaded")

    def time_to_called_allele_counts_numpy(self):
        numpy_backend.genotypes_3d_to_called_allele_counts(self.data)

    def time_to_called_allele_counts_dask(self):
        dask_backend.genotypes_3d_to_called_allele_counts(
            self.data_dask
        ).compute()

    def time_to_missing_allele_counts_numpy(self):
        numpy_backend.genotypes_3d_to_missing_allele_counts(self.data)

    def time_to_missing_allele_counts_dask(self):
        dask_backend.genotypes_3d_to_missing_allele_counts(
            self.data_dask
        ).compute()

    def time_to_allele_counts_numpy(self):
        numpy_backend.genotypes_3d_to_allele_counts(self.data, max_allele=3)

    def time_to_allele_counts_dask(self):
        dask_backend.genotypes_3d_to_allele_counts(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_allele_counts_melt_numpy(self):
        numpy_backend.genotypes_3d_to_allele_counts_melt(
            self.data, max_allele=3
        )

    def time_to_allele_counts_melt_dask(self):
        dask_backend.genotypes_3d_to_allele_counts_melt(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_major_allele_counts_numpy(self):
        numpy_backend.genotypes_3d_to_major_allele_counts(
            self.data, max_allele=3
        )

    def time_to_major_allele_counts_dask(self):
        dask_backend.genotypes_3d_to_major_allele_counts(
            self.data_dask, max_allele=3
        ).compute()


class TimeAlleleCounts2D:
    """Timing benchmarks for allele counts 2D functions."""

    def setup(self):
        self.data = np.random.randint(0, 100, size=(10000000, 4), dtype="i4")
        self.data_dask = da.from_array(self.data, chunks=(100000, -1))

    def time_to_frequencies_numpy(self):
        numpy_backend.allele_counts_2d_to_frequencies(self.data)

    def time_allelism_numpy(self):
        numpy_backend.allele_counts_2d_allelism(self.data)

    def time_max_allele_numpy(self):
        numpy_backend.allele_counts_2d_max_allele(self.data)

    def time_to_frequencies_dask(self):
        dask_backend.allele_counts_2d_to_frequencies(self.data_dask).compute()

    def time_allelism_dask(self):
        dask_backend.allele_counts_2d_allelism(self.data_dask).compute()

    def time_max_allele_dask(self):
        dask_backend.allele_counts_2d_max_allele(self.data_dask).compute()


class TimeAlleleCounts3D:
    """Timing benchmarks for allele counts 3D functions."""

    def setup(self):
        gt = np.random.randint(-1, 4, size=(10000, 1000, 2), dtype="i1")
        self.data = numpy_backend.genotypes_3d_to_allele_counts(
            gt, max_allele=3
        )
        self.data_dask = da.from_array(self.data, chunks=(1000, 200, -1))

    def time_to_frequencies_numpy(self):
        numpy_backend.allele_counts_3d_to_frequencies(self.data)

    def time_to_frequencies_dask(self):
        dask_backend.allele_counts_3d_to_frequencies(self.data_dask).compute()

    def time_allelism_numpy(self):
        numpy_backend.allele_counts_3d_allelism(self.data)

    def time_allelism_dask(self):
        dask_backend.allele_counts_3d_allelism(self.data_dask).compute()

    def time_max_allele_numpy(self):
        numpy_backend.allele_counts_3d_max_allele(self.data)

    def time_max_allele_dask(self):
        dask_backend.allele_counts_3d_max_allele(self.data_dask).compute()
