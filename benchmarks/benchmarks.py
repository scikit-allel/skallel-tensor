import numpy as np
import dask.array as da
from skallel_tensor import numpy_backend, dask_backend


class TimeGenotypeTensor:
    """Timing benchmarks for genotype tensor functions."""

    def setup(self):
        self.data = np.random.randint(-1, 4, size=(10000, 1000, 2), dtype="i1")
        self.data_dask = da.from_array(self.data, chunks=(1000, 200, 2))

    def time_is_called_numpy(self):
        numpy_backend.genotype_tensor_is_called(self.data)

    def time_is_called_dask(self):
        dask_backend.genotype_tensor_is_called(self.data_dask).compute()

    def time_is_missing_numpy(self):
        numpy_backend.genotype_tensor_is_missing(self.data)

    def time_is_missing_dask(self):
        dask_backend.genotype_tensor_is_missing(self.data_dask).compute()

    def time_is_hom_numpy(self):
        numpy_backend.genotype_tensor_is_hom(self.data)

    def time_is_hom_dask(self):
        dask_backend.genotype_tensor_is_hom(self.data_dask).compute()

    def time_is_het_numpy(self):
        numpy_backend.genotype_tensor_is_het(self.data)

    def time_is_het_dask(self):
        dask_backend.genotype_tensor_is_het(self.data_dask).compute()

    def time_is_call_numpy(self):
        numpy_backend.genotype_tensor_is_call(
            self.data, np.array([0, 1], dtype="i1")
        )

    def time_is_call_dask(self):
        dask_backend.genotype_tensor_is_call(
            self.data_dask, np.array([0, 1], dtype="i1")
        ).compute()

    def time_count_alleles_numpy(self):
        numpy_backend.genotype_tensor_count_alleles(self.data, max_allele=3)

    def time_count_alleles_dask(self):
        dask_backend.genotype_tensor_count_alleles(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_allele_counts_numpy(self):
        numpy_backend.genotype_tensor_to_allele_counts(self.data, max_allele=3)

    def time_to_allele_counts_dask(self):
        dask_backend.genotype_tensor_to_allele_counts(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_allele_counts_melt_numpy(self):
        numpy_backend.genotype_tensor_to_allele_counts_melt(
            self.data, max_allele=3
        )

    def time_to_allele_counts_melt_dask(self):
        dask_backend.genotype_tensor_to_allele_counts_melt(
            self.data_dask, max_allele=3
        ).compute()


class TimeAlleleCounts:
    """Timing benchmarks for allele counts functions."""

    def setup(self):
        self.data = np.random.randint(0, 100, size=(1000000, 4), dtype="i4")
        self.data_dask = da.from_array(self.data, chunks=(10000, -1))

    def time_to_frequencies(self):
        numpy_backend.allele_counts_to_frequencies(self.data)

    def time_allelism(self):
        numpy_backend.allele_counts_allelism(self.data)

    def time_max_allele(self):
        numpy_backend.allele_counts_max_allele(self.data)

    def time_is_variant(self):
        numpy_backend.allele_counts_is_variant(self.data)

    def time_is_segregating(self):
        numpy_backend.allele_counts_is_segregating(self.data)
