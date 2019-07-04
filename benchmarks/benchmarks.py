import numpy as np
import dask.array as da
from skallel_tensor import methods_numpy, methods_dask


class TimeGenotypeTensor:
    """Timing benchmarks for genotype tensor functions."""

    def setup(self):
        self.data = np.random.randint(-1, 4, size=(10000, 1000, 2), dtype="i1")
        self.data_dask = da.from_array(self.data, chunks=(1000, 200, 2))

    def time_is_called_numpy(self):
        methods_numpy.genotype_tensor_is_called(self.data)

    def time_is_called_dask(self):
        methods_dask.genotype_tensor_is_called(self.data_dask).compute()

    def time_is_missing_numpy(self):
        methods_numpy.genotype_tensor_is_missing(self.data)

    def time_is_missing_dask(self):
        methods_dask.genotype_tensor_is_missing(self.data_dask).compute()

    def time_is_hom_numpy(self):
        methods_numpy.genotype_tensor_is_hom(self.data)

    def time_is_hom_dask(self):
        methods_dask.genotype_tensor_is_hom(self.data_dask).compute()

    def time_is_het_numpy(self):
        methods_numpy.genotype_tensor_is_het(self.data)

    def time_is_het_dask(self):
        methods_dask.genotype_tensor_is_het(self.data_dask).compute()

    def time_is_call_numpy(self):
        methods_numpy.genotype_tensor_is_call(
            self.data, np.array([0, 1], dtype="i1")
        )

    def time_is_call_dask(self):
        methods_dask.genotype_tensor_is_call(
            self.data_dask, np.array([0, 1], dtype="i1")
        ).compute()

    def time_count_alleles_numpy(self):
        methods_numpy.genotype_tensor_count_alleles(self.data, max_allele=3)

    def time_count_alleles_dask(self):
        methods_dask.genotype_tensor_count_alleles(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_allele_counts_numpy(self):
        methods_numpy.genotype_tensor_to_allele_counts(self.data, max_allele=3)

    def time_to_allele_counts_dask(self):
        methods_dask.genotype_tensor_to_allele_counts(
            self.data_dask, max_allele=3
        ).compute()

    def time_to_allele_counts_melt_numpy(self):
        methods_numpy.genotype_tensor_to_allele_counts_melt(
            self.data, max_allele=3
        )

    def time_to_allele_counts_melt_dask(self):
        methods_dask.genotype_tensor_to_allele_counts_melt(
            self.data_dask, max_allele=3
        ).compute()
