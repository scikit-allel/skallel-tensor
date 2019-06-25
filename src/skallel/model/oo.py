import numpy as np
import dask.array as da


from . import fn_numpy
from . import fn_dask


class GenotypeArray(object):
    """TODO"""

    def __init__(self, data):
        """TODO"""

        # check type
        if isinstance(data, np.ndarray):
            self._fn = fn_numpy
        elif isinstance(data, da.Array):
            self._fn = fn_dask
        else:
            raise TypeError("TODO")

        # check dtype
        if data.dtype != np.dtype("i1"):
            raise TypeError("TODO")

        # check number of dimensions
        if data.ndim != 3:
            raise ValueError("TODO")

        # all good, store data
        self._data = data

    @property
    def data(self):
        """TODO"""
        return self._data

    @property
    def n_variants(self):
        """TODO"""
        return self.data.shape[0]

    @property
    def n_samples(self):
        """TODO"""
        return self.data.shape[1]

    @property
    def ploidy(self):
        """TODO"""
        return self.data.shape[2]

    @property
    def values(self):
        """Deprecated, use the `data` property instead. Provided for
        backwards-compatibility."""
        return self._data

    def is_called(self):
        """TODO"""
        return self._fn.genotype_array_is_called(self.data)

    def is_missing(self):
        """TODO"""
        return self._fn.genotype_array_is_missing(self.data)

    def is_hom(self):
        """TODO"""
        return self._fn.genotype_array_is_hom(self.data)

    def count_alleles(self, max_allele):
        """TODO"""
        # TODO wrap the result as AlleleCountsArray
        return self._fn.genotype_array_count_alleles(self.data, max_allele)

    # TODO __getitem__ with support for simple slices and/or ints only
    # TODO select_variants_by_id
    # TODO select_variants_by_position
    # TODO select_variants_by_region
    # TODO select_variants_by_index
    # TODO select_variants_by_mask
    # TODO select_samples_by_id
    # TODO select_samples_by_index
    # TODO select_samples_by_mask
    # TODO take
    # TODO compress


# TODO Callset
# TODO ContigCallset
# TODO HaplotypeArray
# TODO AlleleCountsArray
# TODO GenotypeAlleleCountsArray
