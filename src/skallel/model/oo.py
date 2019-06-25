import numpy as np
import dask.array as da


from skallel.model import fn_numpy
from skallel.model import fn_dask


class GenotypeArray(object):
    """TODO"""

    def __init__(self, data):
        """TODO"""
        if not isinstance(data, (np.ndarray, da.Array)):
            raise TypeError("TODO")
        if data.dtype != np.dtype("i1"):
            raise TypeError("TODO")
        if data.ndim != 3:
            raise ValueError("TODO")
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
        if isinstance(self.data, np.ndarray):
            return fn_numpy.genotype_array_is_called(self.data)
        if isinstance(self.data, da.Array):
            return fn_dask.genotype_array_is_called(self.data)
