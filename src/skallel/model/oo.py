import numpy as np
import dask.array as da
import pandas as pd


from . import fn_numpy
from . import fn_dask


def is_hdf5_like(x):
    return (
        hasattr(x, "ndim")
        and hasattr(x, "dtype")
        and hasattr(x, "shape")
        and hasattr(x, "chunks")
        and len(x.chunks) == len(x.shape) == x.ndim
    )


class GenotypeArray(object):
    """TODO"""

    def __init__(self, data):
        """TODO"""

        # check type
        if isinstance(data, np.ndarray):
            self._fn = fn_numpy
        elif isinstance(data, da.Array):
            self._fn = fn_dask
        elif is_hdf5_like(data):
            data = da.from_array(data, chunks=data.chunks)
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
        # TODO support allele argument
        return self._fn.genotype_array_is_hom(self.data)

    def is_het(self):
        """TODO"""
        # TODO support allele argument
        return self._fn.genotype_array_is_het(self.data)

    # TODO is_call
    # TODO to_n_ref
    # TODO to_n_alt
    # TODO to_allele_counts
    # TODO to_haplotypes
    # TODO __repr__
    # TODO display
    # TODO map_alleles
    # TODO max

    def count_alleles(self, max_allele):
        """TODO"""
        # TODO support subpop arg
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
    # TODO concatenate


# TODO HaplotypeArray
# TODO n_variants
# TODO n_haplotypes
# TODO __getitem__
# TODO take
# TODO compress
# TODO concatenate
# TODO is_called
# TODO is_missing
# TODO is_ref
# TODO is_alt
# TODO is_call
# TODO to_genotypes
# TODO count_alleles
# TODO map_alleles
# TODO prefix_argsort
# TODO distinct
# TODO distinct_counts
# TODO distinct_frequencies
# TODO display
# TODO __repr__


# TODO AlleleCountsArray
# TODO __add__
# TODO __sub__
# TODO n_variants
# TODO n_alleles
# TODO __getitem__
# TODO compress
# TODO take
# TODO concatenate
# TODO to_frequencies
# TODO allelism
# TODO max_allele
# TODO is_variant
# TODO is_non_variant
# TODO is_segregating
# TODO is_non_segregating
# TODO is_singleton
# TODO is_doubleton
# TODO is_biallelic
# TODO is_biallelic_01
# TODO map_alleles
# TODO display
# TODO __repr__


# TODO GenotypeAlleleCountsArray


# TODO Callset
# TODO __getitem__


# TODO ContigCallset
# TODO __getitem__
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
# TODO concatenate?
# TODO variants_to_dataframe
# TODO variants_to_dask_dataframe


VCF_FIXED_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL"]


class ContigCallset(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def variants_to_dataframe(self, index="POS", columns=None):

        # discover variants array keys
        all_keys = sorted(self.data["variants"])

        if columns is None:
            # use all keys, reordering so VCF fixed fields are first
            keys = [k for k in VCF_FIXED_FIELDS if k in all_keys]
            keys += [k for k in all_keys if k.startswith("FILTER")]
            keys += [k for k in all_keys if k not in keys]

        else:
            # check requested columns are present in data
            for k in columns:
                if k not in all_keys:
                    raise ValueError("TODO")
            # use only requested columns, in requested order
            keys = columns

        # build dataframe
        df_cols = {}
        for k in keys:
            # load values
            a = self.data["variants"][k][:]
            # check number of dimensions
            if a.ndim == 1:
                df_cols[k] = a
            elif a.ndim == 2:
                for i in range(a.shape[1]):
                    df_cols["{}_{}".format(k, i + 1)] = a[:, i]
            else:
                raise ValueError("TODO")
        df = pd.DataFrame(df_cols)

        # set index
        if index is not None and index in df:
            df.set_index(index, inplace=True)

        return df
