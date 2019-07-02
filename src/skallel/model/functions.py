import numbers
from collections.abc import Mapping
import numpy as np
from . import methods_numpy, methods_dask


methods_providers = [methods_numpy, methods_dask]


def int_check(i, dtype):
    """TODO"""

    if not isinstance(i, numbers.Integral):
        raise TypeError  # TODO message
    if not np.can_cast(i, dtype, casting="safe"):
        raise ValueError  # TODO message
    return np.array(i, dtype)[()]


def array_check(a):
    """TODO"""

    for methods in methods_providers:
        try:
            return methods.array_check(a)
        except TypeError:
            pass

    raise TypeError  # TODO message


def get_methods(a):
    """TODO"""

    for methods in methods_providers:
        if isinstance(a, methods.ARRAY_TYPE):
            return methods

    raise RuntimeError  # should not reach here if array checks done properly


def genotype_array_check(gt):
    """TODO"""

    # check type
    gt = array_check(gt)

    # check dtype
    if gt.dtype != np.dtype("i1"):
        raise TypeError  # TODO message

    # check number of dimensions
    if gt.ndim != 3:
        raise ValueError  # TODO message

    return gt


def genotype_array_is_called(gt):
    """TODO"""

    # check arguments
    gt = genotype_array_check(gt)

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_is_called(gt)


def genotype_array_is_missing(gt):
    """TODO"""

    # check arguments
    gt = genotype_array_check(gt)

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_is_missing(gt)


def genotype_array_is_hom(gt):
    """TODO"""

    # check arguments
    gt = genotype_array_check(gt)

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_is_hom(gt)


def genotype_array_is_het(gt):
    """TODO"""

    # check arguments
    gt = genotype_array_check(gt)

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_is_het(gt)


def genotype_array_count_alleles(gt, max_allele):
    """TODO"""

    # TODO support subpop arg

    # check arguments
    gt = genotype_array_check(gt)
    max_allele = int_check(max_allele, "i1")

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_count_alleles(gt, max_allele)


def genotype_array_to_allele_counts(gt, max_allele):
    """TODO"""

    # check arguments
    gt = genotype_array_check(gt)
    max_allele = int_check(max_allele, "i1")

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_to_allele_counts(gt, max_allele)


# genotype array
# TODO is_call
# TODO to_n_ref
# TODO to_n_alt
# TODO to_allele_counts
# TODO to_haplotypes
# TODO __repr__
# TODO display
# TODO map_alleles
# TODO max


VCF_FIXED_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL"]


def get_variants_array_names(variants, names=None):

    # discover array keys
    all_names = sorted(variants)

    if names is None:
        # return all names, reordering so VCF fixed fields are first
        names = [k for k in VCF_FIXED_FIELDS if k in all_names]
        names += [k for k in all_names if k.startswith("FILTER")]
        names += [k for k in all_names if k not in names]

    else:
        # check requested keys are present in data
        for n in names:
            if n not in all_names:
                raise ValueError  # TODO message

    return names


def variants_to_dataframe(variants, columns=None):

    # check variants argument
    if not isinstance(variants, Mapping):
        raise TypeError  # TODO message

    # check columns argument
    if columns is not None:
        if not isinstance(columns, (list, tuple)):
            raise TypeError  # TODO message
        if any(not isinstance(k, str) for k in columns):
            raise TypeError  # TODO message

    # determine array keys to build the dataframe from
    columns = get_variants_array_names(variants, names=columns)
    assert len(columns) > 0

    # peek at one of the arrays to determine dispatch path
    a = variants[columns[0]]
    a = array_check(a)

    # dispatch
    methods = get_methods(a)
    return methods.variants_to_dataframe(variants, columns=columns)


class Selection(Mapping):
    """TODO"""

    def __init__(self, inner, fn, *args, **kwargs):
        self.inner = inner
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, item):
        return self.fn(self.inner[item], *self.args, **self.kwargs)

    def __contains__(self, item):
        return item in self.inner

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        return iter(self.inner)

    def keys(self):
        return self.inner.keys()


def select_slice(o, start=None, stop=None, step=None, axis=0):
    """TODO"""

    # deal with groups
    if isinstance(o, Mapping):
        return Selection(o, select_slice, start=start, stop=stop, step=step, axis=axis)

    # deal with arrays
    a = array_check(o)

    # construct full selection for all array dimensions
    sel = tuple(
        slice(start, stop, step) if i == axis else slice(None) for i in range(a.ndim)
    )

    # no need to dispatch, assume common array API
    return a[sel]


def select_indices(o, indices, axis=0):
    """TODO"""

    # deal with groups
    if isinstance(o, Mapping):
        return Selection(o, select_indices, indices=indices, axis=axis)

    # deal with arrays
    a = array_check(o)

    # dispatch
    methods = get_methods(a)
    return methods.take(a, indices, axis=axis)


def select_mask(o, mask, axis=0):
    """TODO"""

    # deal with groups
    if isinstance(o, Mapping):
        return Selection(o, select_mask, mask=mask, axis=axis)

    # deal with arrays
    a = array_check(o)

    # dispatch
    methods = get_methods(a)
    return methods.compress(mask, a, axis=axis)


# selections
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


# TODO AlleleCountsArray
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


# TODO GenotypeAlleleCountsArray
# TODO ???
