from collections.abc import Mapping
from functools import reduce
import numpy as np
import pandas as pd
from multipledispatch import Dispatcher
from .utils import (
    coerce_scalar,
    get_variants_array_names,
    check_array_like,
    check_list_or_tuple,
    check_mapping,
)


def genotypes_locate_called(gt):
    """Locate non-missing genotype calls.

    Parameters
    ----------
    gt : array_like, int8

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, dtype="i1")
    if gt.ndim == 2:
        return genotypes_2d_locate_called_dispatcher(gt)
    elif gt.ndim == 3:
        return genotypes_3d_locate_called_dispatcher(gt)
    else:
        raise ValueError


genotypes_2d_locate_called_dispatcher = Dispatcher("genotypes_2d_locate_called")
genotypes_3d_locate_called_dispatcher = Dispatcher("genotypes_3d_locate_called")


def genotypes_locate_missing(gt):
    """Locate missing genotype calls.

    Parameters
    ----------
    gt : array_like, int8

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, dtype="i1")
    if gt.ndim == 2:
        return genotypes_2d_locate_missing_dispatcher(gt)
    elif gt.ndim == 3:
        return genotypes_3d_locate_missing_dispatcher(gt)
    else:
        raise ValueError


genotypes_2d_locate_missing_dispatcher = Dispatcher(
    "genotypes_2d_locate_missing"
)
genotypes_3d_locate_missing_dispatcher = Dispatcher(
    "genotypes_3d_locate_missing"
)


def genotypes_locate_hom(gt):
    """Locate homozygous genotype calls.

    Parameters
    ----------
    gt : array_like, int8

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, dtype="i1")
    if gt.ndim == 2:
        return genotypes_2d_locate_hom_dispatcher(gt)
    elif gt.ndim == 3:
        return genotypes_3d_locate_hom_dispatcher(gt)
    else:
        raise ValueError


genotypes_2d_locate_hom_dispatcher = Dispatcher("genotypes_2d_locate_hom")
genotypes_3d_locate_hom_dispatcher = Dispatcher("genotypes_3d_locate_hom")


def genotypes_locate_het(gt):
    """Locate heterozygous genotype calls.

    Parameters
    ----------
    gt : array_like, int8

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, dtype="i1")
    if gt.ndim == 2:
        return genotypes_2d_locate_het_dispatcher(gt)
    elif gt.ndim == 3:
        return genotypes_3d_locate_het_dispatcher(gt)
    else:
        raise ValueError


genotypes_2d_locate_het_dispatcher = Dispatcher("genotypes_2d_locate_het")
genotypes_3d_locate_het_dispatcher = Dispatcher("genotypes_3d_locate_het")


def genotypes_locate_call(gt, *, call):
    """Locate genotypes with the given `call`.

    Parameters
    ----------
    gt : array_like, int8
    call : array_like, int8

    Returns
    -------
    out: array_like, bool, shape (n_variants, n_samples)

    """

    check_array_like(gt, dtype="i1")
    call = np.asarray(call, dtype="i1")
    if call.ndim != 1:
        raise ValueError
    if gt.shape[-1] != call.shape[0]:
        raise ValueError

    if gt.ndim == 2:
        return genotypes_2d_locate_call_dispatcher(gt, call)
    elif gt.ndim == 3:
        return genotypes_3d_locate_call_dispatcher(gt, call)
    else:
        raise ValueError


genotypes_2d_locate_call_dispatcher = Dispatcher("genotypes_2d_locate_call")
genotypes_3d_locate_call_dispatcher = Dispatcher("genotypes_3d_locate_call")


def genotypes_count_alleles(gt, *, max_allele):
    """Count the number of calls for each allele.

    Parameters
    ----------
    gt : array_like, int8, shape
    max_allele : int8

    Returns
    -------
    ac: array_like, int32

    """

    check_array_like(gt, dtype="i1", ndim=3)
    max_allele = coerce_scalar(max_allele, "i1")
    return genotypes_3d_count_alleles_dispatcher(gt, max_allele=max_allele)


genotypes_3d_count_alleles_dispatcher = Dispatcher("genotypes_3d_count_alleles")


def genotypes_to_allele_counts(gt, *, max_allele):
    """Convert genotypes to allele counts.

    Parameters
    ----------
    gt : array_like, int8
    max_allele : int

    Returns
    -------
    gac: array_like, int32

    """

    check_array_like(gt, dtype="i1", ndim=3)
    max_allele = coerce_scalar(max_allele, "i1")
    return genotypes_3d_to_allele_counts_dispatcher(gt, max_allele)


genotypes_3d_to_allele_counts_dispatcher = Dispatcher(
    "genotypes_3d_to_allele_counts"
)


def genotypes_to_allele_counts_melt(gt, *, max_allele):
    """Convert genotypes to allele counts, melting each allele into a
    separate row.

    Parameters
    ----------
    gt : array_like, int8
    max_allele : int

    Returns
    -------
    gac: array_like, int32

    """

    check_array_like(gt, dtype="i1", ndim=3)
    max_allele = coerce_scalar(max_allele, "i1")
    return genotypes_3d_to_allele_counts_melt_dispatcher(gt, max_allele)


genotypes_3d_to_allele_counts_melt_dispatcher = Dispatcher(
    "genotypes_3d_to_allele_counts_melt"
)


# genotype array
# TODO to_haplotypes
# TODO map_alleles


variants_to_dataframe_dispatcher = Dispatcher("variants_to_dataframe")


def variants_to_dataframe(variants, columns=None):

    # Check input types.
    check_mapping(variants, str)
    check_list_or_tuple(columns, str, optional=True)

    # Check requested columns.
    columns = get_variants_array_names(variants, names=columns)

    # Peek at one of the arrays to determine dispatch path, assume all arrays
    # will be of the same type.
    a = variants[columns[0]]

    # Manually dispatch.
    f = variants_to_dataframe_dispatcher.dispatch(type(a))
    if f is None:
        raise NotImplementedError
    return f(variants, columns)


class GroupSelection(Mapping):
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


select_slice_dispatcher = Dispatcher("select_slice")


def select_slice(o, start=None, stop=None, step=None, axis=0):
    """TODO"""

    return select_slice_dispatcher(
        o, start=start, stop=stop, step=step, axis=axis
    )


def group_select_slice(o, start=None, stop=None, step=None, axis=0):
    return GroupSelection(
        o, select_slice, start=start, stop=stop, step=step, axis=axis
    )


select_slice_dispatcher.add((Mapping,), group_select_slice)


select_indices_dispatcher = Dispatcher("select_indices")


def select_indices(o, indices, *, axis=0):
    """TODO"""

    return select_indices_dispatcher(o, indices, axis=axis)


def group_select_indices(o, indices, *, axis=0):
    return GroupSelection(o, select_indices, indices, axis=axis)


select_indices_dispatcher.add((Mapping, object), group_select_indices)


select_mask_dispatcher = Dispatcher("select_mask")


def select_mask(o, mask, *, axis=0):
    """TODO"""

    return select_mask_dispatcher(o, mask, axis=axis)


def group_select_mask(o, mask, *, axis=0):
    return GroupSelection(o, select_mask, mask, axis=axis)


select_mask_dispatcher.add((Mapping, object), group_select_mask)


def select_range(o, index, *, begin=None, end=None, axis=0):

    # Obtain index as a numpy array.
    if isinstance(o, Mapping) and isinstance(index, str):
        # Assume a key.
        index = o[index]
    if not isinstance(index, np.ndarray):
        index = np.asarray(index)

    # Locate slice indices.
    start = stop = None
    if begin is not None:
        start = np.searchsorted(index, begin, side="left")
    if end is not None:
        stop = np.searchsorted(index, end, side="right")

    # Delegate.
    return select_slice(o, start=start, stop=stop, axis=axis)


def select_values(o, index, query, *, axis=0):

    # Obtain index as pandas index.
    if isinstance(o, Mapping) and isinstance(index, str):
        # Assume a key.
        index = o[index]
    if not isinstance(index, pd.Index):
        index = pd.Index(index)

    # Locate indices of requested values.
    indices = index.get_indexer(query)

    # Check no missing values.
    if np.any(indices < 0):
        raise KeyError

    # Delegate.
    return select_indices(o, indices, axis=axis)


concatenate_dispatcher = Dispatcher("concatenate")


def concatenate(seq, *, axis=0):

    # Check inputs.
    check_list_or_tuple(seq, min_length=2)

    # Manually dispatch on type of first object in `seq`.
    o = seq[0]
    f = concatenate_dispatcher.dispatch(type(o))
    if f is None:
        raise NotImplementedError
    return f(seq, axis=axis)


class GroupConcatenation(Mapping):
    def __init__(self, inner, axis, *args, **kwargs):
        self.inner = inner
        self.axis = axis
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, item):
        return concatenate([m[item] for m in self.inner], axis=self.axis)

    def __contains__(self, item):
        return item in self._key_set()

    def __len__(self):
        return len(self._key_set())

    def __iter__(self):
        return self.keys()

    def _key_set(self):
        # Find intersection of keys.
        return reduce(lambda x, y: x & y, [set(m) for m in self.inner])

    def keys(self):
        return iter(sorted(self._key_set()))


def group_concatenate(seq, *, axis=0):
    return GroupConcatenation(seq, axis=axis)


concatenate_dispatcher.add((Mapping,), group_concatenate)


# TODO HaplotypeArray
# TODO locate_called
# TODO locate_missing
# TODO locate_ref
# TODO locate_alt
# TODO locate_call
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
# TODO locate_called
# TODO locate_missing
# TODO locate_hom
# TODO locate_het
# TODO locate_call
# TODO locate_variant
# TODO locate_non_variant
# TODO locate_segregating
# TODO locate_non_segregating
# TODO locate_biallelic
# TODO squeeze_biallelic
# TODO map_alleles
# TODO display


allele_counts_2d_locate_segregating_dispatcher = Dispatcher(
    "allele_counts_locate_segregating"
)


def allele_counts_locate_segregating(ac):
    """TODO"""

    check_array_like(ac, dtype="i4", ndim=2)
    allele_counts_2d_locate_segregating_dispatcher(ac)


# TODO GenotypeAlleleCountsArray
# TODO ???
