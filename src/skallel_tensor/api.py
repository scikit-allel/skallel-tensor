"""This module defines the public API for the skallel_tensor package.

N.B., to allow for implementations of this API using different types of
arrays, such as Numpy, Zarr or Dask arrays, this module implements a
mechanism for dispatching to different backend functions depending on the
type of the arguments. This dispatching is performed via the multipledispatch
package.

"""

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


def genotypes_locate_hom(gt):
    """Locate homozygous genotype calls.

    Parameters
    ----------
    gt : array_like, int

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, kind="i")
    if gt.ndim == 2:
        return dispatch_genotypes_2d_locate_hom(gt)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_locate_hom(gt)
    else:
        raise ValueError


dispatch_genotypes_2d_locate_hom = Dispatcher("genotypes_2d_locate_hom")
dispatch_genotypes_3d_locate_hom = Dispatcher("genotypes_3d_locate_hom")


def genotypes_locate_het(gt):
    """Locate heterozygous genotype calls.

    Parameters
    ----------
    gt : array_like, int

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, kind="i")
    if gt.ndim == 2:
        return dispatch_genotypes_2d_locate_het(gt)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_locate_het(gt)
    else:
        raise ValueError


dispatch_genotypes_2d_locate_het = Dispatcher("genotypes_2d_locate_het")
dispatch_genotypes_3d_locate_het = Dispatcher("genotypes_3d_locate_het")


def genotypes_locate_call(gt, *, call):
    """Locate genotypes with the given `call`.

    Parameters
    ----------
    gt : array_like, int
    call : array_like, int

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, kind="i")
    call = np.asarray(call, dtype=gt.dtype)
    if call.ndim != 1:
        raise ValueError
    if gt.shape[-1] != call.shape[0]:
        raise ValueError

    if gt.ndim == 2:
        return dispatch_genotypes_2d_locate_call(gt, call=call)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_locate_call(gt, call=call)
    else:
        raise ValueError


dispatch_genotypes_2d_locate_call = Dispatcher("genotypes_2d_locate_call")
dispatch_genotypes_3d_locate_call = Dispatcher("genotypes_3d_locate_call")


def genotypes_count_alleles(gt, *, max_allele):
    """Count the number of calls for each allele.

    Parameters
    ----------
    gt : array_like, int
    max_allele : int8

    Returns
    -------
    ac: array_like, int

    """

    check_array_like(gt, kind="i", ndim=3)
    max_allele = coerce_scalar(max_allele, gt.dtype)
    return dispatch_genotypes_3d_count_alleles(gt, max_allele=max_allele)


dispatch_genotypes_3d_count_alleles = Dispatcher("genotypes_3d_count_alleles")


def genotypes_to_called_allele_counts(gt):
    """Count non-missing alleles within genotype calls.

    Parameters
    ----------
    gt : array_like, int

    Returns
    -------
    out: array_like, int

    """

    check_array_like(gt, kind="i")
    if gt.ndim == 2:
        return dispatch_genotypes_2d_to_called_allele_counts(gt)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_to_called_allele_counts(gt)
    else:
        raise ValueError


dispatch_genotypes_2d_to_called_allele_counts = Dispatcher(
    "genotypes_2d_to_called_allele_counts"
)
dispatch_genotypes_3d_to_called_allele_counts = Dispatcher(
    "genotypes_3d_to_called_allele_counts"
)


def genotypes_to_missing_allele_counts(gt):
    """Count missing alleles within genotype calls.

    Parameters
    ----------
    gt : array_like, int

    Returns
    -------
    out: array_like, bool

    """

    check_array_like(gt, kind="i")
    if gt.ndim == 2:
        return dispatch_genotypes_2d_to_missing_allele_counts(gt)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_to_missing_allele_counts(gt)
    else:
        raise ValueError


dispatch_genotypes_2d_to_missing_allele_counts = Dispatcher(
    "genotypes_2d_to_missing_allele_counts"
)
dispatch_genotypes_3d_to_missing_allele_counts = Dispatcher(
    "genotypes_3d_to_missing_allele_counts"
)


def genotypes_to_allele_counts(gt, *, max_allele):
    """Convert genotypes to allele counts.

    Parameters
    ----------
    gt : array_like, int
    max_allele : int

    Returns
    -------
    ac: array_like, int

    """

    check_array_like(gt, kind="i")
    max_allele = coerce_scalar(max_allele, gt.dtype)
    if gt.ndim == 2:
        return dispatch_genotypes_2d_to_allele_counts(gt, max_allele=max_allele)
    elif gt.ndim == 3:
        return dispatch_genotypes_3d_to_allele_counts(gt, max_allele=max_allele)
    else:
        raise ValueError


dispatch_genotypes_2d_to_allele_counts = Dispatcher(
    "genotypes_2d_to_allele_counts"
)
dispatch_genotypes_3d_to_allele_counts = Dispatcher(
    "genotypes_3d_to_allele_counts"
)


def genotypes_to_allele_counts_melt(gt, *, max_allele):
    """Convert genotypes to allele counts, melting each allele into a
    separate row.

    Parameters
    ----------
    gt : array_like, int
    max_allele : int

    Returns
    -------
    ac: array_like, int

    """

    check_array_like(gt, kind="i", ndim=3)
    max_allele = coerce_scalar(max_allele, gt.dtype)
    return dispatch_genotypes_3d_to_allele_counts_melt(
        gt, max_allele=max_allele
    )


dispatch_genotypes_3d_to_allele_counts_melt = Dispatcher(
    "genotypes_3d_to_allele_counts_melt"
)


def genotypes_to_major_allele_counts(gt, *, max_allele):
    """Convert genotypes to major allele counts.

    Parameters
    ----------
    gt : array_like, int
    max_allele : int

    Returns
    -------
    ac: array_like, int

    """

    check_array_like(gt, kind="i", ndim=3)
    max_allele = coerce_scalar(max_allele, gt.dtype)
    return dispatch_genotypes_3d_to_major_allele_counts(
        gt, max_allele=max_allele
    )


dispatch_genotypes_3d_to_major_allele_counts = Dispatcher(
    "genotypes_3d_to_major_allele_counts"
)


def genotypes_to_haplotypes(gt):
    """TODO

    Parameters
    ----------
    gt : array_like, int

    Returns
    -------
    ht : array_like, int

    """

    check_array_like(gt, kind="i", ndim=3)
    return dispatch_genotypes_3d_to_haplotypes(gt)


dispatch_genotypes_3d_to_haplotypes = Dispatcher("genotypes_3d_to_haplotypes")


# genotypes
# TODO to_haplotypes
# TODO map_alleles


# haplotypes
# TODO count_alleles_called
# TODO to_missing_allele_counts
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


# allele counts
# TODO map_alleles


def allele_counts_to_frequencies(ac):
    """TODO"""

    check_array_like(ac, kind="ui")
    if ac.ndim == 2:
        return dispatch_allele_counts_2d_to_frequencies(ac)
    elif ac.ndim == 3:
        return dispatch_allele_counts_3d_to_frequencies(ac)
    else:
        raise ValueError


dispatch_allele_counts_2d_to_frequencies = Dispatcher(
    "allele_counts_2d_to_frequencies"
)
dispatch_allele_counts_3d_to_frequencies = Dispatcher(
    "allele_counts_3d_to_frequencies"
)


def allele_counts_allelism(ac):
    """TODO"""

    check_array_like(ac, kind="ui")
    if ac.ndim == 2:
        return dispatch_allele_counts_2d_allelism(ac)
    elif ac.ndim == 3:
        return dispatch_allele_counts_3d_allelism(ac)
    else:
        raise ValueError


dispatch_allele_counts_2d_allelism = Dispatcher("allele_counts_2d_allelism")
dispatch_allele_counts_3d_allelism = Dispatcher("allele_counts_3d_allelism")


def allele_counts_max_allele(ac):
    """TODO"""

    check_array_like(ac, kind="ui")
    if ac.ndim == 2:
        return dispatch_allele_counts_2d_max_allele(ac)
    elif ac.ndim == 3:
        return dispatch_allele_counts_3d_max_allele(ac)
    else:
        raise ValueError


dispatch_allele_counts_2d_max_allele = Dispatcher("allele_counts_2d_max_allele")
dispatch_allele_counts_3d_max_allele = Dispatcher("allele_counts_3d_max_allele")


def variants_to_dataframe(variants, *, columns=None):

    # Check input types.
    check_mapping(variants, key_type=str)
    check_list_or_tuple(columns, item_type=str, optional=True)

    # Check requested columns.
    columns = get_variants_array_names(variants, names=columns)

    # Peek at one of the arrays to determine dispatch path, assume all arrays
    # will be of the same type.
    a = variants[columns[0]]

    # Manually dispatch.
    f = dispatch_variants_to_dataframe.dispatch(type(a))
    if f is None:
        raise NotImplementedError
    return f(variants, columns=columns)


dispatch_variants_to_dataframe = Dispatcher("variants_to_dataframe")


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


def select_slice(o, *, start=None, stop=None, step=None, axis=0):
    """TODO"""

    return dispatch_select_slice(
        o, start=start, stop=stop, step=step, axis=axis
    )


dispatch_select_slice = Dispatcher("select_slice")


def group_select_slice(o, *, start=None, stop=None, step=None, axis=0):
    return GroupSelection(
        o, select_slice, start=start, stop=stop, step=step, axis=axis
    )


dispatch_select_slice.add((Mapping,), group_select_slice)


def select_indices(o, indices, *, axis=0):
    """TODO"""

    return dispatch_select_indices(o, indices, axis=axis)


dispatch_select_indices = Dispatcher("select_indices")


def group_select_indices(o, indices, *, axis=0):
    return GroupSelection(o, select_indices, indices, axis=axis)


dispatch_select_indices.add((Mapping, object), group_select_indices)


def select_mask(o, mask, *, axis=0):
    """TODO"""

    return dispatch_select_mask(o, mask, axis=axis)


dispatch_select_mask = Dispatcher("select_mask")


def group_select_mask(o, mask, *, axis=0):
    return GroupSelection(o, select_mask, mask, axis=axis)


dispatch_select_mask.add((Mapping, object), group_select_mask)


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


def concatenate(seq, *, axis=0):
    """Concatenate two or more objects along the given `axis`.

    Parameters
    ----------
    seq : sequence of group or array_like
    axis : int

    Returns
    -------
    out : group or array_like

    """

    # Check inputs.
    check_list_or_tuple(seq, min_length=2)

    # Manually dispatch on type of first object in `seq`.
    o = seq[0]
    f = dispatch_concatenate.dispatch(type(o))
    if f is None:
        raise NotImplementedError
    return f(seq, axis=axis)


dispatch_concatenate = Dispatcher("concatenate")


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


dispatch_concatenate.add((Mapping,), group_concatenate)
