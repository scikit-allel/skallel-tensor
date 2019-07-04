import numbers
from collections.abc import Mapping
from functools import reduce
import numpy as np
import pandas as pd
from . import methods_numpy, methods_dask


methods_providers = [methods_numpy, methods_dask]


def get_methods(*args):

    for methods in methods_providers:
        if all(methods.accepts(a) for a in args):
            return methods

    raise TypeError


def int_check(i, dtype=None):

    if not isinstance(i, numbers.Integral):
        raise TypeError
    if dtype is not None:
        if not np.can_cast(i, dtype, casting="safe"):
            raise ValueError
        i = np.array(i, dtype)[()]
    return i


def array_check(a):
    array_attrs = "ndim", "shape", "dtype"
    if not all(hasattr(a, k) for k in array_attrs):
        raise TypeError


def genotype_tensor_check(gt):

    # Check type.
    array_check(gt)

    # Check dtype.
    if gt.dtype != np.dtype("i1"):
        raise TypeError

    # Check number of dimensions.
    if gt.ndim != 3:
        raise ValueError


def genotype_tensor_is_called(gt):

    # Check arguments.
    genotype_tensor_check(gt)

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_is_called(gt)


def genotype_tensor_is_missing(gt):

    # Check arguments.
    genotype_tensor_check(gt)

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_is_missing(gt)


def genotype_tensor_is_hom(gt):

    # Check arguments.
    genotype_tensor_check(gt)

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_is_hom(gt)


def genotype_tensor_is_het(gt):

    # Check arguments.
    genotype_tensor_check(gt)

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_is_het(gt)


def genotype_tensor_is_call(gt, call):

    # Check arguments.
    genotype_tensor_check(gt)
    if not isinstance(call, (list, tuple, np.ndarray)):
        raise TypeError
    for c in call:
        int_check(c, "i1")
    call = np.asarray(call, dtype="i1")
    if call.shape[0] != gt.shape[2]:
        raise ValueError
    call = call.astype("i1")

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_is_call(gt, call)


def genotype_tensor_count_alleles(gt, max_allele):

    # TODO support subpop arg

    # Check arguments.
    genotype_tensor_check(gt)
    max_allele = int_check(max_allele, "i1")

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_count_alleles(gt, max_allele)


def genotype_tensor_to_allele_counts(gt, max_allele):

    # Check arguments.
    genotype_tensor_check(gt)
    max_allele = int_check(max_allele, "i1")

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_to_allele_counts(gt, max_allele)


def genotype_tensor_to_allele_counts_melt(gt, max_allele):

    # Check arguments.
    genotype_tensor_check(gt)
    max_allele = int_check(max_allele, "i1")

    # Dispatch.
    methods = get_methods(gt)
    return methods.genotype_tensor_to_allele_counts_melt(gt, max_allele)


# genotype array
# TODO to_haplotypes
# TODO display
# TODO map_alleles


VCF_FIXED_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL"]


def get_variants_array_names(variants, names=None):

    # Discover array keys.
    all_names = sorted(variants)

    if names is None:
        # Return all names, reordering so VCF fixed fields are first.
        names = [k for k in VCF_FIXED_FIELDS if k in all_names]
        names += [k for k in all_names if k.startswith("FILTER")]
        names += [k for k in all_names if k not in names]

    else:
        # Check requested keys are present in data.
        for n in names:
            if n not in all_names:
                raise ValueError

    return names


def variants_to_dataframe(variants, columns=None):

    # Check variants argument.
    if not isinstance(variants, Mapping):
        raise TypeError

    # Check columns argument.
    if columns is not None:
        if not isinstance(columns, (list, tuple)):
            raise TypeError
        if any(not isinstance(k, str) for k in columns):
            raise TypeError

    # Determine array keys to build the dataframe from.
    columns = get_variants_array_names(variants, names=columns)
    assert len(columns) > 0

    # Peek at one of the arrays to determine dispatch path, assume all arrays
    # will be of the same type.
    a = variants[columns[0]]

    # Dispatch.
    methods = get_methods(a)
    return methods.variants_to_dataframe(variants, columns=columns)


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


def select_slice(o, start=None, stop=None, step=None, axis=0):

    # Deal with groups.
    if isinstance(o, Mapping):
        return GroupSelection(
            o, select_slice, start=start, stop=stop, step=step, axis=axis
        )

    # Deal with arrays.
    array_check(o)

    # Construct full selection for all array dimensions.
    item = tuple(
        slice(start, stop, step) if i == axis else slice(None)
        for i in range(o.ndim)
    )

    # Dispatch.
    methods = get_methods(o)
    return methods.getitem(o, item)


def select_indices(o, indices, axis=0):

    # Check args.
    int_check(axis)

    # Deal with groups.
    if isinstance(o, Mapping):
        return GroupSelection(o, select_indices, indices=indices, axis=axis)

    # Deal with arrays.
    array_check(o)

    # Dispatch.
    methods = get_methods(o)
    return methods.take(o, indices, axis=axis)


def select_mask(o, mask, axis=0):

    # Check args.
    int_check(axis)

    # Deal with groups.
    if isinstance(o, Mapping):
        return GroupSelection(o, select_mask, mask=mask, axis=axis)

    # Deal with arrays.
    array_check(o)

    # Dispatch.
    methods = get_methods(o)
    return methods.compress(mask, o, axis=axis)


def select_range(o, index, begin, end, axis=0):

    # Check args.
    int_check(axis)

    # Obtain index as a numpy array.
    if isinstance(o, Mapping) and isinstance(index, str):
        # Assume a key.
        index = o[index]
    if not isinstance(index, np.ndarray):
        index = np.asarray(index)

    # Locate slice indices.
    start = np.searchsorted(index, begin, side="left")
    stop = np.searchsorted(index, end, side="right")

    # Delegate.
    return select_slice(o, start=start, stop=stop, axis=axis)


def select_values(o, index, query, axis=0):

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


def concatenate(seq, axis=0):

    if not isinstance(seq, (list, tuple)):
        raise TypeError
    if len(seq) < 2:
        raise ValueError

    # What type of thing are we concatenating?
    o = seq[0]

    # Deal with groups.
    if isinstance(o, Mapping):
        return GroupConcatenation(seq, axis=axis)

    # Dispatch.
    methods = get_methods(o)
    return methods.concatenate(seq, axis=axis)


class DictGroup(Mapping):
    def __init__(self, root):
        self._root = root

    def __getitem__(self, item):
        path = item.split("/")
        assert len(path) > 0
        node = self._root
        for p in path:
            node = node[p]
        if isinstance(node, dict):
            # Wrap so we get consistent behaviour.
            node = DictGroup(node)
        return node

    def keys(self):
        return self._root.keys()

    def __len__(self):
        return len(self._root)

    def __iter__(self):
        return iter(self._root)


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
