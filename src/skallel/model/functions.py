import numbers
import numpy as np
from . import methods_numpy, methods_dask


methods_providers = [methods_numpy, methods_dask]


def int_check(i, dtype):
    """TODO"""

    if not isinstance(i, numbers.Integral):
        raise TypeError("TODO")
    if not np.can_cast(i, dtype, casting="safe"):
        raise ValueError("TODO")
    return np.array(i, dtype)[()]


def array_check(a):
    """TODO"""

    for methods in methods_providers:
        try:
            return methods.array_check(a)
        except TypeError:
            pass

    raise TypeError("TODO")


def get_methods(a):
    """TODO"""

    for methods in methods_providers:
        if isinstance(a, methods.ARRAY_TYPE):
            return methods


def genotype_array_check(gt):
    """TODO"""

    # check type
    gt = array_check(gt)

    # check dtype
    if gt.dtype != np.dtype("i1"):
        raise TypeError("TODO")

    # check number of dimensions
    if gt.ndim != 3:
        raise ValueError("TODO")

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

    # check arguments
    gt = genotype_array_check(gt)
    max_allele = int_check(max_allele, "i1")

    # dispatch
    methods = get_methods(gt)
    return methods.genotype_array_count_alleles(gt, max_allele)


VCF_FIXED_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL"]


def get_variants_keys(callset, keys=None):

    # discover variants array keys
    all_keys = sorted(callset)

    if "variants" in all_keys:
        # deal with nested mappings
        all_keys = sorted(callset["variants"])
    else:
        # strip "variants/" prefix
        prefix = "variants/"
        n = len(prefix)
        all_keys = [k[n:] for k in all_keys if k.startswith(prefix)]

    if keys is None:
        # return all keys, reordering so VCF fixed fields are first
        keys = [k for k in VCF_FIXED_FIELDS if k in all_keys]
        keys += [k for k in all_keys if k.startswith("FILTER")]
        keys += [k for k in all_keys if k not in keys]

    else:
        # check requested keys are present in data
        for k in keys:
            if k not in all_keys:
                raise ValueError("TODO")

    return keys


def variants_to_dataframe(callset, columns=None, index="POS"):
    # TODO
    pass
