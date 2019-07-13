# flake8: noqa
from .version import version as __version__


from .api import (
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_called_allele_counts,
    genotypes_to_missing_allele_counts,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
    allele_counts_to_frequencies,
    allele_counts_allelism,
    allele_counts_max_allele,
    allele_counts_locate_variant,
    allele_counts_locate_non_variant,
    allele_counts_locate_segregating,
    allele_counts_locate_hom,
    allele_counts_locate_het,
    GroupSelection,
    select_slice,
    select_indices,
    select_mask,
    select_range,
    select_values,
    GroupConcatenation,
    concatenate,
)


from .utils import DictGroup


from . import numpy_backend

try:
    from . import dask_backend
except ImportError:  # pragma: no cover
    pass
