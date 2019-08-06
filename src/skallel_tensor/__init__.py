# flake8: noqa
from .version import version as __version__

# Import the public API.
from .api import (
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_called_allele_counts,
    genotypes_to_missing_allele_counts,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
    genotypes_to_major_allele_counts,
    genotypes_to_haplotypes,
    allele_counts_to_frequencies,
    allele_counts_allelism,
    allele_counts_max_allele,
    variants_to_dataframe,
    select_slice,
    select_indices,
    select_mask,
    select_range,
    select_values,
    concatenate,
)

# Import these modules to ensure that their implementation functions are
# registered with the API for dispatching.
from . import numpy_backend
from . import dask_backend
from . import cuda_backend

__all__ = [
    'genotypes_locate_hom',
    'genotypes_locate_het',
    'genotypes_locate_call',
    'genotypes_count_alleles',
    'genotypes_to_called_allele_counts',
    'genotypes_to_missing_allele_counts',
    'genotypes_to_allele_counts',
    'genotypes_to_allele_counts_melt',
    'genotypes_to_major_allele_counts',
    'genotypes_to_haplotypes',
    'allele_counts_to_frequencies',
    'allele_counts_allelism',
    'allele_counts_max_allele',
    'variants_to_dataframe',
    'select_slice',
    'select_indices',
    'select_mask',
    'select_range',
    'select_values',
    'concatenate',
]
