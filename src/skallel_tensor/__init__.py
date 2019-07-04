# flake8: noqa
from .version import version as __version__


from .api import (
    genotype_tensor_is_called,
    genotype_tensor_is_missing,
    genotype_tensor_is_hom,
    genotype_tensor_is_het,
    genotype_tensor_is_call,
    genotype_tensor_count_alleles,
    genotype_tensor_to_allele_counts,
    genotype_tensor_to_allele_counts_melt,
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
except ImportError:
    pass
