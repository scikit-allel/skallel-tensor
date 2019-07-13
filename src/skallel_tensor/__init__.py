# flake8: noqa
from .version import version as __version__


from .api import (
    genotypes_locate_called,
    genotypes_locate_missing,
    genotypes_locate_hom,
    genotypes_locate_het,
    genotypes_locate_call,
    genotypes_count_alleles,
    genotypes_to_allele_counts,
    genotypes_to_allele_counts_melt,
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
