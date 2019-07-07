from collections.abc import Mapping
import numpy as np


def coerce_scalar(i, dtype):
    dtype = np.dtype(dtype)
    if not np.can_cast(i, dtype, casting="safe"):
        raise ValueError
    i = np.array(i, dtype)[()]
    return i


def genotype_tensor_check(gt):

    # Check dtype.
    if gt.dtype != np.dtype("i1"):
        raise TypeError

    # Check number of dimensions.
    if gt.ndim != 3:
        raise ValueError


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
        names = list(names)
        for n in names:
            if n not in all_names:
                raise ValueError

    return names


def expand_slice(start, stop, step, axis, ndim):
    return tuple(
        slice(start, stop, step) if i == axis else slice(None)
        for i in range(ndim)
    )


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
