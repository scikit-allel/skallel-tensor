from collections.abc import Mapping
import numpy as np


def coerce_scalar(i, dtype):
    if not np.isscalar(i):
        raise TypeError
    dtype = np.dtype(dtype)
    if not np.can_cast(i, dtype, casting="safe"):
        raise ValueError
    i = np.array(i, dtype)[()]
    return i


def check_array_like(a, dtype=None, kind=None, ndim=None):
    array_attrs = "ndim", "dtype", "shape"
    for k in array_attrs:
        if not hasattr(a, k):
            raise TypeError
    if dtype is not None:
        if isinstance(dtype, set):
            dtype = {np.dtype(t) for t in dtype}
            if a.dtype not in dtype:
                raise TypeError
        elif a.dtype != np.dtype(dtype):
            raise TypeError
    if kind is not None:
        if a.dtype.kind not in kind:
            raise TypeError
    if ndim is not None:
        if isinstance(ndim, set):
            if a.ndim not in ndim:
                raise ValueError
        elif ndim != a.ndim:
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


def check_list_or_tuple(l, item_type=None, optional=False, min_length=0):
    if optional and l is None:
        return
    if not isinstance(l, (list, tuple)):
        raise TypeError
    if item_type is not None:
        if not all(isinstance(v, item_type) for v in l):
            raise TypeError
    if min_length > 0 and len(l) < min_length:
        raise ValueError


def check_mapping(m, key_type=None):
    if not isinstance(m, Mapping):
        raise TypeError
    if key_type is not None:
        if not all(isinstance(k, key_type) for k in m.keys()):
            raise TypeError
