import dask.array as da
import numpy as np


def genotype_array_check(gt):

    if not isinstance(gt, da.Array):
        raise TypeError(
            "Bad type for genotype array; expected {}, found {}.".format(
                da.Array, type(gt)
            )
        )

    if gt.dtype != np.dtype("i1"):
        raise TypeError(
            "Bad dtype for genotype array; expected {}, found {}.".format(
                np.dtype("i1"), gt.dtype
            )
        )

    if gt.ndim != 3:
        raise ValueError(
            "Bad number of dimensions for genotype array; expected 3 "
            "dimensions, found {}.".format(gt.ndim)
        )
