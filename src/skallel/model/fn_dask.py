import dask.array as da


from skallel.model import fn_numpy


def genotype_array_is_called(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_called, gt, drop_axis=2)
    return out


def genotype_array_is_missing(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_missing, gt, drop_axis=2)
    return out
