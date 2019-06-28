import dask.array as da


from skallel.model import fn_numpy


def genotype_array_is_called(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_called, gt, drop_axis=2)
    return out


def genotype_array_is_missing(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_missing, gt, drop_axis=2)
    return out


def genotype_array_is_hom(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_hom, gt, drop_axis=2)
    return out


def genotype_array_is_het(gt):
    out = da.map_blocks(fn_numpy.genotype_array_is_het, gt, drop_axis=2)
    return out


def genotype_array_count_alleles(g, max_allele):

    # determine output chunks - preserve axis 0; change axis 1, axis 2
    chunks = (g.chunks[0], (1,) * len(g.chunks[1]), (max_allele + 1,))

    def f(chunk):
        # compute allele counts for chunk
        ac = fn_numpy.genotype_array_count_alleles(chunk, max_allele)
        # insert extra dimension to allow for reducing
        ac = ac[:, None, :]
        return ac

    # map blocks and reduce via sum
    out = da.map_blocks(f, g, chunks=chunks).sum(axis=1, dtype="i4")

    return out
