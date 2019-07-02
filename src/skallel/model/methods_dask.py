import warnings
import dask.array as da
from dask.array import take  # noqa
import dask.dataframe as dd
from skallel.model import methods_numpy


ARRAY_TYPE = da.Array


def quacks_like_hdf5_dataset(a):
    """Duck typing for objects that behave like an HDF5 dataset or Zarr array."""
    return (
        hasattr(a, "ndim")
        and hasattr(a, "dtype")
        and hasattr(a, "shape")
        and hasattr(a, "chunks")
        and len(a.chunks) == len(a.shape) == a.ndim
        and all(isinstance(n, int) for n in a.chunks)
    )


def array_check(a):
    if isinstance(a, da.Array):
        # pass through
        return a
    if quacks_like_hdf5_dataset(a):
        # convert to dask array
        return da.from_array(a)
    raise TypeError


def genotype_array_is_called(gt):
    out = da.map_blocks(methods_numpy.genotype_array_is_called, gt, drop_axis=2)
    return out


def genotype_array_is_missing(gt):
    out = da.map_blocks(methods_numpy.genotype_array_is_missing, gt, drop_axis=2)
    return out


def genotype_array_is_hom(gt):
    out = da.map_blocks(methods_numpy.genotype_array_is_hom, gt, drop_axis=2)
    return out


def genotype_array_is_het(gt):
    out = da.map_blocks(methods_numpy.genotype_array_is_het, gt, drop_axis=2)
    return out


def genotype_array_count_alleles(g, max_allele):

    # determine output chunks - preserve axis 0; change axis 1, axis 2
    chunks = (g.chunks[0], (1,) * len(g.chunks[1]), (max_allele + 1,))

    def f(chunk):
        # compute allele counts for chunk
        ac = methods_numpy.genotype_array_count_alleles(chunk, max_allele)
        # insert extra dimension to allow for reducing
        ac = ac[:, None, :]
        return ac

    # map blocks and reduce via sum
    out = da.map_blocks(f, g, chunks=chunks).sum(axis=1, dtype="i4")

    return out


def variants_to_dataframe(variants, columns):

    # build dataframe
    df_cols = []
    for c in columns:

        # obtain values
        a = variants[c]

        # check array type
        a = array_check(a)

        # check number of dimensions
        if a.ndim == 1:
            df_cols.append(a.to_dask_dataframe(columns=c))
        elif a.ndim == 2:
            # split columns
            df_cols.append(
                a.to_dask_dataframe(
                    columns=["{}_{}".format(c, i + 1) for i in range(a.shape[1])]
                )
            )
        else:
            warnings.warn(
                "Ignoring {!r} because it has an unsupported number of "
                "dimensions.".format(c)
            )

    df = dd.concat(df_cols, axis=1)

    return df
