import warnings
import numpy as np
import dask.array as da
import dask.dataframe as dd
from . import methods_numpy


def quacks_like_h5py_dataset(a):
    """Duck typing for objects that behave like an HDF5 dataset or Zarr array.
    """
    return (
        hasattr(a, "ndim")
        and hasattr(a, "dtype")
        and hasattr(a, "shape")
        and hasattr(a, "chunks")
        and len(a.chunks) == len(a.shape) == a.ndim
        and all(isinstance(n, int) for n in a.chunks)
    )


def accepts(a):
    if isinstance(a, np.ndarray):
        return True
    if isinstance(a, da.Array):
        return True
    if quacks_like_h5py_dataset(a):
        return True
    return False


def ensure_dask_array(a):
    if isinstance(a, da.Array):
        # Pass through.
        return a
    else:
        # Convert to dask array.
        return da.from_array(a)


def getitem(a, item):
    a = ensure_dask_array(a)
    return a[item]


def take(a, indices, axis):
    a = ensure_dask_array(a)
    return da.take(a, indices, axis=axis)


def compress(condition, a, axis):
    a = ensure_dask_array(a)
    return da.compress(condition, a, axis=axis)


def concatenate(seq, axis):
    seq = [ensure_dask_array(a) for a in seq]
    return da.concatenate(seq, axis=axis)


def genotype_tensor_is_called(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        methods_numpy.genotype_tensor_is_called, gt, drop_axis=2, dtype=bool
    )
    return out


def genotype_tensor_is_missing(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        methods_numpy.genotype_tensor_is_missing, gt, drop_axis=2, dtype=bool
    )
    return out


def genotype_tensor_is_hom(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        methods_numpy.genotype_tensor_is_hom, gt, drop_axis=2, dtype=bool
    )
    return out


def genotype_tensor_is_het(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        methods_numpy.genotype_tensor_is_het, gt, drop_axis=2, dtype=bool
    )
    return out


def genotype_tensor_is_call(gt, call):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        methods_numpy.genotype_tensor_is_call, gt, call, drop_axis=2, dtype=bool
    )
    return out


def _map_genotype_tensor_count_alleles(chunk, max_allele):

    # Compute allele counts for chunk.
    ac = methods_numpy.genotype_tensor_count_alleles(chunk, max_allele)

    # Insert extra dimension to allow for reducing.
    ac = ac[:, None, :]

    return ac


def genotype_tensor_count_alleles(gt, max_allele):
    gt = ensure_dask_array(gt)

    # Determine output chunks - preserve axis 0; change axis 1, axis 2.
    chunks = (gt.chunks[0], (1,) * len(gt.chunks[1]), (max_allele + 1,))

    # Map blocks and reduce via sum.
    out = da.map_blocks(
        _map_genotype_tensor_count_alleles,
        gt,
        max_allele,
        chunks=chunks,
        dtype="i4",
    ).sum(axis=1, dtype="i4")

    return out


def genotype_tensor_to_allele_counts(gt, max_allele):
    gt = ensure_dask_array(gt)

    # Determine output chunks - preserve axis 0, 1; change axis 2.
    chunks = (gt.chunks[0], gt.chunks[1], (max_allele + 1,))

    # Map blocks.
    out = da.map_blocks(
        methods_numpy.genotype_tensor_to_allele_counts,
        gt,
        max_allele,
        chunks=chunks,
        dtype="i1",
    )

    return out


def genotype_tensor_to_allele_counts_melt(gt, max_allele):
    gt = ensure_dask_array(gt)

    # Determine output chunks - change axis 0; preserve axis 1; drop axis 2.
    dim0_chunks = tuple(np.array(gt.chunks[0]) * (max_allele + 1))
    chunks = (dim0_chunks, gt.chunks[1])

    # Map blocks.
    out = da.map_blocks(
        methods_numpy.genotype_tensor_to_allele_counts_melt,
        gt,
        max_allele,
        chunks=chunks,
        dtype="i1",
        drop_axis=2,
    )

    return out


def variants_to_dataframe(variants, columns):

    # Build dataframe.
    df_cols = []
    for c in columns:

        # Obtain values.
        a = variants[c]

        # Check type.
        a = ensure_dask_array(a)

        # Check number of dimensions.
        if a.ndim == 1:
            df_cols.append(a.to_dask_dataframe(columns=c))
        elif a.ndim == 2:
            # Split columns.
            df_cols.append(
                a.to_dask_dataframe(
                    columns=[
                        "{}_{}".format(c, i + 1) for i in range(a.shape[1])
                    ]
                )
            )
        else:
            warnings.warn(
                "Ignoring {!r} because it has an unsupported number of "
                "dimensions.".format(c)
            )

    df = dd.concat(df_cols, axis=1)

    return df
