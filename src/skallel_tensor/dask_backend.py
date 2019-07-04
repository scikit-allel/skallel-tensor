import numbers
import warnings
import numpy as np
import dask.array as da
import dask.dataframe as dd
from . import api, numpy_backend, utils


ARRAY_TYPE = (da.Array,)
try:
    # noinspection PyUnresolvedReferences
    import h5py

    ARRAY_TYPE += (h5py.Dataset,)
except ImportError:
    pass
try:
    # noinspection PyUnresolvedReferences
    import zarr

    ARRAY_TYPE += (zarr.Array,)
except ImportError:
    pass


def ensure_dask_array(a):
    if isinstance(a, da.Array):
        # Pass through.
        return a
    else:
        # Convert to dask array.
        return da.from_array(a)


def select_slice(a, start=None, stop=None, step=None, axis=0):
    a = ensure_dask_array(a)
    item = utils.expand_slice(
        start=start, stop=stop, step=step, axis=axis, ndim=a.ndim
    )
    return a[item]


api.select_slice.add((ARRAY_TYPE,), select_slice)


def select_indices(a, indices, axis=0):
    a = ensure_dask_array(a)
    return da.take(a, indices, axis=axis)


api.select_indices.add((ARRAY_TYPE, np.ndarray), select_indices)


def select_mask(a, mask, axis=0):
    a = ensure_dask_array(a)
    return da.compress(mask, a, axis=axis)


api.select_mask.add((ARRAY_TYPE, np.ndarray), select_mask)


def concatenate(seq, axis=0):
    seq = [ensure_dask_array(a) for a in seq]
    return da.concatenate(seq, axis=axis)


api.concatenate_dispatcher.add((ARRAY_TYPE,), concatenate)


def genotype_tensor_is_called(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        numpy_backend.genotype_tensor_is_called, gt, drop_axis=2, dtype=bool
    )
    return out


api.genotype_tensor_is_called.add((ARRAY_TYPE,), genotype_tensor_is_called)


def genotype_tensor_is_missing(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        numpy_backend.genotype_tensor_is_missing, gt, drop_axis=2, dtype=bool
    )
    return out


api.genotype_tensor_is_missing.add((ARRAY_TYPE,), genotype_tensor_is_missing)


def genotype_tensor_is_hom(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        numpy_backend.genotype_tensor_is_hom, gt, drop_axis=2, dtype=bool
    )
    return out


api.genotype_tensor_is_hom.add((ARRAY_TYPE,), genotype_tensor_is_hom)


def genotype_tensor_is_het(gt):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        numpy_backend.genotype_tensor_is_het, gt, drop_axis=2, dtype=bool
    )
    return out


api.genotype_tensor_is_het.add((ARRAY_TYPE,), genotype_tensor_is_het)


def genotype_tensor_is_call(gt, call):
    gt = ensure_dask_array(gt)
    out = da.map_blocks(
        numpy_backend.genotype_tensor_is_call, gt, call, drop_axis=2, dtype=bool
    )
    return out


api.genotype_tensor_is_call.add(
    (ARRAY_TYPE, np.ndarray), genotype_tensor_is_call
)


def _map_genotype_tensor_count_alleles(chunk, max_allele):

    # Compute allele counts for chunk.
    ac = numpy_backend.genotype_tensor_count_alleles(chunk, max_allele)

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


api.genotype_tensor_count_alleles.add(
    (ARRAY_TYPE, numbers.Integral), genotype_tensor_count_alleles
)


def genotype_tensor_to_allele_counts(gt, max_allele):
    gt = ensure_dask_array(gt)

    # Determine output chunks - preserve axis 0, 1; change axis 2.
    chunks = (gt.chunks[0], gt.chunks[1], (max_allele + 1,))

    # Map blocks.
    out = da.map_blocks(
        numpy_backend.genotype_tensor_to_allele_counts,
        gt,
        max_allele,
        chunks=chunks,
        dtype="i1",
    )

    return out


api.genotype_tensor_to_allele_counts.add(
    (ARRAY_TYPE, numbers.Integral), genotype_tensor_to_allele_counts
)


def genotype_tensor_to_allele_counts_melt(gt, max_allele):
    gt = ensure_dask_array(gt)

    # Determine output chunks - change axis 0; preserve axis 1; drop axis 2.
    dim0_chunks = tuple(np.array(gt.chunks[0]) * (max_allele + 1))
    chunks = (dim0_chunks, gt.chunks[1])

    # Map blocks.
    out = da.map_blocks(
        numpy_backend.genotype_tensor_to_allele_counts_melt,
        gt,
        max_allele,
        chunks=chunks,
        dtype="i1",
        drop_axis=2,
    )

    return out


api.genotype_tensor_to_allele_counts_melt.add(
    (ARRAY_TYPE, numbers.Integral), genotype_tensor_to_allele_counts_melt
)


def variants_to_dataframe(variants, columns=None):

    # Check requested columns.
    columns = utils.get_variants_array_names(variants, names=columns)

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


api.variants_to_dataframe_dispatcher.add((ARRAY_TYPE,), variants_to_dataframe)
