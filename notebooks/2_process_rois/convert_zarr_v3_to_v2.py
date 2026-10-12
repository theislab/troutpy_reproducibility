#!/usr/bin/env python3
"""Convert data.zarr from zarr v3 to zarr v2 format."""
import sys
import os
import shutil
import itertools
import numpy as np
import zarr
import numcodecs
from zarr.core.dtype.npy.string import VariableLengthUTF8

SRC = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_rcc/data.zarr"
DST = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_rcc_oldzarr/data.zarr"

CHUNK_COPY_THRESHOLD_MB = 100  # arrays larger than this are copied chunk by chunk


def is_string_type(src_arr):
    return isinstance(src_arr.metadata.data_type, VariableLengthUTF8)


def get_compressor(src_arr):
    for c in src_arr.metadata.codecs:
        name = type(c).__name__
        if "Zstd" in name:
            return numcodecs.Zstd(level=getattr(c, "level", 0))
        elif "GZip" in name or "Gzip" in name:
            return numcodecs.GZip(level=getattr(c, "level", 1))
        elif "Blosc" in name:
            return numcodecs.Blosc()
    return None


def to_object_array(data):
    """Convert numpy StringDType array to object dtype."""
    flat = [str(x) for x in np.asarray(data).flat]
    return np.array(flat, dtype=object).reshape(np.asarray(data).shape)


def copy_array(src_arr, dst_grp, name):
    string_type = is_string_type(src_arr)
    dtype = VariableLengthUTF8() if string_type else src_arr.dtype
    compressor = get_compressor(src_arr)
    shape = src_arr.shape
    chunks = src_arr.chunks

    print(f"    array '{name}' shape={shape} dtype={src_arr.metadata.data_type} chunks={chunks}")

    kwargs = dict(shape=shape, dtype=dtype, compressor=compressor)
    if shape:
        kwargs["chunks"] = chunks

    if name in dst_grp:
        del dst_grp[name]
    dst_arr = dst_grp.create_array(name, **kwargs)
    dst_arr.attrs.update(dict(src_arr.attrs))

    if not shape:
        # scalar
        val = src_arr[...]
        dst_arr[...] = str(val) if string_type else val
        return

    uncompressed_mb = src_arr.nbytes / 1024**2
    if uncompressed_mb > CHUNK_COPY_THRESHOLD_MB:
        # chunk-by-chunk to avoid large memory allocation
        chunk_iters = [range(0, s, c) for s, c in zip(shape, chunks)]
        n_chunks = 1
        for s, c in zip(shape, chunks):
            n_chunks *= -(-s // c)  # ceiling division
        done = 0
        for starts in itertools.product(*chunk_iters):
            slices = tuple(
                slice(st, min(st + c, dim))
                for st, c, dim in zip(starts, chunks, shape)
            )
            data = src_arr[slices]
            dst_arr[slices] = to_object_array(data) if string_type else data
            done += 1
            if done % 100 == 0:
                print(f"      {done}/{n_chunks} chunks", flush=True)
    else:
        data = src_arr[...]
        dst_arr[...] = to_object_array(data) if string_type else data

    print(f"    -> copied")


def copy_group(src_grp, dst_grp, path=""):
    dst_grp.attrs.update(dict(src_grp.attrs))

    for name in src_grp:
        item = src_grp[name]
        child_path = f"{path}/{name}" if path else name

        if isinstance(item, zarr.Group):
            print(f"  group: {child_path}")
            child_dst = dst_grp.require_group(name)
            copy_group(item, child_dst, child_path)
        elif isinstance(item, zarr.Array):
            print(f"  array: {child_path}")
            copy_array(item, dst_grp, name)
        else:
            print(f"  SKIP (unknown type): {child_path} -> {type(item)}")


def main():
    print(f"zarr version: {zarr.__version__}")
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")

    if os.path.exists(DST):
        print(f"Removing existing destination: {DST}")
        shutil.rmtree(DST)

    os.makedirs(DST, exist_ok=False)

    src = zarr.open_group(SRC, mode="r")
    print(f"Source zarr_format: {src.metadata.zarr_format}")

    dst = zarr.open_group(DST, mode="w", zarr_format=2)
    print(f"Dest   zarr_format: {dst.metadata.zarr_format}")

    print("\nCopying zarr tree...")
    copy_group(src, dst)

    print("\nCopying non-zarr files (parquet)...")
    for dirpath, dirnames, filenames in os.walk(SRC):
        for fname in filenames:
            if fname.endswith(".parquet"):
                src_file = os.path.join(dirpath, fname)
                rel = os.path.relpath(src_file, SRC)
                dst_file = os.path.join(DST, rel)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"  copied: {rel}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
