#!/usr/bin/env python3
"""Re-read and re-write a SpatialData zarr store with the currently-active
spatialdata package, to migrate it away from a newer on-disk format that
older spatialdata installs (e.g. troutpy_drvi's spatialdata==0.4.0) cannot
read.

Root cause (xenium_rcc): the store was written with a newer spatialdata that
uses Zarr v3 storage and nests OME-NGFF metadata under an "ome" key inside
zarr.json (spatialdata raster format version "0.3" / ome.version
"0.5-dev-spatialdata"). troutpy_drvi's spatialdata==0.4.0 only registers
RasterFormats "0.1"/"0.2" and its ome-zarr reader expects a flat top-level
"multiscales" key, so it fails with "len(nodes) = 0, expected 1".

troutpy_drvi and sd_convert both pin zarr==2.18.3, which cannot parse Zarr v3
stores at all (fails instantly with PathNotFoundError). Only envs with
zarr>=3 (e.g. conda env "troutpy_updated", zarr 3.2.1 + spatialdata 0.7.3)
can actually open the source. This script must be run from such an env.

On write, we explicitly pin each element type's on-disk format to match what
a known-good troutpy_drvi-compatible store already uses (verified against
rois/xenium_hsheart/data.zarr): images/labels -> RasterFormatV01 ("0.1"),
points -> PointsFormatV01 ("0.1"), shapes -> ShapesFormatV02 ("0.2"), tables
-> TablesFormatV01. Leaving these at the writer's default ("current" format,
still "0.3"/"0.5"-era) would reproduce the same incompatibility.
"""
import argparse
import os
import sys

import spatialdata as sd
from spatialdata._io.format import (
    RasterFormatV01,
    PointsFormatV01,
    ShapesFormatV02,
    TablesFormatV01,
    SpatialDataContainerFormatV01,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=str, help="Path to source data.zarr")
    parser.add_argument("dst", type=str, help="Path to write the reformatted data.zarr")
    args = parser.parse_args()

    print(f"spatialdata version: {sd.__version__}")
    print(f"Reading: {args.src}")
    sdata = sd.read_zarr(args.src)
    print(sdata)

    if os.path.exists(args.dst):
        print(f"Removing existing destination: {args.dst}")
        import shutil
        shutil.rmtree(args.dst)

    print(f"Writing (legacy-compatible format): {args.dst}")
    sdata.write(
        args.dst,
        overwrite=True,
        sdata_formats=[
            SpatialDataContainerFormatV01(),
            RasterFormatV01(),
            PointsFormatV01(),
            ShapesFormatV02(),
            TablesFormatV01(),
        ],
    )
    print("Done.")


if __name__ == "__main__":
    main()
