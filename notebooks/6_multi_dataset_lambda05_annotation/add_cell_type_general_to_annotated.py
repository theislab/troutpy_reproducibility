#!/usr/bin/env python3
"""Add cell_type_general to the 3 already fine-annotated datasets (atera_hsbreast,
xenium_hsfemur, xenium_msbrain_5k) by mapping their real `cell_type` labels through
fine_to_coarse_mapping.FINE_TO_COARSE, so all 17 datasets end up with a directly
comparable cell_type_general column.

Operates on each dataset's already-computed lambda=0.5 source-score zarr (a lightweight
column addition, no recompute needed) and writes a new zarr with '_unified' appended --
the source-score zarr is left untouched.

Usage: python add_cell_type_general_to_annotated.py [dataset_name ...]
       (with no args, runs all 3; with args, runs only the named ones)
"""
import sys

import pandas as pd
import spatialdata as sd

from fine_to_coarse_mapping import FINE_TO_COARSE

# (dataset_name, path to its lambda=0.5 source-score zarr)
TARGETS = [
    (
        "atera_hsbreast",
        "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/atera_hsbreast/data_annotated_source_score_lambda0.5.zarr",
    ),
    (
        "xenium_hsfemur",
        "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_hsfemur/data_with_source_score_annotated_source_score_lambda0.5.zarr",
    ),
    (
        "xenium_msbrain_5k",
        "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k/data_annotated_lambda0.5.zarr",
    ),
]


def run():
    only = set(sys.argv[1:])
    targets = [(n, p) for n, p in TARGETS if not only or n in only]
    for name, path in targets:
        print(f"[{name}] loading {path}", flush=True)
        sdata = sd.read_zarr(path)
        adata = sdata["table"]

        mapping = FINE_TO_COARSE[name]
        fine_labels = adata.obs["cell_type"].astype(str)
        unmapped = set(fine_labels.unique()) - set(mapping.keys())
        if unmapped:
            raise KeyError(f"[{name}] fine cell_type labels missing from mapping: {unmapped}")

        adata.obs["cell_type_general"] = pd.Categorical(fine_labels.map(mapping))
        sdata["table"] = adata

        print(f"[{name}] cell_type_general value counts:", flush=True)
        print(adata.obs["cell_type_general"].value_counts(), flush=True)

        out_path = path[: -len(".zarr")] + "_unified.zarr"
        print(f"[{name}] writing {out_path}", flush=True)
        sdata.write(out_path, overwrite=False)
        print(f"[{name}] done.", flush=True)
        print(flush=True)


if __name__ == "__main__":
    run()
