#!/usr/bin/env python3
"""Production uRNA source score recompute for xenium_msbrain_5k at lambda=0.5.

Chosen from the lambda sensitivity sweep (see recompute_lambda_scores_msbrain.py):
lambda=0.5 sits at (near-)minimal mean entropy, has much lower median entropy than
lambda=0.25, and has already reached the assignment-concordance plateau (98.0% vs
lambda=1.0, vs 94.5% for lambda=0.25) while staying well clear of the instability
seen at lambda=1.0 (entropy rises back to 0.811 there).

Writes data_annotated_lambda0.5.zarr next to data_annotated.zarr; does NOT touch
the original file. This new zarr is the intended input for
5_computing_cell_scores_msbrain.ipynb.
"""
import os
import sys
from pathlib import Path

import spatialdata as sd

from troutpy import tl

data_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")

print(f"Loading {data_dir / 'data_annotated.zarr'} ...", flush=True)
sdata = sd.read_zarr(data_dir / "data_annotated.zarr")

transcripts = sdata.points["transcripts"]
min_enrichment = (
    transcripts[transcripts["density_similarity_score"] > 0.5]["enrichment_over_random"].min().compute()
)
print(f"ambient_floor (min_enrichment) = {min_enrichment}", flush=True)

print("=== Computing production source score at lambda=0.5 ===", flush=True)
tl.urna_source_score(
    sdata,
    chunk_size=100000,
    cell_type_col="cell_type",
    max_dist=200,
    lmbda=0.5,
    max_k=10,
    ambient_floor=min_enrichment,
    signal_threshold=10,
    residual=0.1,
    copy=False,
)

out_zarr = data_dir / "data_annotated_lambda0.5.zarr"
print(f"Writing {out_zarr} ...", flush=True)
sdata.write(out_zarr, overwrite=False)
print("Done.", flush=True)
