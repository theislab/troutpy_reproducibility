#!/usr/bin/env python3
"""Production uRNA source score recompute for xenium_msbrain_5k at lambda=0.5,
run on the *decontaminated* expression matrix (troutpy.tl.subtract_local_urna_background,
radius=100, inner_radius=10 -- see tl/decontamination.py) instead of raw counts.

Mirrors recompute_source_score_final_lambda05.py exactly (same lambda, max_dist,
ambient_floor computation, chunking) except .X is replaced with the decontaminated
counts before scoring, so urna_source_score's candidate-cell
gene-presence matching reads the cleaned profile. .layers['raw'] is left untouched.

Writes data_annotated_lambda0.5_decontaminated.zarr next to data_annotated.zarr;
does NOT touch the original file or the existing data_annotated_lambda0.5.zarr.
"""
import os
import sys
from pathlib import Path

import spatialdata as sd

from troutpy import tl

data_dir = Path("/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k")

print(f"Loading {data_dir / 'data_annotated.zarr'} ...", flush=True)
sdata = sd.read_zarr(data_dir / "data_annotated.zarr")

print("=== Decontaminating: subtract_local_urna_background (radius=100, inner_radius=10) ===", flush=True)
tl.subtract_local_urna_background(sdata, radius=100.0, inner_radius=10.0, update_X=True)

transcripts = sdata.points["transcripts"]
min_enrichment = (
    transcripts[transcripts["density_similarity_score"] > 0.5]["enrichment_over_random"].min().compute()
)
print(f"ambient_floor (min_enrichment) = {min_enrichment}", flush=True)

print("=== Computing production source score at lambda=0.5 (on decontaminated .X) ===", flush=True)
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

out_zarr = data_dir / "data_annotated_lambda0.5_decontaminated.zarr"
print(f"Writing {out_zarr} ...", flush=True)
sdata.write(out_zarr, overwrite=False)
print("Done.", flush=True)
