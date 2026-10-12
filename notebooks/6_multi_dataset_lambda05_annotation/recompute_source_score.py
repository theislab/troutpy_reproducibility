#!/usr/bin/env python3
"""Production uRNA source score recompute at lambda=0.5 for one dataset.

Mirrors notebooks/4_detailed_analysis/recompute_source_score_final_lambda05.py (the
xenium_msbrain_5k lambda robustness sweep / production rerun), generalized across all
other datasets in dataset_registry.py.

Uses each dataset's own fine `cell_type` where one already exists (atera_hsbreast,
xenium_hsfemur), otherwise the `cell_type_general` column produced by
annotate_cell_type_general.py. Writes a new zarr; the input is never touched.

Usage: python recompute_source_score.py <dataset_name>
"""
import os
import sys

import spatialdata as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import BY_NAME, source_score_input, source_score_output

from troutpy import tl


def run(dataset_name: str):
    ds = BY_NAME[dataset_name]
    in_path, cell_type_col = source_score_input(ds)
    out_path = source_score_output(ds)

    print(f"[{dataset_name}] loading {in_path} (cell_type_col={cell_type_col})", flush=True)
    sdata = sd.read_zarr(in_path)

    obs_cols = list(sdata["table"].obs.columns)
    transcript_cols = list(sdata.points["transcripts"].columns)
    if cell_type_col not in obs_cols:
        raise KeyError(f"{dataset_name}: '{cell_type_col}' missing from table.obs. Found: {obs_cols}")
    for required in ("density_similarity_score", "enrichment_over_random"):
        if required not in transcript_cols:
            raise KeyError(f"{dataset_name}: '{required}' missing from transcripts. Found: {transcript_cols}")

    transcripts = sdata.points["transcripts"]
    min_enrichment = (
        transcripts[transcripts["density_similarity_score"] > 0.5]["enrichment_over_random"].min().compute()
    )
    print(f"[{dataset_name}] ambient_floor (min_enrichment) = {min_enrichment}", flush=True)

    print(f"[{dataset_name}] === Computing production source score at lambda=0.5 ===", flush=True)
    tl.urna_source_score(
        sdata,
        chunk_size=100000,
        cell_type_col=cell_type_col,
        max_dist=200,
        lmbda=0.5,
        max_k=10,
        ambient_floor=min_enrichment,
        signal_threshold=10,
        residual=0.1,
        copy=False,
    )

    print(f"[{dataset_name}] writing {out_path}", flush=True)
    sdata.write(out_path, overwrite=False)
    print(f"[{dataset_name}] done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recompute_source_score.py <dataset_name>")
        sys.exit(1)
    run(sys.argv[1])
