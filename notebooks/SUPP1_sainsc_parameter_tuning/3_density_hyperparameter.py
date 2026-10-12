"""Density-classification hyperparameter sweep on the Xenium 5k mouse brain ROI.

Runs tl.density_similarity over a grid of radius, ambient floor and probability
threshold and stores each run's transcript-level enrichment classes, analysed in
4_density_hyperparameter_analysis.ipynb. Run with 3_density_hyperparameter.sbatch.
"""
import spatialdata_io
import spatialdata as sd
import troutpy
import scanpy as sc
from spatialdata import SpatialData
import matplotlib.pyplot as plt
from spatialdata.transformations import get_transformation
import matplotlib.patches as patches
import os 
import numpy as np
import sys


from troutpy_reproducibility.batch_running import *
from sainsc import *  # the sainsc package (not troutpy_reproducibility.sainsc)


import os
import pandas as pd
import numpy as np
import spatialdata as sd

def run_density_class_iteration(
    sdata: sd.SpatialData,
    base_transcripts: pd.DataFrame,
    radius: int,
    ambient_floor: float,
    p_threshold: float,
    outpath_data: str,
    outpath_figures: str,
    param_suffix: str
) -> None:
    """
    Executes density_similarity and exports the raw transcript-level 
    enrichment classes formatted to fit seamlessly into the downstream script.
    """
    # Enforce double-nested folder paths required by your downstream script
    iteration_figures_dir = os.path.join(outpath_figures, param_suffix, param_suffix)
    iteration_data_dir = os.path.join(outpath_data, param_suffix, param_suffix)
    
    os.makedirs(iteration_figures_dir, exist_ok=True)
    os.makedirs(iteration_data_dir, exist_ok=True)

    # Clean copy reset to isolate parameter runs cleanly
    sdata["transcripts"] = sd.models.PointsModel.parse(base_transcripts.copy())

    # 1. Run density similarity classification
    tl.density_similarity(
        sdata, 
        radius=float(radius), 
        ambient_floor=ambient_floor,
        prob_density=p_threshold,
        process_all=True, 
        segmentation_key='overlaps_cell'
    )

    # 2. Compute and Save Summary Performance Metrics
    transcripts_df = sdata['transcripts'].compute()
    total_transcripts = len(transcripts_df)
    
    prop_seg = (transcripts_df['overlaps_cell'] == True).sum() / total_transcripts
    prop_ext = (transcripts_df['enrichment_class'].isin(["Background", "Moderate Density", "High Density"])).sum() / total_transcripts

    summary = pd.DataFrame({
        'radius_used': [radius],
        'ambient_floor': [ambient_floor],
        'p_threshold': [p_threshold],
        'proportion_of_segmented': [prop_seg],
        'proportion_of_extracellular': [prop_ext]
    })
    summary.to_csv(os.path.join(iteration_figures_dir, 'pipeline_summary.csv'), index=False)
    
    # 3. Export raw transcript classes mapped to the exact column name your downstream script expects
    # We convert categories to integers/codes or strings. Your downstream script computes cosine_similarity,
    # so we map the string classes to numeric identifiers to make them mathematically comparable via cosine similarity.
    class_mapping = {
        "Background": 0,
        "Moderate Density": 1,
        "High Density": 2,
        "Intracellular": 3
    }
    
    transcript_classes_numeric = transcripts_df['enrichment_class'].map(class_mapping).fillna(-1)
    
    # Pack it into a DataFrame named 'extracellular_proportion' so `xrna_df[['extracellular_proportion']]` hooks it cleanly
    class_export_df = pd.DataFrame({'extracellular_proportion': transcript_classes_numeric})
    class_export_df.to_csv(os.path.join(iteration_figures_dir, 'pipeline_xrna_metadata.csv'), index=False)

    print(f"--- Completed Density Class Run: {param_suffix} ---")

import os
import sys
import itertools
import pandas as pd
from pathlib import Path
import spatialdata as sd
from spatialdata import SpatialData

# Ensure local package path takes precedence

from troutpy import pp
from troutpy import tl
from troutpy import pl

# --- CONFIGURATION PATHS ---
inpath = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k"
outpath_base_data = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k/density_tuning"
outpath_base_figures = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k_param_density"

Path(outpath_base_data).mkdir(parents=True, exist_ok=True)
Path(outpath_base_figures).mkdir(parents=True, exist_ok=True)

# ==========================================
# PHASE 1: SHARED UPFRONT PREPROCESSING (ONCE)
# ==========================================
print("Executing shared upfront preprocessing...")

zarr_path = os.path.join(inpath, "data.zarr")
meta_path = os.path.join(inpath, "metadata.csv")
sdata = sd.read_zarr(zarr_path) if hasattr(sd, "read_zarr") else SpatialData.read(zarr_path)
metadata = pd.read_csv(meta_path)

# Nuclei Conditional Override
if "nuclei" in metadata["segmentation_method"][0]:
    transcripts = sdata["transcripts"].compute()
    transcripts["overlaps_cell"] = transcripts["overlaps_nucleus"].astype(bool)
    sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)
    print("Redefined overlaps_cell using nuclei segmentation.")

# Run structural processing dependencies once
clustering_pipeline(sdata)  # Assumes loaded globally

try:
    pp.add_morphological_metrics(sdata)
except Exception as e:
    print(f"Morphological metrics skipped: {e}")

# Cache clean post-processed transcript state before loop runs
base_transcripts = sdata["transcripts"].compute().copy()
print("Shared setup complete. Moving to explicit radius parameter grid...\n")

# ==========================================
# PHASE 2: PARAMETER GRID TUNING LOOP
# ==========================================
# Tuning parameters
radii = [1, 2, 5, 10]  
ambient_floors = [1.2, 1.5, 1.8]
p_thresholds = [0.4, 0.5, 0.6]

grid_combinations = itertools.product(radii, ambient_floors, p_thresholds)

for r, floor, pt in grid_combinations:
    # Formatted strictly to fool your downstream regex/string slicers into running error-free
    # bin -> radius, bgf -> ambient_floor, gk -> pt, pt -> pt
    param_label = f"bin{int(r)}_bgf{floor}_gk{pt}_pt{pt}"
    print(f"Starting Grid Run: {param_label} (Radius: {r}, Floor: {floor})")
    
    try:
        run_density_class_iteration(
            sdata=sdata,
            base_transcripts=base_transcripts,
            radius=int(r),
            ambient_floor=floor,
            p_threshold=pt,
            outpath_data=outpath_base_data,
            outpath_figures=outpath_base_figures,
            param_suffix=param_label
        )
    except Exception as e:
        print(f"ERROR: Execution failed for setup {param_label}. Reason: {e}")
        continue