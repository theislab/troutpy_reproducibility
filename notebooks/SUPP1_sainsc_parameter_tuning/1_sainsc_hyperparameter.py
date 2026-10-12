"""Sainsc hyperparameter sweep on the Xenium 5k mouse brain ROI (ED Fig. 2A-C).

Runs the cell-like uRNA classification (pp.segmentation_free_sainsc +
pp.define_urna_probability) for every combination of bin size (2, 3, 5, 10), background
filter (0.2, 0.4), Gaussian kernel (0, 2, 4) and uRNA probability threshold
(0.4, 0.5, 0.6), saving per-run outputs that 2_sainsc_parameter_analysis_figures.ipynb
compares. Run with 1_sainsc_hyperparameter.sbatch.
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

def run_sainsc_iteration(
    sdata: sd.SpatialData,
    base_transcripts: pd.DataFrame,
    binsize: int,
    background_filter: float,
    gaussian_kernel_key: float,
    p_threshold: float,
    sainsc_threads: int,
    outpath_data: str,
    outpath_figures: str,
    param_suffix: str
) -> None:
    """
    Executes the parameter-dependent portions of the SAiNSC pipeline.
    Ensures outputs match exact downstream nesting and naming conventions.
    """
    # 1. Handle Double-Nested Folder Path Required by Downstream Script
    # Matches: base_figures_dir / param_suffix / param_suffix
    iteration_figures_dir = os.path.join(outpath_figures, param_suffix, param_suffix)
    iteration_data_dir = os.path.join(outpath_data, param_suffix, param_suffix)
    
    os.makedirs(iteration_figures_dir, exist_ok=True)
    os.makedirs(iteration_data_dir, exist_ok=True)

    # Clean reset of the transcripts layer to prevent parameter leakage
    sdata["transcripts"] = sd.models.PointsModel.parse(base_transcripts.copy())

    # 2. Run Segmentation-free SAiNSC
    pp.segmentation_free_sainsc(
        sdata,
        binsize=binsize,
        celltype_key="leiden",
        background_filter=background_filter,
        gaussian_kernel_key=gaussian_kernel_key,
        n_threads=sainsc_threads
    )

    # 3. Bayesian uRNA Assignment
    pp.define_urna_probability(sdata, p_threshold=p_threshold)
    pp.define_urna(sdata, method='sainsc',prob_threshold=p_threshold)

    # 4. Compute and Save Summary Performance Metrics
    transcripts_df = sdata['transcripts'].compute()
    total_transcripts = len(transcripts_df)
    
    prop_seg = (transcripts_df['overlaps_cell'] == True).sum() / total_transcripts
    prop_ext = (transcripts_df['extracellular'] == True).sum() / total_transcripts if 'extracellular' in transcripts_df.columns else 0.0

    summary = pd.DataFrame({
        'binsize_used': [binsize],
        'background_filter': [background_filter],
        'gaussian_kernel_key': [gaussian_kernel_key],
        'p_threshold': [p_threshold],
        'proportion_of_segmented': [prop_seg],
        'proportion_of_extracellular': [prop_ext]
    })
    summary.to_csv(os.path.join(iteration_figures_dir, 'pipeline_summary.csv'), index=False)
    
    # 5. Explicitly Calculate 'extracellular_proportion' to Guarantee Downstream Match
    if 'gene' in transcripts_df.columns and 'extracellular' in transcripts_df.columns:
        gene_counts = transcripts_df.groupby('gene').size()
        extra_counts = transcripts_df[transcripts_df['extracellular'] == True].groupby('gene').size()
        extracellular_prop_series = (extra_counts / gene_counts).fillna(0.0)
        
        # Ensure xrna_metadata exist and add/overwrite the column
        if 'xrna_metadata' in sdata and hasattr(sdata['xrna_metadata'], 'var'):
            sdata['xrna_metadata'].var['extracellular_proportion'] = sdata['xrna_metadata'].var.index.map(extracellular_prop_series).fillna(0.0)
            sdata['xrna_metadata'].var.to_csv(os.path.join(iteration_figures_dir, 'pipeline_xrna_metadata.csv'), index=False)
        else:
            # Fallback if xrna_metadata structural layer isn't initialized yet
            fallback_var = pd.DataFrame({'extracellular_proportion': extracellular_prop_series})
            fallback_var.to_csv(os.path.join(iteration_figures_dir, 'pipeline_xrna_metadata.csv'), index=False)

    print(f"--- Completed: {param_suffix} ---")


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
outpath_base_data = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_msbrain_5k/sainsc_tuning"
outpath_base_figures = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/ms_figures/xenium_msbrain_5k_param"

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

# Run standard structural pipelines
clustering_pipeline(sdata)  # Assumes available globally

try:
    pp.add_morphological_metrics(sdata)
except Exception as e:
    print(f"Morphological metrics skipped: {e}")
    
opt_r = tl.calculate_heuristic_radius_by_cells(sdata, k=10, n_cells=500)
tl.density_similarity(sdata, radius=opt_r, process_all=True, segmentation_key='overlaps_cell')

# Cache the completely preprocessed transcript layer state 
base_transcripts = sdata["transcripts"].compute().copy()
print("Shared preprocessing complete. Moving to hyperparameter tuning grid...\n")

# ==========================================
# PHASE 2: PARAMETER GRID TUNING LOOP
# ==========================================
# Kept strictly as integers to prevent breaking downstream string parsers
binsizes = [2, 3, 5, 10]  
background_filters = [0.2, 0.4]
gaussian_kernel_keys = [0.0, 2.0, 4.0]
p_thresholds = [0.4, 0.5, 0.6]
sainsc_threads = 16

grid_combinations = itertools.product(binsizes, background_filters, gaussian_kernel_keys, p_thresholds)

for b, bgf, gk, pt in grid_combinations:
    # Explicit casting to integer string formatting for parsing safety
    param_label = f"bin{int(b)}_bgf{bgf}_gk{gk}_pt{pt}"
    print(f"Starting Grid Run: {param_label}")
    
    try:
        run_sainsc_iteration(
            sdata=sdata,
            base_transcripts=base_transcripts,
            binsize=int(b),
            background_filter=bgf,
            gaussian_kernel_key=gk,
            p_threshold=pt,
            sainsc_threads=sainsc_threads,
            outpath_data=outpath_base_data,
            outpath_figures=outpath_base_figures,
            param_suffix=param_label
        )
    except Exception as e:
        print(f"ERROR: Execution failed for setup {param_label}. Reason: {e}")
        continue