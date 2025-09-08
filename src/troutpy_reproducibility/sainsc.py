import os
import pandas as pd
from spatialdata import SpatialData  # or sd.read_zarr if applicable
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
from .batch_running import *


def sainsc_tunning_pipeline(
    inpath: str,
    outpath_data: str,
    outpath_figures: str,
    binsize: int = 3,
    background_filter: float = 0.4,
    gaussian_kernel_key: float = 2.0,
    sainsc_threads: int = 3,
    percentile_threshold: float = 10,
    **kwargs
) -> None:
    """
    Runs the full extracellular RNA analysis pipeline and saves outputs.

    Parameters
    ----------
    inpath : str
        Path to the folder containing data.zarr and metadata.csv.
    outpath_data : str
        Path to save modified spatialdata object or derived data.
    outpath_figures : str
        Path to save output figures.
    [other parameters omitted for brevity]
    """

    # Create a suffix for folder names based on selected parameters
    param_suffix = (
        f"bin{binsize}_bgf{background_filter}_gk{gaussian_kernel_key}_pt{percentile_threshold}"
    )

    # Define output folders
    data_output_dir = os.path.join(outpath_data, param_suffix)
    figures_output_dir = os.path.join(outpath_figures, param_suffix)
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(figures_output_dir, exist_ok=True)

    # Load data
    zarr_path = os.path.join(inpath, 'data.zarr')
    meta_path = os.path.join(inpath, 'metadata.csv')
    sdata = sd.read_zarr(zarr_path) if hasattr(sd, 'read_zarr') else SpatialData.read(zarr_path)
    metadata = pd.read_csv(meta_path)

    # 1) Clustering
    clustering_pipeline(sdata)

    # 2) Segmentation-free SAiNSC
    troutpy.pp.segmentation_free_sainsc(
        sdata,
        binsize=binsize,
        celltype_key='leiden',
        background_filter=background_filter,
        gaussian_kernel_key=gaussian_kernel_key,
        n_threads=sainsc_threads
    )

    # 3) Histogram of scores
    troutpy.pl.histogram(
        sdata,
        x='cosine_similarity',
        hue='overlaps_cell',
        palette='troutpy',
        save=True,
        figures_path=figures_output_dir
    )

    # 4) Define extracellular
    troutpy.pp.define_extracellular(
        sdata,
        layer='transcripts',
        method='sainsc',
        copy=False,
        percentile_threshold=percentile_threshold
    )

    # 5) Calculate proportions
    prop_seg = (sdata['transcripts']['overlaps_cell'].compute() == True).sum() / \
               len(sdata['transcripts']['extracellular'].compute())
    prop_ext = (sdata['transcripts']['extracellular'].compute() == True).sum() / \
               len(sdata['transcripts']['extracellular'].compute())

    # 6) Enrichment plot
    troutpy.tl.extracellular_enrichment(sdata, gene_key='gene')

    # 7) Save summary
    summary = pd.DataFrame({
        'proportion_of_segmented': [prop_seg],
        'proportion_of_extracellular': [prop_ext]
    })
    summary.to_csv(os.path.join(figures_output_dir, 'pipeline_summary.csv'), index=False)
    sdata['xrna_metadata'].var.to_csv(os.path.join(figures_output_dir, 'pipeline_xrna_metadata.csv'), index=False)

    # (Optional) Save modified sdata object
    # sdata.write(os.path.join(data_output_dir, 'sdata_processed.zarr'))

    print(f"Pipeline completed.nData saved in {data_output_dir}\nFigures saved in {figures_output_dir}")
