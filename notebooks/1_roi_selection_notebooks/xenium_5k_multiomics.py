"""Build the multimodal (RNA + protein) Xenium renal cell carcinoma ROI.

Reads the Xenium output, max-projects every protein/morphology channel into one
multichannel image ("multiplex_image"), crops the ROI used in Fig. 4 and saves it
for the RCC pipeline (2_process_rois/urna_analysis_pipeline_xenium_rcc.py).
Requires the newer spatialdata (0.7.x) / spatialdata-io (0.7.x) stack (see Methods).
"""
###

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
import zarr


from troutpy_reproducibility.roi_processing import *

inpath = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/raw_datasets/xenium_multimodal"
import spatialdata as sd
import spatialdata_io
from spatialdata.models import Image2DModel
import dask_image.imread
import os

# 1. Load the base Xenium data (transcripts, boundaries, etc.)
# Set morphology_focus=None to skip the problematic auto-loading of images
sdata = spatialdata_io.xenium(inpath, morphology_focus=None)

import dask.array as da
from spatialdata.models import Image2DModel

# 1. Initialize a list to hold the projected arrays and a list for channel names
projected_images = []
channel_names = []

image_folder = os.path.join(inpath, "morphology_focus")

# Sort files to ensure a consistent channel order
files = sorted([f for f in os.listdir(image_folder) if f.endswith(".ome.tif")])

for file in files:
    print(f"Processing: {file}")
    
    # Extract channel name (e.g., 'cd4')
    name = file.replace(".ome.tif", "").split("_")[-1]
    channel_names.append(name)
    
    # Load image: shape is usually (Z, X, Y) or (C, X, Y)
    img_stack = dask_image.imread.imread(os.path.join(image_folder, file))
    
    # (1) Project all planes using Maximum Intensity Projection (MIP)
    # This collapses the first dimension
    mip = img_stack.max(axis=0)
    projected_images.append(mip)

# (2) Concatenate all channels into a (C, X, Y) matrix
# we use da.stack to keep it as a lazy dask array
full_matrix = da.stack(projected_images, axis=0)

# (3) Parse to SpatialData object with scales
# 'scale_factors' creates the pyramid (1.0, 0.8, 0.6, 0.4, 0.2)
sdata.images["multiplex_image"] = Image2DModel.parse(
    full_matrix,
    dims=("c", "y", "x"),
    c_coords=channel_names,
    # These must be integers. 
    # [1, 2, 4, 8] means: full size, 1/2 size, 1/4 size, 1/8 size
    scale_factors=[2, 2, 2, 2] 
)
print(sdata)

# ROI selection
bb_xmin = 23150  # mix x
bb_ymin = 17500  # min y
bb_xmax = 26199  # max x
bb_ymax = 20000  # max y


import copy

import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel, TableModel

cropped_sdata = sdata.query.bounding_box(
    axes=["x", "y"],
    min_coordinate=[bb_xmin, bb_ymin],
    max_coordinate=[bb_xmax, bb_ymax],
    target_coordinate_system="global",
)

# Save metadata
metadata = generate_metadata(
    sdata=cropped_sdata,
    sample_id="rcc_multimodal",
    species="Homo Sapiens",
    technology="Xenium",
    segmentation_method="multimodal",
    roi_name="rcc_ROI_1",
    tissue="kidney",
    disease="renal_carninoma",
)

rechunk_sdata(cropped_sdata)

def xenium_converter(sdata, copy=False, unassigned_tag="UNASSIGNED"):
    """
    Converts a SpatialData object (sdata) into the Xenium format.

    The modifications performed are as follows:

    Points / Transcripts:
      - In sdata.Points['transcripts'] (a pandas DataFrame):
          * Rename the 'feature_name' column to 'Gene' and convert it to a categorical variable.
          * Create a boolean column 'control_probe' based on 'codeword_category':
                if the value is 'predesigned_gene' -> False;
                otherwise -> True.
          * Cast 'transcript_id' to a string and ensure the values are unique.
          * Cast 'overlaps_nucleus' to boolean.
          * Create a new column 'overlaps_cell' where if 'cell_id' == 'UNASSIGNED' then False, else True.

    Tables:
      - In sdata.Tables['table'] (an AnnData object):
          * Check that the .obs DataFrame contains a 'cell_id' column.
          * Ensure that the .obs index is unique.
          * Create a new layer named 'raw' in table.layers containing a copy of table.X (the raw expression data).

    Images:
      - From sdata.Images['morphology_focus'] (a DataTree of images at multiple resolutions):
          * Extract the highest quality image (scale 'scale0') and store it in sdata.Stainings under the key 'default'.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A SpatialData object with attributes such as Images, Points, and Tables, conforming to the structure detailed in the SpatialData documentation.

    Returns
    -------
    sdata : spatialdata.SpatialData
        The converted SpatialData object with the modifications applied.
    """
    # ----- 1. Modify the Points/transcripts DataFrame -----
    try:
        transcripts = sdata["transcripts"].compute().reset_index(drop=True)
    except (KeyError, AttributeError) as e:
        raise ValueError("SpatialData must contain Points['transcripts'] as a DataFrame.") from e

    # a. Rename 'feature_name' to 'Gene' and convert to categorical
    if "feature_name" in transcripts.columns:
        transcripts = transcripts.rename(columns={"feature_name": "gene"})
        transcripts["gene"] = transcripts["gene"].astype("category")
    else:
        print("Warning: 'feature_name' column not found in Points['transcripts'].")

    # b. Create new boolean column 'control_probe' based on 'codeword_category'
    if "codeword_category" in transcripts.columns:
        transcripts["control_probe"] = transcripts["codeword_category"].apply(lambda x: False if x == "predesigned_gene" else True)
    else:
        print("Warning: 'codeword_category' column not found in Points['transcripts']. Gene name-based control probe assignment will be used.")
        transcripts["control_probe"] = transcripts["gene"].str.startswith(("Neg", "BLANK"))

    # c. Ensure 'transcript_id' is a string and has unique values.
    if "transcript_id" in transcripts.columns:
        transcripts["transcript_id"] = transcripts["transcript_id"].astype(str)
        if transcripts["transcript_id"].duplicated().any():
            raise ValueError("Duplicate values found in 'transcript_id'. They must be unique.")
    else:
        print("Warning: 'transcript_id' column not found in Points['transcripts'].")

    # d. Convert 'overlaps_nucleus' to boolean.
    if "overlaps_nucleus" in transcripts.columns:
        transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus"].astype(bool)
    else:
        print("Warning: 'overlaps_nucleus' column not found in Points['transcripts'].")

    # e. Create new 'overlaps_cell' boolean column based on 'cell_id'.
    if "cell_id" in transcripts.columns:
        transcripts["overlaps_cell"] = transcripts["cell_id"].apply(lambda x: False if x == unassigned_tag else True)
    else:
        print("Warning: 'cell_id' column not found in Points['transcripts']. Cannot create 'overlaps_cell'.")

    # Update the transcripts in Points
    sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)

    # ----- 2. Modify the Tables object -----
    try:
        table = sdata["table"]
    except (KeyError, AttributeError) as e:
        raise ValueError("SpatialData must contain Tables['table'] as an AnnData object.") from e

    # Verify that table.obs contains the 'cell_id' column.
    if "cell_id" not in table.obs.columns:
        raise ValueError("Tables['table'].obs must contain a 'cell_id' column.")
    # Ensure the index is unique.
    if table.obs.index.duplicated().any():
        raise ValueError("Tables['table'].obs index contains duplicates. They must be unique.")
    # Create a new layer 'raw' storing a copy of table.X (raw expression data).
    try:
        table.var["gene_id"] = table.var.index
        table.var.index = table.var["gene_names"]
    except KeyError:
        print("Warning: 'gene_names' column not found in Tables['table']. Using default index.")
    if table.X is None:
        raise ValueError("Tables['table'] must have an expression matrix in .X.")
    if not hasattr(table, "layers") or table.layers is None:
        table.layers = {}
    table.layers["raw"] = table.X.copy()

    # ----- 3. Move Images to Stainings -----
    try:
        morph_focus = sdata["morphology_focus"]
    except (KeyError, AttributeError):
        print("Warning: SpatialData does not have Images['morphology_focus'].")
        morph_focus = None

    if morph_focus is not None:
        # Assume morph_focus is a DataTree (or similar dict-like object) with keys for different scales.
        if "scale0" in morph_focus:
            default_image = morph_focus["scale0"]
        else:
            print("Warning: 'scale0' not found in Images['morphology_focus']. Using first available scale.")
            default_image = next(iter(morph_focus.values()))

        # Create or update the Stainings attribute.
        # if not hasattr(sdata, "Stainings") or sdata.images is None:
        #    sdata.images = {}
        #sdata["default"] = default_image

    return sdata if copy else None

# troutpy converter to xenium
xenium_converter(cropped_sdata, unassigned_tag="UNASSIGNED")


from pathlib import Path

output_path = '/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rcc_multiomics'

# parents=True: creates all missing parent directories
# exist_ok=True: prevents an error if the directory already exists
Path(output_path).mkdir(parents=True, exist_ok=True)

print(f"Directory ready: {output_path}")
# Define output paths
zarr_path = os.path.join(output_path, "data.zarr")
metadata_path = os.path.join(output_path, "metadata.csv")

import shutil
import os

zarr_path = os.path.join(output_path, "data.zarr")

if os.path.exists(zarr_path):
    # This removes the folder entirely so spatialdata can start fresh
    shutil.rmtree(zarr_path) 

cropped_sdata.write(zarr_path,overwrite=True)
metadata.to_csv(metadata_path, index=False)