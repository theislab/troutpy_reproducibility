"""Select and save the ROI of the Atera breast cancer dataset (script version of Atera.ipynb).

Reads the raw Atera (Xenium-format) output, crops the bounding box used in the
manuscript, writes metadata.csv (troutpy_reproducibility.roi_processing.generate_metadata)
and converts the object to troutpy's format (troutpy.pp.xenium_converter) before
saving data.zarr. Run with atera_processing.sbatch.
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


    
from troutpy import pl
from troutpy import tl
from troutpy import pp

from troutpy_reproducibility.roi_processing import *

inpath='/ictstr01/groups/ml01/datasets/projects/2025_spatialdoublets_sergio.salas/breast_cancer_atera'
sdata = spatialdata_io.xenium(inpath,morphology_focus=False)

output_path = "/ictstr01/groups/ml01/datasets/projects/2025_spatialdoublets_sergio.salas/breast_cancer_atera_formatted"


# Visualize using global coordinates
local_coords = sdata["table"].obsm["spatial"]
# In case it's needed, chec transformations
# print(  get_transformation(sdata["nucleus_boundaries"]))
spat = [g[0] for g in sdata["table"].obsm["spatial"]]
spat2 = [g[1] for g in sdata["table"].obsm["spatial"]]
plt.scatter(spat, spat2, s=0.01, c="grey")

# ROI selection
bb_xmin = 20150  # min x
bb_ymin = 9000  # min y
bb_xmax = 30199  # max x
bb_ymax = 15899.9  # max y


cropped_sdata = sdata.query.bounding_box(
    axes=["x", "y"],
    min_coordinate=[bb_xmin, bb_ymin],
    max_coordinate=[bb_xmax, bb_ymax],
    target_coordinate_system="global",
)

# Save metadata
metadata = generate_metadata(
    sdata=cropped_sdata,
    sample_id="atera_breast_cancer",
    species="Homo sapiens",
    technology="Atera",
    segmentation_method="multimodal",
    roi_name="atera_breast",
    tissue="breast",
    disease="breast_cancer",
)

rechunk_sdata(cropped_sdata)


# troutpy converter to xenium
troutpy.pp.xenium_converter(cropped_sdata, unassigned_tag="UNASSIGNED")

# Define output paths
zarr_path = os.path.join(output_path, "data.zarr")
metadata_path = os.path.join(output_path, "metadata.csv")

cropped_sdata.write(zarr_path, overwrite=True)
metadata.to_csv(metadata_path, index=False)