"""Region-of-interest (ROI) selection helpers used by ``notebooks/1_roi_selection_notebooks``.

Each dataset is cropped to one ROI that is then processed by the uRNA pipeline; the
ROIs are available on Zenodo (10.5281/zenodo.17446348).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def plot_spatial_with_box_and_scale(
    spatial_coords,
    bb_xmin,
    bb_xmax,
    bb_ymin,
    bb_ymax,
    scale_um=5000,
    pixel_size_um=4.705,
    dot_size=0.1,
    figsize=(7, 7),
    scale_bar_loc="bottom_left",
):
    """
    Plot spatial coordinates with bounding box and a scale bar.

    Parameters
    ----------
    spatial_coords : array-like of shape (n, 2)
        Spatial coordinates (assumed in pixels) to be plotted.
    bb_xmin, bb_xmax, bb_ymin, bb_ymax : float
        Coordinates for the bounding box in micrometers.
    scale_um : float
        Length of the scale bar in micrometers.
    pixel_size_um : float
        Micrometers per pixel.
    dot_size : float
        Size of the dots in the scatter plot.
    figsize : tuple
        Size of the figure.
    scale_bar_loc : str
        Location of the scale bar: 'bottom_left', 'bottom_right', etc. (currently only 'bottom_left' implemented)
    """

    # Convert spatial coordinates to micrometers
    spat_x = [g[0] * pixel_size_um for g in spatial_coords]
    spat_y = [g[1] * pixel_size_um for g in spatial_coords]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(spat_x, spat_y, s=dot_size, c="grey")

    # Draw bounding box
    bb_w = bb_xmax - bb_xmin
    bb_h = bb_ymax - bb_ymin
    rect = patches.Rectangle((bb_xmin, bb_ymin), bb_w, bb_h, linewidth=3, edgecolor="red", facecolor="none")
    ax.add_patch(rect)

    # Add scale bar
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if scale_bar_loc == "bottom_left":
        x_pos = x_min + 0.05 * (x_max - x_min)
        y_pos = y_min + 0.05 * (y_max - y_min)
    else:
        raise NotImplementedError("Only 'bottom_left' scale_bar_loc is implemented.")

    ax.plot([x_pos, x_pos + scale_um], [y_pos, y_pos], color="black", linewidth=3)
    ax.text(x_pos, y_pos - 0.01 * (y_max - y_min), f"{int(scale_um)} µm", fontsize=10, ha="left", va="top")

    # Clean up axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def generate_metadata(
    sdata, sample_id, species, technology, segmentation_method, roi_name, tissue, disease, pixel_size_um=4.705
):
    """
    Generate and save metadata for a SpatialData sample.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object for which to compute metadata.
    sample_id : str
        A unique identifier for the sample.
    output_path : str
        Path to save the metadata CSV file.
    species, technology, segmentation_method, roi_name, tissue, disease : str
        User-defined metadata fields.
    pixel_size_um : float
        Microns per pixel (default is 4.705 for Visium).
    """
    import pandas as pd
    import numpy as np
    import os

    # Compute derived features
    table = sdata["table"]
    total_cells = table.shape[0]
    total_cellular_transcripts = int(table.X.sum())
    total_transcripts = len(sdata["transcripts"].index.compute())
    total_genes = table.shape[1]

    coords = table.obsm["spatial"] * pixel_size_um
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area_profiled = x_range * y_range  # in µm²
    cell_density = total_cells / area_profiled * 1e6  # cells per mm²

    metadata = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "species": species,
                "technology": technology,
                "segmentation_method": segmentation_method,
                "roi_name": roi_name,
                "tissue": tissue,
                "disease": disease,
                "total_cells": total_cells,
                "total_cellular_transcripts": total_cellular_transcripts,
                "proportion_cellular_transcripts": total_cellular_transcripts / total_transcripts,
                "total_transcripts": total_transcripts,
                "total_genes": total_genes,
                "area_profiled_um2": area_profiled,
                "cell_density_per_mm2": cell_density,
            }
        ]
    )

    return metadata


def rechunk_sdata(sdata, chunk_size={"x": 512, "y": 512}):
    """
    Rechunks all image and label layers in a SpatialData object to the specified chunk size.

    Parameters:
        sdata: SpatialData object containing images and labels.
        chunk_size: dict, chunk sizes for each dimension (default is {'x': 512, 'y': 512}).

    Returns:
        None — modifies sdata in-place.
    """
    # Process images
    for image_name, image_tree in sdata.images.data.items():
        print(f"Processing image: {image_name}")
        for scale_name, da in image_tree.items():
            print(f"  Rechunking scale: {scale_name}")
            if hasattr(da, "chunk"):
                print("  Applying chunking...")
                image_tree[scale_name] = da.chunk(chunk_size)

    # Process labels
    for labels_name, labels_tree in sdata.labels.data.items():
        print(f"Processing label: {labels_name}")
        for scale_name, da in labels_tree.items():
            print(f"  Rechunking scale: {scale_name}")
            if hasattr(da, "chunk"):
                print("  Applying chunking...")
                labels_tree[scale_name] = da.chunk(chunk_size)

