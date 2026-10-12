#!/usr/bin/env python3
"""Format segger cell-segmentation output into a troutpy-ready SpatialData object.

Merges segger's per-transcript cell assignments (segger_segmentation.parquet)
into the original ROI SpatialData object, rebuilds cell boundary shapes and a
matching AnnData table, and writes the result to <output_dir>/<sample>/data.zarr.

The merge/shapes/table logic (`run_segger_sdata_pipeline`) is copied verbatim
from the tested cell in notebooks/2_process_rois_segger/segger_formatting_to_sdata.ipynb.
"""
import argparse
import os
import sys
import traceback

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely.geometry import MultiPoint
from spatialdata.models import PointsModel, ShapesModel, TableModel

DEFAULT_SEGGER_MODELS_DIR = "/lustre/groups/ml01/workspace/mostafa.shahhosseini/data/troutpy_rois/segger_out/models"
DEFAULT_ROIS_DIR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois"
DEFAULT_OUTPUT_DIR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_segger"


def run_segger_sdata_pipeline(sdata, segger_segmentation, old_adata):
    """Complete wrapper pipeline to extract transcripts, merge with Segger predictions,

    update sdata elements, generate new cell boundary shapes, and rebuild a
    cleanly matched AnnData table object.
    """
    print("--- Step 1: Extracting & Merging Transcripts ---")
    # Pull transcripts out of the dask layer into memory
    transcripts = sdata["transcripts"].compute()

    # CRITICAL FIX: Ensure we keep transcripts index intact by resetting it to a column
    # before merging, or explicitly joining on the correct aligned keys.
    transcripts_df = transcripts.reset_index()
    transcript_orig_index_col = (
        transcripts.index.name if transcripts.index.name else "index"
    )

    # segger's row_index is the transcript's 0-based POSITION in the file it
    # read (row order), not the transcripts table's own index label. These
    # coincide for Xenium/MERSCOPE zarrs (already 0-based) but not for CosMx
    # zarrs, whose index can retain an upstream partition offset (e.g.
    # starting at 5296430 instead of 0) - merging on the label there matches
    # nothing. Merge on true row position instead, via an explicit plain
    # integer column (NOT left_index=True: pandas silently turns a left-join
    # index into a non-unique, non-monotonic float64 index whenever any row
    # goes unmatched, which then breaks dask's per-partition dtype/schema
    # inference on write - a column-to-column merge avoids that entirely).
    segger_segmentation["row_index"] = segger_segmentation["row_index"].astype("int64")
    transcripts_df["_position"] = range(len(transcripts_df))
    df_combined = transcripts_df.merge(
        segger_segmentation,
        left_on="_position",
        right_on="row_index",
        how="left",
    )
    df_combined = df_combined.drop(columns=["_position"])

    # NOTE: deliberately NOT restoring transcript_orig_index_col as the real
    # pandas index here (the original code did). Nothing downstream keys off
    # the index value - it's used purely column-wise (cell_id filters,
    # groupby). Setting a non-monotonic index (CosMx's original index can
    # start at an arbitrary offset like 5296430, with gaps) before
    # PointsModel.parse() corrupts dask's dtype meta inference on write
    # (int64/bool columns silently get declared as double/string in the
    # meta while the real data stays int64/bool, and dask's to_parquet
    # then hard-fails on that mismatch). A clean default RangeIndex avoids
    # the bug entirely and costs nothing since the index isn't used.
    if transcript_orig_index_col in df_combined.columns:
        df_combined = df_combined.drop(columns=[transcript_orig_index_col])
    if "row_index" in df_combined.columns:
        df_combined = df_combined.drop(columns=["row_index"])

    # Rebrand columns to track origin vs update
    df_combined = df_combined.rename(
        columns={"cell_id": "original_cell_id", "segger_cell_id": "cell_id"}
    )

    # Catch remaining ambient noise / graph dropouts
    df_combined["cell_id"] = df_combined["cell_id"].fillna("UNASSIGNED")
    df_combined["overlaps_cell"] = df_combined["cell_id"] != "UNASSIGNED"

    print("\n--- Step 2: Updating SpatialData Points Element ---")
    # Parse points safely using the perfectly preserved index alignment
    updated_points = PointsModel.parse(
        df_combined, coordinates={"x": "x", "y": "y"}
    )
    sdata.points["transcripts"] = updated_points

    print(
        "\n--- Step 3: Extracting Cells & Generating Convex Hulls (Shapes) ---"
    )
    # Separate biological signal from ambient backgrounds
    df_cells_only = df_combined[df_combined["cell_id"] != "UNASSIGNED"]

    polygons = []
    cell_ids = []

    # Draw geometric boundaries around localized transcript clusters
    for c_id, group in df_cells_only.groupby("cell_id"):
        if len(group) >= 3:  # Geometric boundary definition baseline rule
            points = MultiPoint(group[["x", "y"]].values)
            polygons.append(points.convex_hull)
            cell_ids.append(c_id)

    # Convert geometries to GeoDataFrame and push into shapes slot
    gdf_shapes = gpd.GeoDataFrame({"geometry": polygons}, index=cell_ids)
    sdata.shapes["cell_boundaries"] = ShapesModel.parse(gdf_shapes)

    print("\n--- Step 4: Constructing Mirrored AnnData Table ---")
    # Create the cell-by-gene expression profile matrix
    count_matrix = (
        df_cells_only.groupby(["cell_id", "gene"])
        .size()
        .unstack(fill_value=0)
    )

    # Explicitly enforce the original var indexing gene layout order
    common_genes = [
        g for g in old_adata.var_names if g in count_matrix.columns
    ]
    count_matrix = count_matrix.reindex(columns=common_genes, fill_value=0)

    # Initialize the core matrix layout
    adata = ad.AnnData(X=count_matrix.values, dtype="float32")
    adata.obs_names = count_matrix.index.astype(str)
    adata.var_names = count_matrix.columns.astype(str)

    # Bind structural layers and carry over strict manufacturer var specs
    adata.layers["raw"] = adata.X.copy()
    adata.var = old_adata.var.copy().loc[adata.var_names]

    # Calculate transcript spatial centroids for obsm['spatial']
    centroids = df_cells_only.groupby("cell_id")[["x", "y"]].mean()
    centroids = centroids.loc[adata.obs_names]
    adata.obsm["spatial"] = centroids.values

    # Match every exact metadata column requested from your template schema
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["total_counts"] = adata.X.sum(axis=1)
    adata.obs["transcript_counts"] = adata.X.sum(axis=1)

    # Check for non-gene target control sets
    control_features = (
        adata.var_names[adata.var["feature_types"] != "Gene Expression"]
        if "feature_types" in adata.var
        else []
    )
    if len(control_features) > 0:
        adata.obs["control_probe_counts"] = (
            adata[:, control_features].X.sum(axis=1)
        )
    else:
        adata.obs["control_probe_counts"] = 0.0

    adata.obs["genomic_control_counts"] = 0.0
    adata.obs["control_codeword_counts"] = 0.0
    adata.obs["unassigned_codeword_counts"] = 0.0
    adata.obs["deprecated_codeword_counts"] = 0.0

    # Extract area size from the shapes generated in Step 3
    adata.obs["cell_area"] = (
        adata.obs_names.map(gdf_shapes.geometry.area).fillna(0.0)
    )
    adata.obs["nucleus_area"] = 0.0
    adata.obs["nucleus_count"] = 1.0

    # Infrastructure configuration tags
    adata.obs["segmentation_method"] = "segger"
    adata.obs["region"] = "cell_boundaries"
    adata.obs["z_level"] = (
        old_adata.obs["z_level"].iloc[0] if "z_level" in old_adata.obs else 3
    )
    adata.obs["cell_labels"] = adata.obs_names.astype(str)

    print("\n--- Step 5: Replacing & Linking Tables in sdata ---")
    # Clean out the table dictionary maps entirely to break old links
    if "table" in sdata.tables:
        del sdata.tables["table"]

    # Strip existing out-of-date metadata signatures
    if "spatialdata_attrs" in adata.uns:
        del adata.uns["spatialdata_attrs"]

    # Mount the new table layer back into SpatialData
    sdata.table = TableModel.parse(
        adata,
        region="cell_boundaries",
        region_key="region",
        instance_key="cell_id",
    )

    print("\n[SUCCESS] Pipeline Complete. Updated SpatialData layout:")
    return sdata


def extract_spatial_metrics(sdata):
    """Extracts high-level spatial statistics and returns them as a dictionary."""
    metrics = {
        "total_transcripts": 0,
        "pct_assigned": 0.0,
        "num_cells": 0,
        "avg_counts_per_cell": 0.0,
        "avg_cell_area": 0.0,
    }

    if "transcripts" in sdata.points:
        df_tx = sdata["transcripts"].compute()
        metrics["total_transcripts"] = len(df_tx)

        cell_col = "cell_id" if "cell_id" in df_tx.columns else ("segger_cell_id" if "segger_cell_id" in df_tx.columns else None)
        if cell_col:
            assigned_tx = df_tx[~df_tx[cell_col].isin(["UNASSIGNED", "None", None, -1, "-1"])].shape[0]
            metrics["pct_assigned"] = (assigned_tx / metrics["total_transcripts"]) * 100

    try:
        table = sdata.table
        metrics["num_cells"] = table.n_obs

        if "total_counts" in table.obs.columns:
            metrics["avg_counts_per_cell"] = float(table.obs["total_counts"].mean())
        if "cell_area" in table.obs.columns:
            metrics["avg_cell_area"] = float(table.obs["cell_area"].mean())
    except (AttributeError, KeyError):
        pass

    return metrics


def write_metadata(rois_dir, sample, saving_path):
    """Copy metadata.csv into the formatted output folder, forcing
    segmentation_method to 'segger' so the downstream troutpy pipeline does not
    fall back to overwriting overlaps_cell from nucleus-overlap logic.
    """
    src = os.path.join(rois_dir, sample, "metadata.csv")
    if not os.path.exists(src):
        print(f"No metadata.csv found at {src}; skipping metadata copy.")
        return

    metadata = pd.read_csv(src)
    if "segmentation_method" in metadata.columns:
        metadata["segmentation_method"] = "segger"
    dst = os.path.join(saving_path, "metadata.csv")
    metadata.to_csv(dst, index=False)
    print(f"Wrote metadata (segmentation_method=segger) to {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Format segger segmentation output into a troutpy-ready SpatialData zarr store."
    )
    parser.add_argument("sample", type=str, help="Sample folder name (must match a subfolder in both --segger-models-dir and --rois-dir)")
    parser.add_argument("--segger-models-dir", type=str, default=DEFAULT_SEGGER_MODELS_DIR)
    parser.add_argument("--rois-dir", type=str, default=DEFAULT_ROIS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--xenium-zarr",
        type=str,
        default=None,
        help="Override the full path to the source data.zarr (defaults to <rois-dir>/<sample>/data.zarr). "
        "Useful for samples whose original data.zarr needed reformatting for compatibility (e.g. xenium_rcc).",
    )
    args = parser.parse_args()

    sample = args.sample
    segger_path = os.path.join(args.segger_models_dir, sample)
    xenium_path = args.xenium_zarr or os.path.join(args.rois_dir, sample, "data.zarr")
    saving_path = os.path.join(args.output_dir, sample)
    zarr_out = os.path.join(saving_path, "data.zarr")

    try:
        seg_parquet = os.path.join(segger_path, "segger_segmentation.parquet")
        if not os.path.exists(seg_parquet):
            raise FileNotFoundError(f"segger_segmentation.parquet not found in {segger_path}")
        if not os.path.exists(xenium_path):
            raise FileNotFoundError(f"data.zarr not found at {xenium_path}")

        os.makedirs(saving_path, exist_ok=True)

        print(f"[{sample}] Loading segger segmentation from: {seg_parquet}")
        segger_segmentation = pd.read_parquet(seg_parquet)

        print(f"[{sample}] Loading SpatialData from: {xenium_path}")
        sdata = sd.read_zarr(xenium_path)

        old_adata = sdata["table"]
        metrics_before = extract_spatial_metrics(sdata)

        sdata_updated = run_segger_sdata_pipeline(sdata, segger_segmentation, old_adata)

        metrics_after = extract_spatial_metrics(sdata_updated)

        print(f"[{sample}] Writing formatted SpatialData to: {zarr_out}")
        sdata_updated.write(zarr_out, overwrite=True)

        write_metadata(args.rois_dir, sample, saving_path)

        print(
            f"SUMMARY sample={sample} status=success "
            f"cells_before={metrics_before['num_cells']} cells_after={metrics_after['num_cells']} "
            f"pct_assigned_before={metrics_before['pct_assigned']:.2f} pct_assigned_after={metrics_after['pct_assigned']:.2f} "
            f"total_transcripts={metrics_after['total_transcripts']}"
        )
    except Exception as e:
        print(f"SUMMARY sample={sample} status=failed error={e!r}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
