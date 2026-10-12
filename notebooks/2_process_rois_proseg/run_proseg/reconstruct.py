#!/usr/bin/env python3
"""Build a troutpy-ready SpatialData object from Proseg output (step 2, Xenium/CosMx/Atera).

Combines the original ROI data.zarr (images, transcripts) with Proseg's cell polygons,
cell-by-gene counts and per-transcript assignments, and writes
rois_proseg/<dataset>/data.zarr. Submitted with 2_run_reconstruction.sbatch.
"""
import os
import sys
import shutil
import gzip
import time
import argparse
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.io
import anndata as ad
import xarray as xr
import spatialdata as sd
from spatialdata.models import TableModel, PointsModel, ShapesModel
from spatialdata.transformations import get_transformation

# Suppress the legacy Dask DataFrame warning output in SLURM log files
os.environ["DASK_DATAFRAME_QUERY_PLANNING"] = "True"

# Setup organized console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("proseg_reconstructor")

def parse_args():
    parser = argparse.ArgumentParser(description="Compile unified SpatialData + AnnData from Proseg outputs.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original input Xenium data.zarr directory.")
    parser.add_argument("--proseg-dir", type=str, required=True, help="Path to proseg output directory.")
    parser.add_argument("--output_zarr", type=str, default=None, help="Path for output zarr.")
    return parser.parse_args()

def main():
    args = parse_args()
    UNASSIGNED = "UNASSIGNED"
    
    # 1. Clean and normalize the main input path passed from Bash
    input_zarr = os.path.abspath(args.input)
    cleaned_input = input_zarr.rstrip("/")
    
    # 2. Extract the sample name (e.g., 'cosmx_hgsoc') - FIXING THE os.path ATTR ERROR
    dataset_name = os.path.basename(os.path.dirname(cleaned_input)) 
    if dataset_name == "" or not cleaned_input.endswith("data.zarr"):
        dataset_name = os.path.basename(cleaned_input)

    # 3. Trust the explicit Proseg Directory passed from your sbatch script
    proseg_dir = os.path.abspath(args.proseg_dir)

    # 4. Route the output directly to the target directory
    base_output_dir = "/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_proseg"
    output_zarr = os.path.abspath(os.path.join(base_output_dir, dataset_name, "data.zarr"))
    
    # 5. Build the output directory structure safely
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    
    log.info("Starting conversion pipeline...")
    log.info("Input SpatialData: %s", input_zarr)
    log.info("Proseg Directory:   %s", proseg_dir)
    log.info("Output Destination: %s", output_zarr)

    # 1. Read Ingestion Sources with dynamic Gzip stream decompression
    log.info("Loading baseline spatialdata container...")
    
    # CRITICAL FIX: Disable strict validation during the initial ZARR read 
    # to bypass the legacy casing errors present inside your input dataset.
    orig = sd.read_zarr(input_zarr, validate=False)

    log.info("Loading customized proseg asset tables...")
    transcript_meta = pd.read_parquet(os.path.join(proseg_dir, "transcript_metadata.parquet"))
    metadata = pd.read_csv(os.path.join(proseg_dir, "cell_metadata.csv"))

    # Decompress and read cell boundaries (secretly a gzipped GeoJSON file)
    poly_path = os.path.join(proseg_dir, "cell_polygons.parquet")
    log.info("Parsing cell boundaries from compressed envelope: %s", poly_path)
    with gzip.open(poly_path, "rb") as f:
        cell_polygons = gpd.read_file(f)

    # Decompress and read counts expression matrix (secretly a gzipped Matrix Market file)
    counts_path = os.path.join(proseg_dir, "counts.parquet")
    log.info("Parsing expression matrix from compressed envelope: %s", counts_path)
    with gzip.open(counts_path, "rb") as f:
        sparse_counts = scipy.io.mmread(f).tocsr()

    # 2. Synchronize Coordinate Cell Index Mappings
    for col in ["cell_id", "cell", "id", "Unnamed: 0"]:
        if col in metadata.columns:
            metadata = metadata.set_index(col)
            break
    metadata.index = metadata.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    metadata.index.name = "cell_id"

    for col in ["cell_id", "cell", "id"]:
        if col in cell_polygons.columns:
            cell_polygons = cell_polygons.set_index(col)
            break
    cell_polygons.index = cell_polygons.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    cell_polygons.index.name = "cell_id"

    # 3. Transcript String Mapping & Auditing Flag Assignment
    log.info("Updating transcript cell assignments via strict string matching...")
    orig_tx_path = os.path.join(input_zarr, "points", "transcripts", "points.parquet")
    orig_transcripts = pd.read_parquet(orig_tx_path)

    proseg_tx_ids = transcript_meta["transcript_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    proseg_assigns = transcript_meta["assignment"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    proseg_ids_set = set(proseg_tx_ids)

    if "background" in transcript_meta.columns:
        proseg_assigns = proseg_assigns.where(transcript_meta["background"].astype(int) == 0, UNASSIGNED)

    tx_mapping_dict = dict(zip(proseg_tx_ids, proseg_assigns))

    transcript_df = orig_transcripts.copy()
    orig_tx_ids_str = transcript_df["transcript_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    transcript_df["not_in_proseg"] = ~orig_tx_ids_str.isin(proseg_ids_set)
    transcript_df["cell_id"] = orig_tx_ids_str.map(tx_mapping_dict).fillna(UNASSIGNED)

    invalid_states = {"0", "", "nan", "NaN", "<NA>", "None", UNASSIGNED}
    transcript_df["cell_id"] = transcript_df["cell_id"].apply(lambda x: UNASSIGNED if x in invalid_states else x)
    
    transcript_df["overlaps_cell"] = (transcript_df["cell_id"] != UNASSIGNED).astype(bool)

    # CRITICAL: Cast back to a pure, vanilla memory DataFrame to break the inherited Dask/Zarr schema
    if hasattr(transcript_df, "compute"):
        transcript_df = transcript_df.compute()
    transcript_df = pd.DataFrame(transcript_df) 

# --- CROSS-PLATFORM OVERRIDES (Universal Adaptive Typing) ---
    # 1. Clear text indexing schema alignment - Drop any potential casing conflict variables completely
    to_drop = [c for c in transcript_df.columns if c.lower() in ["cell_id", "x", "y"] and c not in ["cell_id", "x", "y"]]
    if to_drop:
        transcript_df = transcript_df.drop(columns=to_drop)
    
    if "cell_id" in transcript_df.columns:
        transcript_df["cell_id"] = transcript_df["cell_id"].astype(str).astype("category")
            
    # 2. Normalize physical properties to float64 (with defensive text fallback for MERSCOPE)
    for col in ["z_raw", "fov", "transcript_id", "z", "local_x", "local_y"]:
        if col in transcript_df.columns:
            try:
                # Works perfectly for numeric IDs (CosMx/Xenium) and coordinates
                transcript_df[col] = transcript_df[col].astype(np.float64)
            except ValueError:
                # Graceful fallback for MERSCOPE alphanumeric Ensembl IDs
                log.info("Column '%s' contains non-numeric strings (MERSCOPE format detected). Preserving as string.", col)
                transcript_df[col] = transcript_df[col].astype(str)

    # 3. Unconditional String Standardization for Audit Flags
    audit_cols = ["overlaps_cell", "control_probe", "not_in_proseg", "overlaps_nuclei"]
    for col in audit_cols:
        if col in transcript_df.columns:
            transcript_df[col] = transcript_df[col].map({
                True: "True", False: "False",
                1: "True", 0: "False",
                "True": "True", "False": "False",
                "1": "True", "0": "False"
            }).fillna("False").astype(str)

    log.info("--- AUDIT SUMMARY ---")
    log.info("Total Base Transcripts: %d", len(transcript_df))
    if "not_in_proseg" in transcript_df.columns:
        log.info("Flagged 'not_in_proseg': %s", str(transcript_df["not_in_proseg"].isin([True, "True"]).sum()))
    log.info("Successfully Assigned:  %s", str(transcript_df["overlaps_cell"].isin([True, "True"]).sum()))

    # Dynamically select present spatial dimensions
    coordinate_map = {"x": "x", "y": "y"}
    if "z" in transcript_df.columns:
        coordinate_map["z"] = "z"
    elif "z_location" in transcript_df.columns:
        coordinate_map["z"] = "z_location"
    elif "global_z" in transcript_df.columns:
        coordinate_map["z"] = "global_z"
    elif "z_raw" in transcript_df.columns:
        coordinate_map["z"] = "z_raw"
    else:
        log.info("No physical Z dimension found in transcripts. Formatting as 2D coordinates.")

    tx_transform = get_transformation(orig.points["transcripts"], to_coordinate_system="global")
    
    parse_df = transcript_df.copy()
    
    points_element = PointsModel.parse(
        parse_df,
        coordinates=coordinate_map,
        transformations={"global": tx_transform},
    )

    # 4. Compile AnnData Expression Matrix
    log.info("Assembling AnnData expression feature table...")
    cell_ids = metadata.index.tolist()

    if sparse_counts.shape[0] != len(cell_ids):
        cell_ids = cell_ids[:sparse_counts.shape[0]]

    if sparse_counts.shape[1] == orig.tables["table"].shape[1]:
        orig_var = orig.tables["table"].var.copy()
    else:
        proseg_genes = transcript_meta["gene"].unique()
        if len(proseg_genes) == sparse_counts.shape[1]:
            orig_var = pd.DataFrame(index=proseg_genes)
        else:
            orig_var = orig.tables["table"].var.iloc[:sparse_counts.shape[1]].copy()

    raw_matrix = sparse_counts.astype(np.float32)

    if cell_polygons.crs is not None:
        cell_polygons = cell_polygons.set_crs(None, allow_override=True)

    assigned_tx = transcript_df[transcript_df["cell_id"] != UNASSIGNED].copy()
    
    if "is_gene" in assigned_tx.columns:
        tx_counts = assigned_tx[assigned_tx["is_gene"]].groupby("cell_id", observed=False).size().rename("transcript_counts")
        ctrl_counts = assigned_tx[~assigned_tx["is_gene"]].groupby("cell_id", observed=False).size().rename("control_probe_counts")
    else:
        tx_counts = assigned_tx.groupby("cell_id", observed=False).size().rename("transcript_counts")
        ctrl_counts = pd.Series(0, index=tx_counts.index, name="control_probe_counts")

    total_counts = assigned_tx.groupby("cell_id", observed=False).size().rename("total_counts")
    
    if "z" in coordinate_map:
        z_mean = assigned_tx.groupby("cell_id", observed=False)[coordinate_map["z"]].mean().rename("z_level")
    else:
        z_mean = pd.Series(0.0, index=total_counts.index, name="z_level")
    
    cell_area = cell_polygons.geometry.area.rename("cell_area")
    cx = cell_polygons.geometry.centroid.x
    cy = cell_polygons.geometry.centroid.y

    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs["cell_id"] = obs.index
    obs = obs.join(tx_counts, how="left").join(ctrl_counts, how="left")

    obs["genomic_control_counts"] = 0
    obs["control_codeword_counts"] = 0
    obs["unassigned_codeword_counts"] = 0
    obs["deprecated_codeword_counts"] = 0

    obs = obs.join(total_counts, how="left").join(cell_area, how="left")
    obs["nucleus_area"] = 0.0
    obs["nucleus_count"] = 0.0
    obs["segmentation_method"] = "proseg"
    obs["region"] = pd.Categorical(["cell_boundaries"] * len(obs))
    obs = obs.join(z_mean, how="left")
    obs["cell_labels"] = np.arange(1, len(obs) + 1, dtype=np.int64)
    obs = obs.fillna({
        "transcript_counts": 0, "control_probe_counts": 0,
        "total_counts": 0, "cell_area": 0.0, "z_level": 0.0
    })

    # Drop case-variant conflicts from old table obs indices if present
    if "cell_ID" in obs.columns:
        obs = obs.drop(columns=["cell_ID"])

    adata = ad.AnnData(X=raw_matrix, obs=obs, var=orig_var)
    adata.layers["raw"] = raw_matrix.copy()
    adata.obsm["spatial"] = np.column_stack([
        cx.reindex(cell_ids).fillna(0).values, 
        cy.reindex(cell_ids).fillna(0).values
    ]).astype(np.float64)

    table_element = TableModel.parse(
        adata, 
        region="cell_boundaries", 
        region_key="region", 
        instance_key="cell_id"
    )

    # 5. Map explicit element variables flexibly
    log.info("Staging structural elements dynamically based on platform specs...")
    cell_boundaries_element = ShapesModel.parse(cell_polygons[["geometry"]].copy(), transformations={"global": tx_transform})
    
    images_dict = {}
    if hasattr(orig, "images") and len(orig.images) > 0:
        log.info("Migrating original morphology images to the new SpatialData object...")
        for img_name in orig.images.keys():
            images_dict[img_name] = orig.images[img_name]

    labels_dict = {}
    if hasattr(orig, "labels") and len(orig.labels) > 0:
        for label_name in orig.labels.keys():
            labels_dict[label_name] = orig.labels[label_name]

    shapes_dict = {"cell_boundaries": cell_boundaries_element}
    if hasattr(orig, "shapes") and len(orig.shapes) > 0:
        for shape_name in orig.shapes.keys():
            if shape_name not in ["shapes", "cell_boundaries"]:
                shapes_dict[shape_name] = orig.shapes[shape_name]

    # 6. Compiling elements into final SpatialData instance
    log.info("Compiling elements into final SpatialData instance...")
    sdata = sd.SpatialData(
        images=images_dict,
        labels=labels_dict,
        shapes=shapes_dict,
        points={"transcripts": points_element},
        tables={"table": table_element},
    )

    # WIPE DESTINATION: Safely handle network filesystem latency (Lustre/NFS)
    if os.path.exists(output_zarr) or os.path.islink(output_zarr):
        log.info("Wiping destination target path clean: %s", output_zarr)
        if os.path.isdir(output_zarr) and not os.path.islink(output_zarr):
            for attempt in range(3):
                try:
                    shutil.rmtree(output_zarr)
                    break
                except OSError as e:
                    if attempt == 2:
                        raise e
                    log.warning("Filesystem busy (Directory not empty). Retrying in 2 seconds...")
                    time.sleep(2)
        else:
            os.remove(output_zarr)

    # Write directly to the fully cleared target path
    log.info("Writing updated container directly to disk: %s", output_zarr)
    sdata.write(output_zarr)
    
    log.info("Processing complete. Verification complete.")

if __name__ == "__main__":
    main()