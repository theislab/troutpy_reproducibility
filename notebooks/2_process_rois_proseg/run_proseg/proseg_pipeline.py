"""Run Proseg (v3.1.1) on one ROI (step 1 of the Proseg resegmentation).

Exports the ROI transcripts in the column layout Proseg expects for the given
technology (xenium / cosmx / merscope), runs the proseg binary and keeps its raw
outputs. Submitted with 1_run_proseg.sbatch; all calls are listed in
all_runs_commands.sh. Step 2 (reconstruct.py / reconstruct_merscope.py) turns the
output into a troutpy-ready SpatialData object in rois_proseg/<dataset>/data.zarr.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION VARIABLES
# ==============================================================================
INPUT_ZARR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/cosmx_hgsoc/data.zarr"
OUTPUT_DIR = "/lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_proseg/cosmx_hgsoc"
TECHNOLOGY = "cosmx"  # Options: "xenium", "cosmx", "cosmx-micron", "merscope"
NTHREADS = 16
SKIP_PROSEG = False

PROSEG_BIN = "/home/icb/sergio.salas/.cargo/bin/proseg"

# Configure logging to display cleanly inside your notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
log = logging.getLogger(__name__)

# Added "transcript_id" mapping targets here to preserve alignment
TECH_COLUMN_RENAME = {
    "xenium": {
        "x": "x_location",
        "y": "y_location",
        "z": "z_location",
        "gene": "feature_name",
        "transcript_id": "transcript_id"
    },
    "cosmx": {
        "x": "x_global_px",
        "y": "y_global_px",
        "gene": "target",
        "transcript_id": "transcript_id"
    },
    "cosmx-micron": {
        "x": "x_global_px",
        "y": "y_global_px",
        "gene": "target",
        "transcript_id": "transcript_id"
    },
    "merscope": {
        "x": "global_x",
        "y": "global_y",
        "z": "global_z",
        "gene": "gene",
        "transcript_id": "transcript_id"
    },
}

# ==============================================================================
# 2. RUNNER FUNCTIONS
# ==============================================================================
def _export_transcripts_csv(input_zarr: str, output_dir: str, technology: str) -> str:
    """Read transcripts from spatialdata parquet, rename columns to match
    the technology preset expected by proseg, and write a CSV."""
    parquet_path = os.path.join(input_zarr, "points", "transcripts", "points.parquet")
    log.info("Reading transcripts from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    
    # CRITICAL FIX FOR COSMX: If transcript_id is hiding in the dataframe index, 
    # extract it explicitly into a real column so Proseg can ingest it.
    if "transcript_id" not in df.columns:
        if df.index.name == "transcript_id" or "__null_dask_index__" in str(df.index.name):
            log.info("Moving transcript_id from DataFrame index into an explicit column.")
            df = df.reset_index().rename(columns={df.index.name: "transcript_id"})
        else:
            log.info("No transcript_id column found. Generating an explicit numeric row ID range.")
            df["transcript_id"] = range(len(df))
            
    # ==============================================================================
    # SAFE INTEGER MAPPING FOR PROSEG
    # Create an explicit integer row index for Proseg to consume as a unique token, 
    # while preserving your original string-based 'transcript_id' in the output CSV.
    # ==============================================================================
    df["proseg_numeric_id"] = range(len(df))
    # ==============================================================================

    rename = TECH_COLUMN_RENAME.get(technology, {})
    df = df.rename(columns=rename)
        
    csv_path = os.path.join(output_dir, "transcripts_input.csv")
    
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    log.info("Writing %d transcripts to %s", len(df), csv_path)
    df.to_csv(csv_path, index=False)
    return csv_path


def run_proseg(input_zarr: str, output_dir: str, technology: str, nthreads: int) -> None:
    """Call the proseg binary and verify outputs."""
    orig_tech = technology
    if technology == "cosmx":
        technology = "cosmx-micron"

    transcript_csv = _export_transcripts_csv(input_zarr, output_dir, orig_tech)
    
    # Verify the columns present in the CSV file
    sample_df = pd.read_csv(transcript_csv, nrows=1)
    available_columns = sample_df.columns.tolist()
    log.info("CSV verified columns: %s", available_columns)

    cmd = [
        PROSEG_BIN,
        transcript_csv,
        f"--{technology}",
        "--output-transcript-metadata", os.path.join(output_dir, "transcript_metadata.parquet"),
        "--output-cell-metadata",       os.path.join(output_dir, "cell_metadata.csv"),
        "--output-counts",              os.path.join(output_dir, "counts.parquet"),
        "--output-cell-polygons",       os.path.join(output_dir, "cell_polygons.parquet"),
        "--overwrite",
        "--nthreads", str(nthreads),
    ]

    rename_mapping = TECH_COLUMN_RENAME.get(orig_tech, {})
    
    # Map spatial coordinate dimensions
    if "x" in rename_mapping and rename_mapping["x"] in available_columns:
        cmd += ["--x-column", rename_mapping["x"]]
    if "y" in rename_mapping and rename_mapping["y"] in available_columns:
        cmd += ["--y-column", rename_mapping["y"]]
        
    if "global_z" in available_columns:
        cmd += ["--z-column", "global_z"]
    elif "z" in available_columns:
        cmd += ["--z-column", "z"]
    elif "z_raw" in available_columns:
        cmd += ["--z-column", "z_raw"]

    if "gene" in rename_mapping and rename_mapping["gene"] in available_columns:
        cmd += ["--gene-column", rename_mapping["gene"]]

    # CRITICAL FIX: Pass explicit transcript identifier target to the proseg CLI tool
# CRITICAL FIX: Tell Proseg to read our dummy integer column 
    if "proseg_numeric_id" in available_columns:
        cmd += ["--transcript-id-column", "proseg_numeric_id"]
        
    log.info("Running proseg:\n  %s", " \\\n  ".join(cmd))
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if res.returncode != 0:
        log.error("=== PROSEG CRASHED (Exit Code: %d) ===", res.returncode)
        log.error("STDOUT:\n%s", res.stdout)
        log.error("STDERR:\n%s", res.stderr)
        raise RuntimeError(f"Proseg segmentation binary failed execution with error code {res.returncode}")
        
    log.info("Proseg finished command. Syncing filesystem and waiting for I/O flush...")
    os.system("sync") 
    time.sleep(15)
    log.info("All outputs successfully written to disk.")

# ==============================================================================
# 3. PIPELINE MAIN EXECUTION ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proseg Pipeline Wrapper")
    parser.add_argument("--input", type=str, required=True, help="Path to input Zarr file")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--technology", type=str, default="cosmx", help="Technology profile preset")
    parser.add_argument("--nthreads", type=int, default=16, help="Worker processing thread allocation")
    
    args = parser.parse_args()

    input_zarr_abs = os.path.abspath(args.input)
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    if not SKIP_PROSEG:
        run_proseg(input_zarr_abs, output_dir_abs, args.technology, args.nthreads)
    else:
        log.info("SKIP_PROSEG flag parsed as True. Bypassing execution.")