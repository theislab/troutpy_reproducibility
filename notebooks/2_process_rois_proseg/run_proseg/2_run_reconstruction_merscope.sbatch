#!/bin/bash
#SBATCH --job-name=proseg_reconstruct
#SBATCH --output=/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/proseg_runs/logs/reconstruct_%j.out
#SBATCH --error=/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/proseg_runs/logs/reconstruct_%j.err
#SBATCH --cpus-per-task=4        # SpatialData Zarr writing utilizes fewer threads effectively
#SBATCH --mem=180G               # High RAM needed to load complete transcript tables + raster matrices
#SBATCH --time=04:00:00          # Allocation window for full container construction
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
INPUT_ZARR="${1:-}"

if [[ -z "$INPUT_ZARR" ]]; then
    echo "ERROR: input zarr path required."
    echo "Usage: sbatch run_reconstruction.sbatch <input_zarr>"
    exit 1
fi

INPUT_ZARR="$(realpath "$INPUT_ZARR")"

# Derive dataset name from the directory containing the zarr
DATASET_NAME="$(basename "$(dirname "$INPUT_ZARR")")"

PROJECT_ROOT="/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois"
PROSEG_DIR="${PROJECT_ROOT}/rois_proseg/${DATASET_NAME}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"  # submit these jobs from notebooks/2_process_rois_proseg/run_proseg/

# Double-check that proseg outputs actually exist before consuming cluster time
if [[ ! -f "${PROSEG_DIR}/counts.parquet" || ! -f "${PROSEG_DIR}/cell_polygons.parquet" ]]; then
    echo "ERROR: Missing required proseg output files in ${PROSEG_DIR}"
    echo "Please ensure the standalone proseg stage ran completely first."
    exit 1
fi

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Dataset:         $DATASET_NAME"
echo "Input Zarr:      $INPUT_ZARR"
echo "Proseg Data Dir: $PROSEG_DIR"
echo "============================================"

# ── Environment Activation ──────────────────────────────────────────────────
# Source the miniconda profile handler to bring the 'conda' command into bash scope
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

echo "Activating conda context..."
conda activate troutpy_drvi

# Log python location to verify successful activation
echo "Using execution context: $(which python)"

# ── Run the reconstruction script ───────────────────────────────────────────
python "${SCRIPT_DIR}/reconstruct_merscope.py" \
    --input      "$INPUT_ZARR" \
    --proseg-dir "$PROSEG_DIR"

echo "Reconstruction job completed successfully."