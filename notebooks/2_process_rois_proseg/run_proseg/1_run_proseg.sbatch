#!/bin/bash
#SBATCH --job-name=proseg
#SBATCH --output=/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/proseg_runs/logs/proseg_%j.out
#SBATCH --error=/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/proseg_runs/logs/proseg_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

set -euo pipefail

INPUT_ZARR="${1:-}"
TECHNOLOGY="${2:-xenium}"

if [[ -z "$INPUT_ZARR" ]]; then
    echo "ERROR: input zarr path required."
    exit 1
fi

INPUT_ZARR="$(realpath "$INPUT_ZARR")"
DATASET_NAME="$(basename "$(dirname "$INPUT_ZARR")")"

PROJECT_ROOT="/ictstr01/groups/ml01/datasets/projects/2025_sergio_troutpy/rois"
OUTPUT_DIR="${PROJECT_ROOT}/rois_proseg/${DATASET_NAME}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"  # submit these jobs from notebooks/2_process_rois_proseg/run_proseg/
PYTHON="${HOME}/miniconda3/envs/troutpy_drvi/bin/python"

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Dataset:     $DATASET_NAME"
echo "Technology:  $TECHNOLOGY"
echo "Input:       $INPUT_ZARR"
echo "Output dir:  $OUTPUT_DIR"
echo "Threads:     $SLURM_CPUS_PER_TASK"
echo "============================================"

# Run python pipeline directly—no extra array unpacking needed!
"$PYTHON" "${SCRIPT_DIR}/proseg_pipeline.py" \
    --input       "$INPUT_ZARR" \
    --output-dir  "$OUTPUT_DIR" \
    --technology  "$TECHNOLOGY" \
    --nthreads    "$SLURM_CPUS_PER_TASK"