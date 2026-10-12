#!/bin/bash
#SBATCH --job-name=segger_cosmx_pancreas
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=run_segger_cosmx_pancreas.txt
#SBATCH --error=run_segger_cosmx_pancreas_error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/icb/mostafa.shahhosseini/miniconda3/envs/seggerv2
mkdir -p /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/segger_out_fixed/cosmx_pancreas
segger segment --n-epochs 50 --prediction-scale-factor 1.1 --prediction-mode cell \
  --min-similarity-shift 0.3 --tiling-margin-training 5 --tiling-margin-prediction 5 \
  -i /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_segger_input_fixed/cosmx_pancreas/data.zarr \
  -o /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/segger_out_fixed/cosmx_pancreas
