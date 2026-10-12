#!/bin/bash
#SBATCH --job-name=fmt_cosmx_mousebrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=format_cosmx_mousebrain.txt
#SBATCH --error=format_cosmx_mousebrain_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger/format_segger_to_sdata.py cosmx_mousebrain --segger-models-dir /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/segger_out_fixed
