#!/bin/bash
#SBATCH --job-name=xenium_hsputamen_segger
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_segger_xenium_hsputamen.txt
#SBATCH --error=pipeline_segger_xenium_hsputamen_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger/urna_analysis_pipeline.py /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_segger/xenium_hsputamen
