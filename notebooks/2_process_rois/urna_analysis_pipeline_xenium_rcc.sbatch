#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_rcc.txt
#SBATCH --error=pipeline_rcc_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatialdata
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois/urna_analysis_pipeline_xenium_rcc.py /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_rcc

