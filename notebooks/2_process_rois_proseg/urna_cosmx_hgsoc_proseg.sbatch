#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_hgsoc.txt
#SBATCH --error=pipeline_hgsoc_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois/urna_analysis_pipeline_cosmx_merscope.py /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois_proseg/cosmx_hgsoc

