#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_atera_processing.txt
#SBATCH --error=pipeline_atera_processing_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/1_roi_selection_notebooks/atera_processing.py