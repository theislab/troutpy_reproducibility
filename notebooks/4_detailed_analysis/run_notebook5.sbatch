#!/bin/bash
#SBATCH --job-name=nb5_msbrain_lambda05
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=run_notebook5.txt
#SBATCH --error=run_notebook5_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
jupyter execute --inplace --allow-errors --timeout=3600 \
    /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis/5_computing_cell_scores_msbrain.ipynb
