#!/bin/bash
#SBATCH --job-name=msbrain5k_lambda_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=23:00:00
#SBATCH --output=lambda_sweep_msbrain.txt
#SBATCH --error=lambda_sweep_msbrain_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis/recompute_lambda_scores_msbrain.py
