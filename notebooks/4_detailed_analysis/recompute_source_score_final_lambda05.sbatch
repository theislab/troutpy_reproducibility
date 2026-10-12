#!/bin/bash
#SBATCH --job-name=msbrain5k_lambda05_final
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=recompute_source_score_final_lambda05.txt
#SBATCH --error=recompute_source_score_final_lambda05_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis/recompute_source_score_final_lambda05.py
