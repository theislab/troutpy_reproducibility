#!/bin/bash
#SBATCH --job-name=msbrain5k_lambda05_decontam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=recompute_source_score_final_lambda05_decontaminated.txt
#SBATCH --error=recompute_source_score_final_lambda05_decontaminated_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

/home/icb/sergio.salas/miniconda3/envs/troutpy_updated/bin/python -u recompute_source_score_final_lambda05_decontaminated.py
