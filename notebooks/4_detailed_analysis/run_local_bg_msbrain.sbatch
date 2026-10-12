#!/bin/bash
#SBATCH --job-name=local_urna_bg_msbrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=run_local_bg_msbrain.txt
#SBATCH --error=run_local_bg_msbrain_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

/home/icb/sergio.salas/miniconda3/envs/troutpy_updated/bin/python -u /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis/run_local_bg_msbrain.py
