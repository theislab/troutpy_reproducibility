#!/bin/bash
#SBATCH --job-name=nuclei_cyto_uncorrected
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=nuclei_cyto_validation_uncorrected.txt
#SBATCH --error=nuclei_cyto_validation_uncorrected_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

/home/icb/sergio.salas/miniconda3/envs/troutpy_updated/bin/python -u nuclei_cyto_validation.py
