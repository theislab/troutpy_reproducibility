#!/bin/bash
#SBATCH --job-name=compute_missing_morph
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=compute_missing_morphology.slurm.out
#SBATCH --error=compute_missing_morphology.slurm.err
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/3_analysis_of_results/morphology

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python -u compute_missing_morphology.py
