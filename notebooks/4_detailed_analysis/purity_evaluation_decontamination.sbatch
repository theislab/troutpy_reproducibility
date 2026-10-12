#!/bin/bash
#SBATCH --job-name=purity_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=purity_evaluation_decontamination.txt
#SBATCH --error=purity_evaluation_decontamination_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/icb/sergio.salas/miniconda3/envs/troutpy_updated/bin/python -u purity_evaluation_decontamination.py --levels lineage
