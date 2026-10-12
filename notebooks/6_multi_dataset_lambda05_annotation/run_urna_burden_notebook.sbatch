#!/bin/bash
#SBATCH --job-name=urna_burden_nb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/urna_burden_nb.txt
#SBATCH --error=logs/urna_burden_nb_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --exclude=cpusrv16
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
jupyter execute --inplace --timeout=1800 urna_burden_by_celltype.ipynb
