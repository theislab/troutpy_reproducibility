#!/bin/bash
#SBATCH --job-name=annot_xenium_msbrain_300
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/annot_xenium_msbrain_300.txt
#SBATCH --error=logs/annot_xenium_msbrain_300_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python annotate_cell_type_general.py xenium_msbrain_300
