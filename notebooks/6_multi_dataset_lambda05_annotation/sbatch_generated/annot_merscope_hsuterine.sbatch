#!/bin/bash
#SBATCH --job-name=annot_merscope_hsuterine
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/annot_merscope_hsuterine.txt
#SBATCH --error=logs/annot_merscope_hsuterine_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --exclude=cpusrv16
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python annotate_cell_type_general.py merscope_hsuterine
