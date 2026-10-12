#!/bin/bash
#SBATCH --job-name=unify_atera_refix
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/unify_atera_refix.txt
#SBATCH --error=logs/unify_atera_refix_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --exclude=cpusrv16
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python add_cell_type_general_to_annotated.py atera_hsbreast
