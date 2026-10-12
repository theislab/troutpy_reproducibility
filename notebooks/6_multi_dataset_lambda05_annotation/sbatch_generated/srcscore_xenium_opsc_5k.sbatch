#!/bin/bash
#SBATCH --job-name=srcscore_xenium_opsc_5k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/srcscore_xenium_opsc_5k.txt
#SBATCH --error=logs/srcscore_xenium_opsc_5k_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --exclude=cpusrv16
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/6_multi_dataset_lambda05_annotation

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python recompute_source_score.py xenium_opsc_5k
