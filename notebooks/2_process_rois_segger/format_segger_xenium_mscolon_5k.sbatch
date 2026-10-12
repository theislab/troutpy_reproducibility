#!/bin/bash
#SBATCH --job-name=fmt_xenium_mscolon_5k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=format_xenium_mscolon_5k.txt
#SBATCH --error=format_xenium_mscolon_5k_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/2_process_rois_segger/format_segger_to_sdata.py xenium_mscolon_5k
