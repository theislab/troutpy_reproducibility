#!/bin/bash
#SBATCH --job-name=som_rcc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=som_rcc.txt
#SBATCH --error=som_rcc_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_updated
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis/rcc_som_multimodal_part2_batch/run_som_clustering.py /lustre/groups/ml01/datasets/projects/2025_sergio_troutpy/rois/rois/xenium_rcc
