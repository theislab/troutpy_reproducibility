#!/bin/bash
#SBATCH --job-name=msbrain5k_srcscore_neigh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=source_score_neighborhood_comparison.txt
#SBATCH --error=source_score_neighborhood_comparison_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

# usage: sbatch source_score_neighborhood_comparison.sbatch [--test]
source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
python source_score_neighborhood_comparison.py "$@"
