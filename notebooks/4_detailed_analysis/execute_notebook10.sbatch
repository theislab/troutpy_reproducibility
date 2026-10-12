#!/bin/bash
#SBATCH --job-name=exec_notebook10_srcscore_neigh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=execute_notebook10.txt
#SBATCH --error=execute_notebook10_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

/home/icb/sergio.salas/miniconda3/envs/my_jupyter_env/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=troutpy_drvi \
  --ExecutePreprocessor.timeout=3600 \
  10_source_score_neighborhood_comparison.ipynb
