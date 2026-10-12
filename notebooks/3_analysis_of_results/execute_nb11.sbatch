#!/bin/bash
#SBATCH --job-name=execute_nb11
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/execute_nb11.txt
#SBATCH --error=logs/execute_nb11_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility

/home/icb/sergio.salas/miniconda3/envs/my_jupyter_env/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=troutpy_drvi \
  --ExecutePreprocessor.timeout=18000 \
  notebooks/3_analysis_of_results/11_full_rna_classification.ipynb
