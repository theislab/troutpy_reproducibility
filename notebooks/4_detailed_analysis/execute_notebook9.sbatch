#!/bin/bash
#SBATCH --job-name=exec_notebook9_urna_proportion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=execute_notebook9.txt
#SBATCH --error=execute_notebook9_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/4_detailed_analysis

/home/icb/sergio.salas/miniconda3/envs/my_jupyter_env/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=troutpy_updated \
  --ExecutePreprocessor.timeout=3600 \
  9_urna_proportion_corrected_vs_uncorrected.ipynb
