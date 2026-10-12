#!/bin/bash
#SBATCH --job-name=nb12_seg_consistency
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=12_run_notebook.out
#SBATCH --error=12_run_notebook.err
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/3_analysis_of_results

source $(conda info --base)/etc/profile.d/conda.sh
conda activate my_jupyter_env

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=86000 \
    --ExecutePreprocessor.kernel_name=troutpy_drvi \
    --inplace \
    12_segmentation_method_consistency.ipynb
