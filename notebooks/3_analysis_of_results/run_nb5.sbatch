#!/bin/bash
#SBATCH --job-name=nb5_gene_scatters
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=run_nb5.out
#SBATCH --error=run_nb5.err
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --chdir=/ictstr01/home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/3_analysis_of_results
# Execute 5_overall_per_noise_gene_scatters_revisions.ipynb in place (kernel troutpy_drvi)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate my_jupyter_env
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=20000 \
    --ExecutePreprocessor.kernel_name=troutpy_drvi \
    5_overall_per_noise_gene_scatters_revisions.ipynb
