#!/bin/bash
#SBATCH --job-name=density
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=density_output_log.txt
#SBATCH --error=density_output_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/SUPP1_sainsc_parameter_tuning/3_density_hyperparameter.py 

