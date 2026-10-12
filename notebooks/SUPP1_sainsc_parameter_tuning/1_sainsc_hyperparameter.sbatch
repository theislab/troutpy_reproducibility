#!/bin/bash
#SBATCH --job-name=sainsc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=sainsc_output_log.txt
#SBATCH --error=sainsc_output_error.txt
#SBATCH --nice=10000
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal

# Run your job commands

source $(conda info --base)/etc/profile.d/conda.sh
conda activate troutpy_drvi
python /home/icb/sergio.salas/notebooks/troutpy_reproducibility/troutpy_reproducibility/notebooks/SUPP1_sainsc_parameter_tuning/1_sainsc_hyperparameter.py 

