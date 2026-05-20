#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --partition=gpu4,gpu3,gpu6,gpu2,gpu1,cpu1
#SBATCH --job-name=NSGA2
#SBATCH -o ./log/SLURM.%N.%j.out
#SBATCH -e ./log/SLURM.%N.%j.err

mkdir -p ./log

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deap_25v1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

srun --cpu-bind=cores NSGA2_260520.py