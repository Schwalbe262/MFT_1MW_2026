#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=CNN_tuning
#SBATCH -o ./log/SLURM.%N.%j.out         # STDOUT
#SBATCH -e ./log/SLURM.%N.%j.err         # STDERR



source ~/miniconda3/etc/profile.d/conda.sh
conda activate lightGBM

# python tuning_wandb_Lmt.py
# python tuning_wandb_Llt.py
# python tuning_wandb_copperloss_Tx.py
# python tuning_wandb_ratio_100M.py
# python tuning_wandb_LT_100M.py
# python tuning_wandb_LR_100M.py
# python tuning_wandb_RT_100M.py
# python tuning_wandb_RR_100M.py
# python tuning_wandb_TT_resonant.py


for i in {1..10}
do
    python tuning_multi_model.py
done