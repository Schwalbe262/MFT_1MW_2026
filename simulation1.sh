#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu5,gpu4,gpu3,gpu6,gpu2,gpu1,cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --job-name=ANSYS
#SBATCH -o ./log/SLURM.%N.%j.out
#SBATCH -e ./log/SLURM.%N.%j.err

set -euo pipefail

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyaedt2026v1

module load ansys-electronics/v252

export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/AnsysEM
export ANSYSLMD_LICENSE_FILE=1055@172.16.10.81

mkdir -p ./log

echo "HOST=$(hostname)"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "ANSYSEM_ROOT252=$ANSYSEM_ROOT252"
echo "ANSYSLMD_LICENSE_FILE=$ANSYSLMD_LICENSE_FILE"
which python
python --version
python -c "import sys; print(sys.executable)"
python -c "import os; print(os.path.exists('/opt/ohpc/pub/Electronics/v252/AnsysEM/ansysedt'))"

python run_simulation.py