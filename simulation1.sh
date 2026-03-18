#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu4,gpu3,gpu6,gpu2,gpu1,cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --job-name=ANSYS
#SBATCH -o ./log/SLURM.%N.%j.out
#SBATCH -e ./log/SLURM.%N.%j.err

set -eo pipefail
export LC_ALL=${LC_ALL:-}
export LANG=${LANG:-C.UTF-8}

mkdir -p ./log

source /etc/profile || true
source /etc/profile.d/modules.sh || true

set -u

echo "=== before module ==="
echo "HOST=$(hostname)"
echo "SHELL=$SHELL"
echo "MODULEPATH=${MODULEPATH:-<empty>}"
type module || true

module --ignore_cache avail 2>&1 | head -n 50 || true
module --ignore_cache load ansys-electronics/v252
module list || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyaedt2026v1

export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/AnsysEM
export ANSYSLMD_LICENSE_FILE=1055@172.16.10.81

which python
python --version

python run_simulation.py