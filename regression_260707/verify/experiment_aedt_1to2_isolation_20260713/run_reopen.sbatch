#!/bin/bash
#SBATCH --job-name=aedt12reopen
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --priority=TOP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
umask 077
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load ansys-electronics/v252 2>/dev/null || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pyaedt2026v1
export FLEXLM_TIMEOUT=3000000
export ANSYSEM_FEATURE_FEA_ENV=1
export PILOT_PRIORITY_INTENT=high_over_production

ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
ARTIFACT="$ROOT/reopen-artifact-${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT"
scontrol show job -dd "$SLURM_JOB_ID" > "$ARTIFACT/slurm_job.txt"
python -u "$ROOT/reopen_probe.py" \
  --source-project /gpfs/home1/harry261/slurm_scheduler/experiments/aedt12iso-20260713T033349/artifact-732549/projects/pilot_B.aedt \
  --artifact-dir "$ARTIFACT"
