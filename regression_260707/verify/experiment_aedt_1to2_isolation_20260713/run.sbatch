#!/bin/bash
#SBATCH --job-name=aedt12iso
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
ARTIFACT="$ROOT/artifact-${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT"
scontrol show job -dd "$SLURM_JOB_ID" > "$ARTIFACT/slurm_job.txt"
env | sort > "$ARTIFACT/environment.txt"
python -V > "$ARTIFACT/versions.txt" 2>&1
python -c 'import ansys.aedt.core as a; print(a.__version__)' >> "$ARTIFACT/versions.txt" 2>&1

set +e
timeout --signal=TERM --kill-after=60s 25m python -u "$ROOT/pilot.py" \
  --artifact-dir "$ARTIFACT" \
  --lmutil /opt/ohpc/pub/Electronics/v242/Linux64/licensingclient/linx64/lmutil \
  --license-server 1055@172.16.10.81 \
  --cylinders 12 \
  --mesh-mm 0.55 \
  --passes 12 \
  --phase-timeout 180 \
  --completion-timeout 900
RC=$?
set -e
printf '%s\n' "$RC" > "$ARTIFACT/pilot_exit_code.txt"
exit "$RC"
