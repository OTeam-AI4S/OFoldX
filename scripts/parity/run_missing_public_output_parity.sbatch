#!/bin/bash
#SBATCH --job-name=ofoldx_missing_public_parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=artifacts/parity/slurm/%x_%A_%a.out

set -euo pipefail

COMMON_CANDIDATES=(
  "${OFOLDX_ROOT:-}/scripts/parity/common.sh"
  "${SLURM_SUBMIT_DIR:-}/scripts/parity/common.sh"
  "$(dirname "${BASH_SOURCE[0]}")/common.sh"
  "/home/z/ofoldx/scripts/parity/common.sh"
)
for common in "${COMMON_CANDIDATES[@]}"; do
  if [[ -f "$common" ]]; then
    # shellcheck source=scripts/parity/common.sh
    source "$common"
    break
  fi
done
if [[ -z "${OFOLDX_ROOT:-}" ]]; then
  echo "could not locate scripts/parity/common.sh" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""

parity_banner

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    activate_base
    echo "### Boltz2 final diffusion-output parity"
    python -m scripts.parity.boltz2.sampler_parity \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/boltz2_conf.ckpt"
    ;;
  *)
    echo "unknown array task ${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

echo "=== done: $(date) ==="
