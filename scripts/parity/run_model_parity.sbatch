#!/bin/bash
#SBATCH --job-name=ofoldx_gpu_parity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
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
activate_zyc

parity_banner
python -c 'import sys, torch; print("python", sys.version.replace("\n", " ")); print("executable", sys.executable); print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), torch.cuda.device_count())'

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    echo "### Protenix/AF3 full forward parity"
    python -m scripts.parity.ref_forward \
      --reference-root "$OFOLDX_REFERENCE_PROTENIX" \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/protenix_base_default_v0.5.0.pt" \
      --input 2orb \
      --sample-2orb "$OFOLDX_ROOT/data/benchmark/2orb.pkl.gz" \
      --max-tokens 64 \
      --device cuda:0 \
      --n-cycle 4 \
      --n-sample 2 \
      --n-step 1 \
      --atol 2e-2 \
      --json "$OFOLDX_PARITY_LOGDIR/protenix_ref_forward_${SLURM_JOB_ID:-manual}.json"
    ;;
  1)
    echo "### ODesign complete graph parity"
    python -m scripts.parity.odesign.forward_parity \
      --reference-root "$OFOLDX_REFERENCE_ODESIGN" \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/odesign_base_prot_flex.pt" \
      --device cuda:0 \
      --n-cycle 1 \
      --n-token 3 \
      --n-msa 1 \
      --json "$OFOLDX_PARITY_LOGDIR/odesign_trunk_${SLURM_JOB_ID:-manual}.json"
    ;;
  2)
    echo "### DEBUG ONLY: Boltz2 whole-trunk synthetic forward parity"
    python -m scripts.parity.boltz2.forward_parity \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/boltz2_conf.ckpt"
    ;;
  3)
    echo "### DEBUG ONLY: Boltz2 diffusion module parity"
    python -m scripts.parity.boltz2.diffusion_parity \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/boltz2_conf.ckpt"
    ;;
  4)
    echo "### DEBUG ONLY: Boltz2 confidence-head parity"
    python -m scripts.parity.boltz2.confidence_parity \
      --checkpoint "$OFOLDX_WEIGHTS_ROOT/boltz2_conf.ckpt"
    ;;
  *)
    echo "unknown array task ${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

echo "=== done: $(date) ==="
