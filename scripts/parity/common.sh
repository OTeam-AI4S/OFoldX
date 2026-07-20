#!/bin/bash

set -euo pipefail

PARITY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OFOLDX_ROOT="${OFOLDX_ROOT:-$(cd "$PARITY_SCRIPT_DIR/../.." && pwd)}"
export OFOLDX_WEIGHTS_ROOT="${OFOLDX_WEIGHTS_ROOT:-/data/users/z/weights}"
export OFOLDX_PARITY_LOGDIR="${OFOLDX_PARITY_LOGDIR:-$OFOLDX_ROOT/artifacts/parity/slurm}"

export OFOLDX_REFERENCE_PROTENIX="${OFOLDX_REFERENCE_PROTENIX:-/data/users/z/reference/Protenix}"
export OFOLDX_REFERENCE_ODESIGN="${OFOLDX_REFERENCE_ODESIGN:-/data/users/z/reference/ODesign}"
export OFOLDX_REFERENCE_RFDIFFUSION="${OFOLDX_REFERENCE_RFDIFFUSION:-/data/users/z/reference/RFdiffusion}"
export OFOLDX_REFERENCE_LIGANDMPNN="${OFOLDX_REFERENCE_LIGANDMPNN:-/data/users/z/reference/LigandMPNN}"
export OFOLDX_REFERENCE_PROTEINMPNN="${OFOLDX_REFERENCE_PROTEINMPNN:-/data/users/z/reference/ProteinMPNN}"
export OFOLDX_REFERENCE_BOLTZ="${OFOLDX_REFERENCE_BOLTZ:-/data/users/z/reference/boltz}"
export OFOLDX_OOPS_ROOT="${OFOLDX_OOPS_ROOT:-/data/users/z/OOps}"

mkdir -p "$OFOLDX_PARITY_LOGDIR"
mkdir -p "${MPLCONFIGDIR:-/tmp/ofoldx-mpl-${USER:-user}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ofoldx-mpl-${USER:-user}}"

cd "$OFOLDX_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export LAYERNORM_TYPE=torch
export TRIANGLE_ATTENTION=torch
export TRIANGLE_MULTIPLICATIVE=torch
export USE_DEEPSPEED_EVO_ATTENTION=false

source /data/miniforge3/etc/profile.d/conda.sh

activate_zyc() {
  conda activate zyc
  export PYTHONPATH="$OFOLDX_ROOT:$OFOLDX_OOPS_ROOT"
}

activate_base() {
  conda activate base
  export PYTHONPATH="$OFOLDX_ROOT"
}

parity_banner() {
  echo "=== node: $(hostname) job: ${SLURM_JOB_ID:-none} task: ${SLURM_ARRAY_TASK_ID:-0} start: $(date) ==="
  echo "OFOLDX_ROOT=$OFOLDX_ROOT"
  echo "OFOLDX_WEIGHTS_ROOT=$OFOLDX_WEIGHTS_ROOT"
  echo "OFOLDX_PARITY_LOGDIR=$OFOLDX_PARITY_LOGDIR"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}
