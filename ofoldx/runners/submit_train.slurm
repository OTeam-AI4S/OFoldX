#!/bin/bash
#SBATCH --job-name=ofoldx-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/users/qinghanw313/AlloDesign/logs/%x-%j.out
#SBATCH --error=/data/users/qinghanw313/AlloDesign/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/users/qinghanw313/AlloDesign}"
PYTHON="${PYTHON:-/data/users/qinghanw313/miniforge3/envs/allodesign-cu132/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

fail() {
    echo "submit_train.slurm: $*" >&2
    exit 2
}

usage() {
    echo "Usage: sbatch [sbatch-options] $0 [--] <ofoldx.runners.train arguments...>" >&2
    echo "All arguments after an optional leading -- are forwarded unchanged." >&2
    echo "Crop example: sbatch $0 data/cifs --max-tokens 384 --cluster-file data/clusters.txt" >&2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "submit this script with sbatch; do not run it on a management node"
[[ -d "${REPO_ROOT}" ]] || fail "REPO_ROOT is not a directory: ${REPO_ROOT}"
[[ -f "${REPO_ROOT}/ofoldx/runners/train.py" ]] || fail "missing Python module ofoldx.runners.train under ${REPO_ROOT}"
[[ -x "${PYTHON}" ]] || fail "PYTHON is not executable: ${PYTHON}"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || fail "NPROC_PER_NODE must be a positive integer: ${NPROC_PER_NODE}"
[[ "${SLURM_NNODES:-}" =~ ^[1-9][0-9]*$ ]] || fail "SLURM_NNODES must be a positive integer: ${SLURM_NNODES:-unset}"
case "${NCCL_SOCKET_FAMILY}" in
    AF_INET|AF_INET6) ;;
    *) fail "NCCL_SOCKET_FAMILY must be AF_INET or AF_INET6: ${NCCL_SOCKET_FAMILY}" ;;
esac

NNODES="${SLURM_NNODES}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi
if (( $# == 0 )); then
    usage
    exit 2
fi

mkdir -p "${REPO_ROOT}/tmp/mplconfig"
cd "${REPO_ROOT}"

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || fail "SLURM_CPUS_PER_TASK must be a positive integer: ${CPUS_PER_TASK}"
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS=$(( CPUS_PER_TASK / NPROC_PER_NODE ))
    (( OMP_NUM_THREADS >= 1 )) || OMP_NUM_THREADS=1
fi
[[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]] || fail "OMP_NUM_THREADS must be a positive integer: ${OMP_NUM_THREADS}"
export OMP_NUM_THREADS
export NCCL_SOCKET_FAMILY
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/tmp/mplconfig}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if (( NNODES == 1 && NPROC_PER_NODE == 1 )); then
    command=("${PYTHON}" -m ofoldx.runners.train "$@")
elif (( NNODES == 1 )); then
    command=(
        "${PYTHON}" -m torch.distributed.run
        --standalone
        --nproc-per-node "${NPROC_PER_NODE}"
        -m ofoldx.runners.train
        "$@"
    )
else
    command -v srun >/dev/null 2>&1 || fail "srun is required for a multi-node launch"
    command -v scontrol >/dev/null 2>&1 || fail "scontrol is required to resolve the first allocated hostname"

    if [[ -z "${MASTER_ADDR:-}" ]]; then
        [[ -n "${SLURM_JOB_NODELIST:-}" ]] || fail "SLURM_JOB_NODELIST is required to derive MASTER_ADDR"
        MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
    fi
    [[ -n "${MASTER_ADDR}" ]] || fail "MASTER_ADDR must be non-empty"

    if [[ -z "${MASTER_PORT:-}" ]]; then
        [[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "numeric SLURM_JOB_ID is required to derive MASTER_PORT: ${SLURM_JOB_ID}"
        MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 50000 ))
    fi
    [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || fail "MASTER_PORT must be an integer: ${MASTER_PORT}"
    (( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || fail "MASTER_PORT must be between 1 and 65535: ${MASTER_PORT}"

    RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID}}"
    [[ -n "${RDZV_ID}" ]] || fail "RDZV_ID must be non-empty"
    export MASTER_ADDR MASTER_PORT RDZV_ID

    command=(
        srun
        --nodes "${NNODES}"
        --ntasks "${NNODES}"
        --ntasks-per-node=1
        --kill-on-bad-exit=1
        "${PYTHON}" -m torch.distributed.run
        --nnodes "${NNODES}"
        --nproc-per-node "${NPROC_PER_NODE}"
        --rdzv-backend c10d
        --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}"
        --rdzv-id "${RDZV_ID}"
        -m ofoldx.runners.train
        "$@"
    )
fi

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

exec "${command[@]}"
