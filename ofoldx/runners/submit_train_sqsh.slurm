#!/bin/bash
#SBATCH --job-name=ofoldx-train-sqsh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
# Slurm opens these paths before the script runs; create the logs directory before sbatch.
#SBATCH --output=/data/users/qinghanw313/AlloDesign/logs/%x-%j.out
#SBATCH --error=/data/users/qinghanw313/AlloDesign/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/users/qinghanw313/AlloDesign}"
SQSH_IMAGE="${SQSH_IMAGE:-${REPO_ROOT}/tmp/enroot-allodesign-cu132/allodesign-cu132.sqsh}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-/opt/allodesign-cu132/bin/python}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/data:/data}"
PYXIS_EXTRA_ENV_NAMES="${PYXIS_EXTRA_ENV_NAMES:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

fail() {
    echo "submit_train_sqsh.slurm: $*" >&2
    exit 2
}

usage() {
    echo "Usage: sbatch [resources] $0 -- --config YAML_PATH" >&2
    echo "Config smoke: sbatch --cpus-per-task=4 --mem=96G --time=00:10:00 $0 -- \\" >&2
    echo "  --config configs/train.yaml --print-config" >&2
    echo "Before sbatch, create /data/users/qinghanw313/AlloDesign/logs for Slurm output." >&2
    echo "Put GPU/node/job resources before the script; they are not YAML keys." >&2
    echo "All arguments after an optional leading -- are forwarded unchanged." >&2
    echo "Legacy CLI-only mode: sbatch $0 -- data/cifs --epochs 10 --device cuda" >&2
    echo "Default image: ${SQSH_IMAGE}" >&2
    echo "Override image/mounts with SQSH_IMAGE and CONTAINER_MOUNTS in --export." >&2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "submit this script with sbatch; do not run it on a management node"
[[ -d "${REPO_ROOT}" ]] || fail "REPO_ROOT is not a directory: ${REPO_ROOT}"
[[ -f "${REPO_ROOT}/ofoldx/runners/train.py" ]] || fail \
    "missing Python module ofoldx.runners.train under ${REPO_ROOT}"
[[ -r "${SQSH_IMAGE}" && -s "${SQSH_IMAGE}" ]] || fail "SQSH_IMAGE is not a readable, nonempty file: ${SQSH_IMAGE}"
[[ "${CONTAINER_PYTHON}" == /* ]] || fail "CONTAINER_PYTHON must be an absolute path: ${CONTAINER_PYTHON}"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || fail \
    "NPROC_PER_NODE must be a positive integer: ${NPROC_PER_NODE}"
[[ "${SLURM_NNODES:-}" =~ ^[1-9][0-9]*$ ]] || fail \
    "SLURM_NNODES must be a positive integer: ${SLURM_NNODES:-unset}"
case "${NCCL_SOCKET_FAMILY}" in
    AF_INET|AF_INET6) ;;
    *) fail "NCCL_SOCKET_FAMILY must be AF_INET or AF_INET6: ${NCCL_SOCKET_FAMILY}" ;;
esac
command -v srun >/dev/null 2>&1 || fail "srun is required for a Pyxis launch"

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
[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || fail \
    "SLURM_CPUS_PER_TASK must be a positive integer: ${CPUS_PER_TASK}"
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS=$(( CPUS_PER_TASK / NPROC_PER_NODE ))
    (( OMP_NUM_THREADS >= 1 )) || OMP_NUM_THREADS=1
fi
[[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]] || fail \
    "OMP_NUM_THREADS must be a positive integer: ${OMP_NUM_THREADS}"

CONTAINER_CACHE_ROOT="${CONTAINER_CACHE_ROOT:-/tmp/ofoldx-${SLURM_JOB_ID}}"
[[ "${CONTAINER_CACHE_ROOT}" == /* ]] || fail \
    "CONTAINER_CACHE_ROOT must be an absolute path: ${CONTAINER_CACHE_ROOT}"

export OMP_NUM_THREADS
export NCCL_SOCKET_FAMILY
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/tmp/mplconfig}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HOME="${CONTAINER_CACHE_ROOT}/home"
export XDG_CACHE_HOME="${CONTAINER_CACHE_ROOT}/xdg"
export TRITON_CACHE_DIR="${CONTAINER_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CONTAINER_CACHE_ROOT}/cuda"
export TORCHINDUCTOR_CACHE_DIR="${CONTAINER_CACHE_ROOT}/torchinductor"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"

container_env_names=(
    OMP_NUM_THREADS
    NCCL_SOCKET_FAMILY
    PYTHONNOUSERSITE
    PYTHONDONTWRITEBYTECODE
    PYTHONPATH
    MPLCONFIGDIR
    HF_HUB_OFFLINE
    HOME
    XDG_CACHE_HOME
    TRITON_CACHE_DIR
    CUDA_CACHE_PATH
    TORCHINDUCTOR_CACHE_DIR
    NVIDIA_DRIVER_CAPABILITIES
)
for name in NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_NET NCCL_P2P_DISABLE NCCL_IB_DISABLE; do
    if [[ -v "${name}" ]]; then
        container_env_names+=("${name}")
    fi
done
if [[ -n "${PYXIS_EXTRA_ENV_NAMES}" ]]; then
    IFS=',' read -r -a extra_env_names <<<"${PYXIS_EXTRA_ENV_NAMES}"
    for name in "${extra_env_names[@]}"; do
        [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail \
            "invalid name in PYXIS_EXTRA_ENV_NAMES: ${name}"
        [[ -v "${name}" ]] || fail "PYXIS_EXTRA_ENV_NAMES references an unset variable: ${name}"
        container_env_names+=("${name}")
    done
fi
container_env_csv="$(IFS=,; echo "${container_env_names[*]}")"

if (( NNODES > 1 )); then
    command -v scontrol >/dev/null 2>&1 || fail "scontrol is required to resolve the first allocated hostname"

    if [[ -z "${MASTER_ADDR:-}" ]]; then
        [[ -n "${SLURM_JOB_NODELIST:-}" ]] || fail "SLURM_JOB_NODELIST is required to derive MASTER_ADDR"
        MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
    fi
    [[ -n "${MASTER_ADDR}" ]] || fail "MASTER_ADDR must be non-empty"

    if [[ -z "${MASTER_PORT:-}" ]]; then
        [[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail \
            "numeric SLURM_JOB_ID is required to derive MASTER_PORT: ${SLURM_JOB_ID}"
        MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 50000 ))
    fi
    [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || fail "MASTER_PORT must be an integer: ${MASTER_PORT}"
    (( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || fail \
        "MASTER_PORT must be between 1 and 65535: ${MASTER_PORT}"

    RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID}}"
    [[ -n "${RDZV_ID}" ]] || fail "RDZV_ID must be non-empty"
    export MASTER_ADDR MASTER_PORT RDZV_ID
    container_env_csv+=",MASTER_ADDR,MASTER_PORT,RDZV_ID"
fi

pyxis_options=(
    --container-image="${SQSH_IMAGE}"
    --container-workdir="${REPO_ROOT}"
    --container-readonly
    --no-container-mount-home
    --no-container-entrypoint
    --container-env="${container_env_csv}"
)
if [[ -n "${CONTAINER_MOUNTS}" ]]; then
    pyxis_options+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

if (( NNODES == 1 && NPROC_PER_NODE == 1 )); then
    command=(
        srun
        --nodes=1
        --ntasks=1
        --kill-on-bad-exit=1
        "${pyxis_options[@]}"
        "${CONTAINER_PYTHON}" -m ofoldx.runners.train
        "$@"
    )
elif (( NNODES == 1 )); then
    command=(
        srun
        --nodes=1
        --ntasks=1
        --kill-on-bad-exit=1
        "${pyxis_options[@]}"
        "${CONTAINER_PYTHON}" -m torch.distributed.run
        --standalone
        --nproc-per-node "${NPROC_PER_NODE}"
        -m ofoldx.runners.train
        "$@"
    )
else
    command=(
        srun
        --nodes "${NNODES}"
        --ntasks "${NNODES}"
        --ntasks-per-node=1
        --kill-on-bad-exit=1
        "${pyxis_options[@]}"
        "${CONTAINER_PYTHON}" -m torch.distributed.run
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
