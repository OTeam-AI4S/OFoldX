#!/bin/bash
#SBATCH --job-name=ofoldx-protenix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/users/qinghanw313/AlloDesign/tmp/slurm-ofoldx-%x-%j.out
#SBATCH --error=/data/users/qinghanw313/AlloDesign/tmp/slurm-ofoldx-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/users/qinghanw313/AlloDesign}"
PYTHON="${PYTHON:-/data/users/qinghanw313/miniforge3/envs/allodesign-cu132/bin/python}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${REPO_ROOT}/tmp/mini_dataset}"
CLUSTER_FILE="${CLUSTER_FILE:-${REPO_ROOT}/tmp/clusters-by-entity-40.txt}"
CLUSTER_INDICES_FILE="${CLUSTER_INDICES_FILE:-${REPO_ROOT}/tmp/ofoldx-train/cache/mini-cluster-indices.jsonl}"
CLUSTER_EPOCH_SIZE="${CLUSTER_EPOCH_SIZE:-100}"
SEQ_TO_PDB_INDEX="${SEQ_TO_PDB_INDEX:-/data/shared/datasets/protenix/seq_to_pdb_index.json}"
MSA_PATH="${MSA_PATH:-/data/shared/datasets/protenix/mmcif_msa}"
CCD_SNAPSHOT="${CCD_SNAPSHOT:-${REPO_ROOT}/tmp/ccd-v20240608}"
WORKSPACE="${WORKSPACE:-${REPO_ROOT}/tmp/ofoldx-train/workspace}"
RUN_NAME="${RUN_NAME:-protenix-train}"
EPOCHS="${EPOCHS:-}"
STEPS="${STEPS:-}"
WORKERS="${WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
CHECKPOINT="${CHECKPOINT:-1}"

fail() {
    echo "submit_train.slurm: $*" >&2
    exit 2
}

[[ -d "${REPO_ROOT}" ]] || fail "REPO_ROOT is not a directory: ${REPO_ROOT}"
[[ -f "${REPO_ROOT}/ofoldx/runners/train.py" ]] || \
    fail "ofoldx/runners/train.py is missing under REPO_ROOT: ${REPO_ROOT}"
[[ -x "${PYTHON}" ]] || fail "PYTHON is not an executable file: ${PYTHON}"
[[ -e "${TRAIN_SOURCE}" ]] || fail "TRAIN_SOURCE does not exist: ${TRAIN_SOURCE}"
[[ -f "${CLUSTER_FILE}" ]] || fail "CLUSTER_FILE is not a file: ${CLUSTER_FILE}"
[[ -n "${CLUSTER_INDICES_FILE}" ]] || fail "CLUSTER_INDICES_FILE cannot be empty"
[[ ! -e "${CLUSTER_INDICES_FILE}" || -f "${CLUSTER_INDICES_FILE}" ]] || \
    fail "CLUSTER_INDICES_FILE exists but is not a file: ${CLUSTER_INDICES_FILE}"
[[ -f "${SEQ_TO_PDB_INDEX}" ]] || fail "SEQ_TO_PDB_INDEX is not a file: ${SEQ_TO_PDB_INDEX}"
[[ -d "${MSA_PATH}" ]] || fail "MSA_PATH is not a directory: ${MSA_PATH}"
[[ -d "${CCD_SNAPSHOT}" ]] || fail "CCD_SNAPSHOT is not a directory: ${CCD_SNAPSHOT}"
[[ "${CLUSTER_EPOCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
    fail "CLUSTER_EPOCH_SIZE must be a positive integer: ${CLUSTER_EPOCH_SIZE}"
[[ -z "${EPOCHS}" || -z "${STEPS}" ]] || fail "set only one of EPOCHS or STEPS"
if [[ -n "${STEPS}" ]]; then
    [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer: ${STEPS}"
else
    EPOCHS="${EPOCHS:-1}"
    [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || fail "EPOCHS must be a positive integer: ${EPOCHS}"
fi
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be a non-negative integer: ${WORKERS}"
[[ "${DEVICE}" == "cuda" || "${DEVICE}" == "auto" || "${DEVICE}" == "cpu" ]] || \
    fail "DEVICE must be cuda, auto, or cpu: ${DEVICE}"
[[ "${PRECISION}" == "bf16" || "${PRECISION}" == "fp16" ]] || fail "PRECISION must be bf16 or fp16: ${PRECISION}"
[[ "${CHECKPOINT}" == "0" || "${CHECKPOINT}" == "1" ]] || fail "CHECKPOINT must be 0 or 1: ${CHECKPOINT}"
[[ -n "${RUN_NAME}" ]] || fail "RUN_NAME cannot be empty"

mkdir -p "$(dirname "${CLUSTER_INDICES_FILE}")" "${WORKSPACE}" "${REPO_ROOT}/tmp/ofoldx-train/mplconfig"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${REPO_ROOT}/tmp/ofoldx-train/mplconfig"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

checkpoint_flag="--checkpoint"
if [[ "${CHECKPOINT}" == "0" ]]; then
    checkpoint_flag="--no-checkpoint"
fi

boundary_args=(--epochs "${EPOCHS}")
if [[ -n "${STEPS}" ]]; then
    boundary_args=(--steps "${STEPS}")
fi

train_args=(
    "${TRAIN_SOURCE}"
    --cluster-file "${CLUSTER_FILE}"
    --cluster-indices-file "${CLUSTER_INDICES_FILE}"
    --cluster-epoch-size "${CLUSTER_EPOCH_SIZE}"
    --seq-to-pdb-index "${SEQ_TO_PDB_INDEX}"
    --msa-path "${MSA_PATH}"
    "${boundary_args[@]}"
    --device "${DEVICE}"
    --precision "${PRECISION}"
    --workers "${WORKERS}"
    --ccd-snapshot-path "${CCD_SNAPSHOT}"
    "${checkpoint_flag}"
    --workspace "${WORKSPACE}"
    --run-name "${RUN_NAME}"
)

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname) repo=${REPO_ROOT} workspace=${WORKSPACE}"
if [[ "${DEVICE}" != "cpu" ]]; then
    nvidia-smi
fi
"${PYTHON}" -c \
    'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'

exec "${PYTHON}" -m ofoldx.runners.train "${train_args[@]}"
