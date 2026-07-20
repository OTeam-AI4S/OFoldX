#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/parity/common.sh
source "${SCRIPT_DIR}/common.sh"

parity_banner "Submitting OFoldX parity jobs"
sbatch "${OFOLDX_ROOT}/scripts/parity/run_model_parity.sbatch"
sbatch "${OFOLDX_ROOT}/scripts/parity/run_model_parity_cpu.sbatch"
sbatch "${OFOLDX_ROOT}/scripts/parity/run_missing_public_output_parity.sbatch"
