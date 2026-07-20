#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/parity/common.sh
source "${SCRIPT_DIR}/common.sh"

parity_banner "Submitting OFoldX e2e public-output parity jobs"
sbatch --array=0 "${OFOLDX_ROOT}/scripts/parity/run_model_parity.sbatch"
sbatch --array=0 "${OFOLDX_ROOT}/scripts/parity/run_missing_public_output_parity.sbatch"
