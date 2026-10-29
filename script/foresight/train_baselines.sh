#!/bin/bash
# Per-scenario training of the RoboTwin baselines on the collected safety demonstrations, with the settings
# used for the paper's runs (ACT and DP: 40 demonstrations per scenario, seed 0). Wraps each policy's own
# RoboTwin scripts; fetch the policies first (script/foresight/fetch_policies.sh) and follow their install
# notes. OpenVLA-OFT and Pi0/Pi0.5 are trained on multi-task data, see docs/BASELINES.md.
#   bash script/foresight/train_baselines.sh <ACT|DP|DP3> [gpu_id] [seed] [num_demos] [task ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
POLICY=${1:?ACT, DP or DP3}
GPU_ID=${2:-0}; SEED=${3:-0}; NUM=${4:-40}
shift $(( $# < 4 ? $# : 4 ))

if [[ $# -gt 0 ]]; then
  PAIRS=$(python3 -c "import sys; from foresight_safety.tasks import get_task; [print(t, get_task(t).collect_config) for t in sys.argv[1:]]" "$@")
else
  PAIRS=$(python3 -c "from foresight_safety.tasks import TASKS; [print(t, s.collect_config) for t, s in sorted(TASKS.items())]")
fi

while read -r task config; do
  [[ -z "${task}" ]] && continue
  if [[ ! -d "data/${task}/${config}/data" ]]; then
    echo "skip ${task}: no demonstrations in data/${task}/${config}/data"; continue
  fi
  echo "===== ${POLICY} | ${task} | ${config} | ${NUM} demos ====="
  case "${POLICY}" in
    ACT)
      (cd policy/ACT && bash process_data.sh "${task}" "${config}" "${NUM}" \
        && bash train.sh "${task}" "${config}" "${NUM}" "${SEED}" "${GPU_ID}") < /dev/null ;;
    DP)
      (cd policy/DP && bash process_data.sh "${task}" "${config}" "${NUM}" \
        && bash train.sh "${task}" "${config}" "${NUM}" "${SEED}" 14 "${GPU_ID}") < /dev/null ;;
    DP3)
      (cd policy/DP3 && bash process_data.sh "${task}" "${config}" "${NUM}" \
        && bash train.sh "${task}" "${config}" "${NUM}" "${SEED}" "${GPU_ID}") < /dev/null ;;
    *) echo "unsupported policy ${POLICY}"; exit 1 ;;
  esac
done <<< "${PAIRS}"
