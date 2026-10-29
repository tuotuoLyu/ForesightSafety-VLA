#!/bin/bash
# Collect expert demonstrations for the 66 ForesightSafety-VLA scenarios (RoboTwin collect_data.py),
# each with the task config used for the paper's training data (foresight_safety/tasks.py: collect_config).
#   bash script/foresight/collect_all.sh [gpu_id] [task ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU_ID=${1:-0}
shift || true
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ $# -gt 0 ]]; then
  PAIRS=$(python3 -c "import sys; from foresight_safety.tasks import get_task; [print(t, get_task(t).collect_config) for t in sys.argv[1:]]" "$@")
else
  PAIRS=$(python3 -c "from foresight_safety.tasks import TASKS; [print(t, s.collect_config) for t, s in sorted(TASKS.items())]")
fi

while read -r task config; do
  [[ -z "${task}" ]] && continue
  echo "===== ${task} (task_config=${config}) ====="
  PYTHONWARNINGS=ignore::UserWarning python3 script/collect_data.py "${task}" "${config}" < /dev/null
  rm -rf "data/${task}/${config}/.cache"
done <<< "${PAIRS}"
