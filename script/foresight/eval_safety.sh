#!/bin/bash
# Safety evaluation of one policy on one scenario at one (L, W, V) setting.
#   bash script/foresight/eval_safety.sh <policy_dir> <task> <task_config> <ckpt_setting> [L] [W] [V] [gpu_id] [-- extra overrides]
# <policy_dir> is a folder under policy/ with deploy_policy.yml (e.g. ACT, DP, RDT, openvla-oft, pi0).
# Defaults: L0 W0 V0, 50 episodes over seeds 0,1,2, 2 warm-up episodes, lambda = 1.
set -euo pipefail
cd "$(dirname "$0")/../.."
POLICY=${1:?policy folder under policy/}
TASK=${2:?task name}
TASK_CONFIG=${3:?task config}
CKPT=${4:?ckpt setting}
L=${5:-L0}; W=${6:-W0}; V=${7:-V0}; GPU_ID=${8:-0}
shift $(( $# < 8 ? $# : 8 ))
[[ "${1:-}" == "--" ]] && shift
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PYTHONWARNINGS=ignore::UserWarning \
python3 script/eval_policy.py --config "policy/${POLICY}/deploy_policy.yml" \
  --overrides \
  --task_name "${TASK}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT}" \
  --policy_name "${POLICY}" \
  --seed 0 \
  --safety_eval True \
  --foresight_structure_level "${L}" \
  --foresight_language_level "${W}" \
  --foresight_visual_level "${V}" \
  "$@"
