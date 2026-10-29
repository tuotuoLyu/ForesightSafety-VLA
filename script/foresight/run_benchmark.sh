#!/bin/bash
# Full protocol for one policy (paper Sec. V-B):
#   1) aggregate evaluation on all 66 scenarios at L0/W0/V0 (Table II / Table III),
#   2) one-factor diagnostic sweeps L0-L2, W0-W4, V0-V4 on the 10-scenario subset (Fig. 4),
#   3) report.
#   bash script/foresight/run_benchmark.sh <policy_dir> <ckpt_setting> [gpu_id] [-- policy-specific overrides]
# "{task}" and "{config}" in <ckpt_setting> and in the overrides are replaced per scenario, e.g. for ACT
#   bash script/foresight/run_benchmark.sh ACT "{config}" 0 -- \
#       --ckpt_dir "policy/ACT/act_ckpt/act-{task}/{config}-40" --temporal_agg true
# The task config of each scenario is its collect_config (the config its demonstrations were collected in).
set -euo pipefail
cd "$(dirname "$0")/../.."
POLICY=${1:?policy folder under policy/}
CKPT=${2:?ckpt setting}
GPU_ID=${3:-0}
shift $(( $# < 3 ? $# : 3 ))
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

ALL=$(python3 -c "from foresight_safety.tasks import TASKS; [print(t, s.collect_config) for t, s in sorted(TASKS.items())]")
SUBSET=$(python3 -c "from foresight_safety.tasks import DIAGNOSTIC_SUBSET, get_task; [print(t, get_task(t).collect_config) for t in DIAGNOSTIC_SUBSET]")

cell() {  # task config L W V
  local task=$1 config=$2 ckpt=${CKPT//\{task\}/$1}
  ckpt=${ckpt//\{config\}/$2}
  local args=()
  for a in "${EXTRA[@]}"; do a=${a//\{task\}/$1}; args+=("${a//\{config\}/$2}"); done
  bash script/foresight/eval_safety.sh "${POLICY}" "${task}" "${config}" "${ckpt}" "$3" "$4" "$5" "${GPU_ID}" -- "${args[@]}" < /dev/null
}

while read -r task config; do
  cell "${task}" "${config}" L0 W0 V0
done <<< "${ALL}"

while read -r task config; do
  for L in L1 L2; do cell "${task}" "${config}" "${L}" W0 V0; done
  for W in W1 W2 W3 W4; do cell "${task}" "${config}" L0 "${W}" V0; done
  for V in V1 V2 V3 V4; do cell "${task}" "${config}" L0 W0 "${V}"; done
done <<< "${SUBSET}"

python3 -m foresight_safety.report "eval_result" --out "eval_result/foresight_report_${POLICY}"
