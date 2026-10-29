#!/bin/bash
# Fetch RoboTwin policy baselines from the pinned upstream commit and apply the ForesightSafety-VLA patches.
#   bash script/foresight/fetch_policies.sh                 # all evaluated baselines
#   bash script/foresight/fetch_policies.sh ACT DP openvla-oft
# Baselines keep their own licences (see THIRD_PARTY.md). Patches live in policy_patches/.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROBOTWIN_COMMIT=03a1a5542c1c9264f468a803846afdff42aee3d1
ROBOTWIN_REPO=${ROBOTWIN_REPO:-https://github.com/RoboTwin-Platform/RoboTwin.git}
POLICIES=("$@")
if [[ ${#POLICIES[@]} -eq 0 ]]; then
  POLICIES=(ACT DP DP3 RDT pi0 openvla-oft DexVLA TinyVLA LLaVA-VLA)
fi

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT
git -C "${TMP}" init -q
git -C "${TMP}" remote add origin "${ROBOTWIN_REPO}"
git -C "${TMP}" sparse-checkout set --no-cone $(printf '/policy/%s/ ' "${POLICIES[@]}")
git -C "${TMP}" fetch -q --depth 1 --filter=blob:none origin "${ROBOTWIN_COMMIT}"
git -C "${TMP}" checkout -q FETCH_HEAD

for p in "${POLICIES[@]}"; do
  if [[ -e "policy/${p}" ]]; then
    echo "policy/${p} already exists, skipped"
    continue
  fi
  cp -R "${TMP}/policy/${p}" "policy/${p}"
  if [[ -f "policy_patches/${p}.patch" ]]; then
    patch -p1 < "policy_patches/${p}.patch"
  fi
  echo "policy/${p} ready"
done
