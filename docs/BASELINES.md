# Baselines: fetching, training, evaluating

The paper evaluates OpenVLA-oft, RDT, DP and ACT on all 66 scenarios (Table II, Table III) and adds Pi0.5, Pi0,
DexVLA, LLaVA-VLA, TinyVLA and DP3 in the landscape (Fig. 3) and diagnostics (Fig. 4). All of them are the
RoboTwin integrations at the pinned RoboTwin commit.

## 1. Fetch

```bash
bash script/foresight/fetch_policies.sh                    # all ten (nine folders; Pi0 and Pi0.5 share policy/pi0)
bash script/foresight/fetch_policies.sh ACT DP openvla-oft # or a subset
```

Then follow each policy's own installation notes (RoboTwin documentation, "Policy Baselines"). The script
applies `policy_patches/<name>.patch` where one exists:

| Patch | Change | Why |
|---|---|---|
| `openvla-oft.patch` | registers every scenario as a bimanual RLDS dataset (`foresight_tasks.py`), a mixture per scenario, `foresight_safety_all`, and the 22-scenario `robotwin_all_22` mixture used for the OpenVLA-oft runs; selects the ALOHA action constants for these datasets; enables gradient checkpointing | needed to fine-tune on the scenario demonstrations |
| `TinyVLA.patch` | falls back to "Perform the task: <name>." when a task has no hand-written prompt | the upstream prompt table does not list the scenarios |
| `RDT.patch` | loads `.safetensors` and plain checkpoints; drops the `repo_url` class argument rejected by recent `huggingface_hub` | checkpoint loading with current `huggingface_hub` and safetensors |

## 2. Demonstrations

```bash
bash script/foresight/collect_all.sh 0            # all 66 scenarios on GPU 0
bash script/foresight/collect_all.sh 0 safe_lift_pot_thermal safe_handover_block
```

Each scenario is collected with its `collect_config` (`foresight_safety/tasks.py`), the config used for the
paper's runs: template and `_c` scenarios use their own `task_config/<name>.yml`, composed scenarios use
`demo_randomized`. RoboTwin's default of 50 episodes per config applies unless the config says otherwise.

## 3. Training settings used for the paper's runs

Baseline training is not specified in the paper; the settings used for the paper's runs are:

| Baseline | Data | Settings |
|---|---|---|
| ACT | per scenario, 40 demonstrations | RoboTwin `policy/ACT/train.sh`, seed 0 (`script/foresight/train_baselines.sh ACT`) |
| DP | per scenario, 40 demonstrations | RoboTwin `policy/DP/train.sh`, action dim 14, seed 0 (`train_baselines.sh DP`) |
| DP3 | per scenario | RoboTwin `policy/DP3/train.sh` (`train_baselines.sh DP3`) |
| OpenVLA-oft | multi-task mixture `robotwin_all_22` (22 composed scenarios with RLDS builds + `beat_block_hammer`); `foresight_safety_all` covers all 66 | LoRA rank 32, L1 regression, FiLM, 2 input images, proprio, per-GPU batch 1 x grad. accumulation 4 x 4 GPUs (global 16), lr 5e-4, decay after 50k steps, 50k steps (resumed to 100k), image augmentation, `policy/openvla-oft/finetune_aloha.sh` |
| Pi0 / Pi0.5 | 50 demonstrations per scenario, multi-task LeRobot repo | `pi0_base_aloha_robotwin_lora` (openpi), 30k steps, batch 32, 4 FSDP devices |
| RDT, DexVLA, TinyVLA, LLaVA-VLA | RoboTwin defaults | see each policy's RoboTwin page |

Not applied by the patches (machine-specific): GPU ids, FSDP device count, offline W&B logging, dataset repo
ids. Set them in the policy's own scripts.

## 4. Evaluate

```bash
# one cell: policy folder, scenario, task config, checkpoint setting, L, W, V, GPU, then policy-specific overrides
bash script/foresight/eval_safety.sh ACT safe_lift_pot_thermal safe_lift_pot_thermal safe_lift_pot_thermal L0 W0 V0 0 \
    -- --ckpt_dir policy/ACT/act_ckpt/act-safe_lift_pot_thermal/safe_lift_pot_thermal-40 --temporal_agg true

# the full protocol for one policy (Table II/III cells + Fig. 4 sweeps) and the report;
# {task} and {config} are filled in per scenario
bash script/foresight/run_benchmark.sh ACT "{config}" 0 -- \
    --ckpt_dir "policy/ACT/act_ckpt/act-{task}/{config}-40" --temporal_agg true
```

Policy-specific overrides are the arguments each policy's `eval.sh` passes to `script/eval_policy.py`.
`python -m foresight_safety.report eval_result` aggregates any set of runs.
