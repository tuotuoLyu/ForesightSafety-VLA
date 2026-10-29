# Changes relative to RoboTwin

Base: [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) commit
`03a1a5542c1c9264f468a803846afdff42aee3d1` (main, 2025-10-22). To see every change:

```bash
git remote add robotwin https://github.com/RoboTwin-Platform/RoboTwin.git
git fetch robotwin 03a1a5542c1c9264f468a803846afdff42aee3d1
git diff --stat 03a1a5542c1c9264f468a803846afdff42aee3d1 HEAD
```

## Modified upstream files

| File | Change |
|---|---|
| `envs/_base_task.py` | (1) structure level L1/L2 applied after `load_actors` (clutter count, keep-free boxes); (2) `get_obs` passes observations through the visual perturber when one is attached; (3) `take_action` calls the safety monitor after every `scene.step()` and reads task success through `_foresight_task_success` when a monitor is attached; (4) `attach_foresight`, `_foresight_observe`, `_foresight_task_success`; (5) guards in `merge_pkl_to_hdf5_video` / `remove_data_cache` for runs without a data cache; paths built with `os.path.join`. Without a monitor attached, behaviour is identical to upstream. |
| `envs/robot/robot.py` | `_trans_from_gripper_to_endlink` casts the target pose to float (accepts integer poses). |
| `script/eval_policy.py` | `safety_eval` dispatches to `script/foresight_eval.py`; results go to `eval_result/<task>/<policy>/<task_config>/<ckpt>/<L-W-V>/<time>/`; optional `embodiment` override. With `safety_eval` false the upstream loop runs unchanged. |
| `.gitignore` | keeps task configs, README assets and the benchmark files tracked; ignores fetched baselines. |
| `README.md`, `LICENSE` | replaced; the RoboTwin licence is kept as `LICENSE-RoboTwin`. |

## Removed upstream content

| Path | Reason |
|---|---|
| `policy/ACT`, `DP`, `DP3`, `RDT`, `pi0`, `openvla-oft`, `DexVLA`, `TinyVLA`, `LLaVA-VLA` | third-party baselines with their own licences; fetched unchanged from the pinned commit by `script/foresight/fetch_policies.sh`, then patched (`policy_patches/`) |
| `policy/GO1` | not evaluated in the paper |
| `code_gen/` | LLM task generation; the benchmark's scenarios are composed by hand |
| `assets/files/` | media of the upstream README |

`policy/Your_Policy/` (template for evaluating a new policy) is kept.

## Added

| Path | Content |
|---|---|
| `envs/safe_*.py` (66) | the safety-augmented base scenarios |
| `task_config/safe_*.yml` (66) | per-scenario task configs |
| `description/task_instruction/safe_*.json` (66) | instruction templates (seen / unseen) |
| `description/foresight_language_levels.json` | W2-W4 language-variation templates |
| `foresight_safety/` | metrics, taxonomy, probes, scenario registry, monitor, perturbations, protocol, report |
| `script/foresight_eval.py`, `script/foresight/` | evaluation loop and shell entry points |
| `policy_patches/` | patches for TinyVLA, RDT and OpenVLA-OFT |
| `tools/`, `tests/`, `docs/`, `assets/banner.png`, `assets/readme/` | audit, smoke run, tests, documentation, figures |
