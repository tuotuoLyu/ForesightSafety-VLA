# Third-party components

ForesightSafety-VLA is released under the MIT License (`LICENSE`). It is a fork of RoboTwin and depends on the
projects below. Licence texts of included upstream code are kept unchanged.

## Included in this repository

| Component | Where | Upstream | Licence |
|---|---|---|---|
| RoboTwin 2.0 (simulation tasks, robot/camera stack, data collection and evaluation scripts, task descriptions) | `envs/`, `script/`, `task_config/`, `description/`, `assets/_download.py`, `policy/Your_Policy/`, `collect_data.sh` | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), pinned commit `03a1a5542c1c9264f468a803846afdff42aee3d1` | MIT, `LICENSE-RoboTwin` |

The files changed relative to that commit are listed in `docs/UPSTREAM_CHANGES.md`.

## Fetched on demand, not redistributed

`script/foresight/fetch_policies.sh` copies the baseline policies from the same pinned RoboTwin commit into
`policy/` and applies the small patches in `policy_patches/` (patches are MIT, like this repository). Each
baseline keeps the licence files that ship with it; review them before redistribution.

| Baseline (paper name) | Folder | Original project | Licence of the RoboTwin copy |
|---|---|---|---|
| ACT | `policy/ACT` | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | MIT (`policy/ACT/LICENSE`); DETR part Apache-2.0 (`policy/ACT/detr/LICENSE`) |
| DP (Diffusion Policy) | `policy/DP` | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | MIT (upstream project) |
| DP3 | `policy/DP3` | [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | MIT (upstream project) |
| RDT | `policy/RDT` | [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) | MIT (upstream project) |
| Pi0, Pi0.5 | `policy/pi0` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 (`policy/pi0/LICENSE`) |
| OpenVLA-oft | `policy/openvla-oft` | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | MIT (`policy/openvla-oft/LICENSE`, `rlds_dataset_builder/LICENSE`) |
| DexVLA | `policy/DexVLA` | DexVLA authors | MIT (`policy/DexVLA/LICENSE`); policy heads Apache-2.0 |
| TinyVLA | `policy/TinyVLA` | TinyVLA authors | MIT (`policy/TinyVLA/LICENSE`); policy heads Apache-2.0 |
| LLaVA-VLA | `policy/LLaVA-VLA` | LLaVA-VLA authors, built on [LLaVA](https://github.com/haotian-liu/LLaVA) | no licence file in the RoboTwin copy; LLaVA is Apache-2.0 |

Pretrained weights and datasets of these baselines are distributed by their authors under their own terms.

## Runtime dependencies

SAPIEN, mplib, CuRobo, PyTorch, NumPy and the other packages installed by `script/_install.sh` (RoboTwin's
installer) keep their own licences. RoboTwin's simulation assets (embodiments, objects, textures) are
downloaded by `script/_download_assets.sh` from the RoboTwin release and are subject to RoboTwin's terms.

## Figures

The figures in `assets/readme/` are taken from the ForesightSafety-VLA paper by its authors.
