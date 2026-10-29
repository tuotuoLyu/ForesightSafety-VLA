<p align="center">
  <img src="assets/banner.png" alt="ForesightSafety-VLA" width="100%">
</p>

<p align="center">
  <img alt="IROS 2026" src="https://img.shields.io/badge/IROS-2026-1f6feb">
  <a href="https://arxiv.org/abs/2606.27079"><img alt="arXiv 2606.27079" src="https://img.shields.io/badge/arXiv-2606.27079-b31b1b"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2ea44f">
  <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-3776ab">
</p>

<h1 align="center">ForesightSafety-VLA：面向视觉—语言—动作模型的统一诊断性安全基准</h1>

<p align="center">
Mingyang Lyu<sup>a,b,c,e,*</sup>, Yinqian Sun<sup>a,*</sup>, Yiyang Jia<sup>a</sup>, Sicheng Shen<sup>a,e</sup>, Moquan Sha<sup>a</sup>,
Huangrui Li<sup>e</sup>, Feifei Zhao<sup>a,e,b,c,†</sup>, and Yi Zeng<sup>d,c,b,a,†</sup>
</p>

<p align="center">
<sup>a</sup> Brain-inspired Cognitive AI Lab, Institute of Automation, Chinese Academy of Sciences, Beijing, China<br>
<sup>b</sup> Beijing Key Laboratory of Safe AI and Superalignment, China<br>
<sup>c</sup> Beijing Institute of AI Safety and Governance, China<br>
<sup>d</sup> Gaoling School of AI, Renmin University of China, Beijing, China<br>
<sup>e</sup> University of Chinese Academy of Sciences (UCAS), Beijing, China<br>
<sup>*</sup> 共同第一作者 &nbsp; <sup>†</sup> 共同通讯作者
</p>

<p align="center"><b>已被 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS 2026）接收。</b><br>
<a href="https://arxiv.org/abs/2606.27079">arXiv:2606.27079</a></p>

[English](README.md) | [简体中文](README_zh.md)

**一句话概括**：在 RoboTwin 中把安全作为 VLA 的首要评测目标——66 个安全增强基础场景、13 类安全分类体系、场景结构 / 语言指令 / 视觉观测三个受控变化维度，以及区分安全/危险成功与失败的过程级指标（累积安全成本 CC、风险暴露时间 RET）。

<p align="center">
  <img src="assets/readme/overview.png" alt="安全分类体系与三个诊断维度" width="100%">
</p>
<p align="center"><sub>论文图 1：左为 13 类安全分类体系；右为结构（L0–L2）、语言（W0–W4）、视觉（V0–V4）三个诊断维度。</sub></p>

## 要点

- **安全嵌入场景构造**：基于 RoboTwin 现有资产，通过危险因素注入、约束收紧、时序前提插入构建 66 个基础场景，覆盖 5 类 Safe-Core 物理安全套件。
- **13 类安全分类**：Safe-Core（物理交互）、Safe-Lang（指令侧）、Safe-Vis（感知侧）；语言与视觉类别以评测时叠加扰动的方式作用于同一批场景。
- **过程级指标**：软—硬双阈值监测每个仿真步；每个回合划分为安全成功、危险成功、安全失败、危险失败，并报告 CC、RET 与安全调整成功率 SASR（λ = 1）。
- **按来源诊断**：分别改变结构、语言、视觉，定位安全退化的主要来源。

## 指标（论文第 IV 节）

对每个安全通道 k 计算带符号安全裕度 q<sub>k</sub>(s<sub>t</sub>)：q ≥ ε 为安全，0 ≤ q < ε 为预警，q < 0 为违规（式 1）。预警区内累积成本 c<sub>k</sub> = g<sub>k</sub>(max(0, ε<sub>k</sub> − q<sub>k</sub>))（式 2），CC = Σ<sub>t</sub>Σ<sub>k</sub> w<sub>k</sub>c<sub>k</sub>（式 3），表中报告按回合平均的 CC<sub>i</sub>/T<sub>i</sub>；SASR(λ) = (1/N)Σ 1(S<sub>i</sub>∧¬V<sub>i</sub>) exp(−λ CC<sub>i</sub>/T<sub>i</sub>)（式 4）；RET<sub>k</sub> 与 RET 分别统计单通道和任一通道处于预警或违规区的步数（式 5、6）。实现见 `foresight_safety/metrics.py`，论文未指定的取值（ε、g、w 等）及其默认值见 [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md)。

## 演示

<p align="center"><img src="assets/readme/levels.gif" alt="结构与视觉变化等级" width="80%"></p>
<p align="center"><img src="assets/readme/process_risk.gif" alt="危险成功回合的过程级风险" width="80%"></p>

## 结果（来自论文）

以下数字取自论文（每个模型—任务—设置 3 个随机种子共 50 个回合的平均值，L0/W0/V0）。

| 模型 | SSR ↑ | USR ↓ | SFR | UFR ↓ | CC ↓ | SASR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA-oft | **0.42** | **0.06** | 0.37 | **0.15** | **0.18** | **0.35** |
| RDT | 0.30 | 0.10 | 0.34 | 0.26 | 0.29 | 0.22 |
| DP | 0.24 | 0.10 | 0.34 | 0.32 | 0.34 | 0.16 |
| ACT | 0.20 | 0.12 | 0.31 | 0.37 | 0.39 | 0.12 |

分套件结果（论文表 III）见英文 README。论文结论：所有被评测基线均非完全安全；较弱模型失败时更危险（成功回合中危险成功占比 OpenVLA-oft 为 12.5%，ACT 为 37.5%）；热/能量安全最难、时序安全最易；结构与视觉变化造成的安全退化大于普通语言变化，但 W3–W4 的对抗性语言仍明显有害。

## 安装与快速上手

```bash
git clone https://github.com/tuotuoLyu/ForesightSafety-VLA.git && cd ForesightSafety-VLA
conda create -n foresight python=3.10 -y && conda activate foresight
bash script/_install.sh && bash script/_download_assets.sh      # 与 RoboTwin 相同
bash script/foresight/fetch_policies.sh ACT DP RDT openvla-oft    # 按需获取基线

python3 tools/smoke_cpu.py                       # CPU 冒烟测试（无需仿真器与 GPU）
python3 -m unittest discover -s tests -t .       # CPU 单元测试

bash script/foresight/collect_all.sh 0           # 66 个场景的专家示教
bash script/foresight/train_baselines.sh ACT 0 0 40
bash script/foresight/run_benchmark.sh ACT "{config}" 0 -- \
    --ckpt_dir "policy/ACT/act_ckpt/act-{task}/{config}-40" --temporal_agg true
python3 -m foresight_safety.report eval_result
```

更多说明：场景列表 [docs/SCENARIOS.md](docs/SCENARIOS.md)，基线 [docs/BASELINES.md](docs/BASELINES.md)，与论文的逐项对应 [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md)，相对 RoboTwin 的改动 [docs/UPSTREAM_CHANGES.md](docs/UPSTREAM_CHANGES.md)。

## 引用

```bibtex
@inproceedings{lyu2026foresightsafety,
  title     = {ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for Vision-Language-Action Models},
  author    = {Lyu, Mingyang and Sun, Yinqian and Jia, Yiyang and Shen, Sicheng and Sha, Moquan and Li, Huangrui and Zhao, Feifei and Zeng, Yi},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  eprint        = {2606.27079},
  archivePrefix = {arXiv}
}
```

## 致谢与许可

本项目基于 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 与 SAPIEN，并使用其中的 ACT、DP、DP3、RDT、OpenVLA-OFT、openpi（Pi0、Pi0.5）、DexVLA、TinyVLA、LLaVA-VLA 集成。本研究得到国家自然科学基金（62576341、32441109）、北京市自然科学基金（4252052）和中国科学院自动化研究所（E411230101）资助。代码以 MIT 许可发布（[LICENSE](LICENSE)）；RoboTwin 派生文件保留其 MIT 许可（[LICENSE-RoboTwin](LICENSE-RoboTwin)），各基线保留各自许可（[THIRD_PARTY.md](THIRD_PARTY.md)）。
