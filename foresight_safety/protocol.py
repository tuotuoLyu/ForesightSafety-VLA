"""Evaluation protocol (paper Sec. V-A, V-B)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .taxonomy import LANGUAGE_LEVELS, STRUCTURE_LEVELS, VISUAL_LEVELS

EPISODES_PER_SETTING = 50  # "50 evaluation episodes distributed across 3 random seeds"
SEEDS = (0, 1, 2)
WARMUP_EPISODES = 2  # "a small number of unscored canonical-scene episodes (L0/W0/V0)"
WARMUP_SEED = 99  # RoboTwin seed index of the warm-up block (disjoint from the scored seeds)
CANONICAL = ("L0", "W0", "V0")
EMBODIMENTS = ("aloha-agilex", "franka-panda", "ur5-wsg", "piper", "ARX-X5")  # task_config/_embodiment_config.yml

# Evaluated baselines (Sec. V-A(a)).
AGGREGATE_BASELINES = ("OpenVLA-oft", "RDT", "DP", "ACT")  # Table II
LANDSCAPE_BASELINES = ("Pi0.5", "Pi0", "DexVLA", "LLaVA-VLA", "TinyVLA", "DP3")  # Fig. 3 / Fig. 4 only
DIAGNOSTIC_MODELS = ("Pi0.5", "Pi0", "OpenVLA-oft", "DP", "ACT")  # Fig. 4
# RoboTwin policy folder for each baseline (Pi0 and Pi0.5 share the openpi integration in policy/pi0).
POLICY_DIRS = {
    "OpenVLA-oft": "openvla-oft", "RDT": "RDT", "DP": "DP", "ACT": "ACT", "Pi0.5": "pi0", "Pi0": "pi0",
    "DexVLA": "DexVLA", "LLaVA-VLA": "LLaVA-VLA", "TinyVLA": "TinyVLA", "DP3": "DP3",
}


def episodes_per_seed(total: int = EPISODES_PER_SETTING, seeds: Sequence[int] = SEEDS) -> Dict[int, int]:
    """Split ``total`` episodes over the seeds as evenly as possible (50 over 3 -> 17, 17, 16)."""
    base, extra = divmod(int(total), len(seeds))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(seeds)}


def robotwin_start_seed(seed: int) -> int:
    """RoboTwin's first scene seed for a seed index (script/eval_policy.py: 100000 * (1 + seed))."""
    return 100000 * (1 + int(seed))


def sweeps() -> Dict[str, List[Tuple[str, str, str]]]:
    """One-factor-at-a-time diagnostic sweeps of Fig. 4 (other axes held canonical)."""
    return {
        "structure": [(l, "W0", "V0") for l in STRUCTURE_LEVELS],
        "language": [("L0", w, "V0") for w in LANGUAGE_LEVELS],
        "vision": [("L0", "W0", v) for v in VISUAL_LEVELS],
    }
