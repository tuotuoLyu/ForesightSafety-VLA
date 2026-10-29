"""ForesightSafety-VLA: process-level safety evaluation for VLA policies on RoboTwin.

Modules
-------
metrics        Eq. (1)-(6): dual-threshold states, step cost, CC, RET, four-quadrant outcomes, SASR.
taxonomy       Table I (13 categories, 3 families) and the L/W/V diagnostic levels.
probes         signed safety margins q_k(s_t) computed from the simulator state.
tasks          the 66 base scenarios, their Safe-Core suite and monitored channels.
adapters       evaluation-time adapters for the scenario classes.
monitor        per-episode monitor and run-level episode writer.
perturbations  structure (L), language (W) and visual (V) variation.
protocol       episodes, seeds, warm-up, sweeps and baselines of Sec. V-A.
report         aggregation into the paper's tables.
paper_results  the paper's reported numbers (reference only).

``metrics``, ``taxonomy``, ``protocol``, ``paper_results`` and ``report`` are pure Python; the rest need
numpy, and only the simulator hooks need SAPIEN.
"""

from .metrics import aggregate, classify, finalize_episode, quadrant, soft_cost  # noqa: F401
from .taxonomy import SAFE_CORE_SUITES, TAXONOMY  # noqa: F401

__version__ = "1.0.0"
