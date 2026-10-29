"""Per-episode dual-threshold safety monitor (paper Sec. IV-A) and the run-level episode writer.

``Base_Task.take_action`` calls :meth:`SafetyMonitor.observe` after every physics step
(``scene.step()``), so ``t`` in Eq. (2)-(6) indexes simulator steps and ``T_i`` is the number of monitored
steps of episode ``i``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from . import adapters
from .metrics import DEFAULT_LAMBDA, EpisodeAccumulator, aggregate, finalize_episode, group_by
from .probes import EnvView
from .tasks import TaskSpec, get_task


class SafetyMonitor:
    def __init__(self, spec: TaskSpec, lam: float = DEFAULT_LAMBDA, keep_trace: bool = False,
                 channel_weights: Optional[Dict[str, float]] = None, disabled_channels=()):
        self.spec = spec
        self.lam = float(lam)
        self.keep_trace = keep_trace
        weights = dict(channel_weights or {})
        self.channels = [replace(c, weight=float(weights.get(c.name, c.weight)))
                         for c in spec.channels if c.name not in set(disabled_channels)]
        self.meta = {c.name: c.meta() for c in self.channels}
        self.mem: Dict = {}
        self.acc = EpisodeAccumulator(self.meta, keep_trace=keep_trace)
        self.errors: Dict[str, str] = {}
        self.observer = adapters.PHASE_OBSERVERS.get(spec.name)

    @classmethod
    def for_task(cls, task_name: str, **kw) -> "SafetyMonitor":
        return cls(get_task(task_name), **kw)

    def reset(self):
        self.mem = {}
        self.acc = EpisodeAccumulator(self.meta, keep_trace=self.keep_trace)
        self.errors = {}

    def observe(self, env) -> Dict[str, Optional[float]]:
        """Evaluate every channel at the current state and accumulate cost/exposure."""
        if self.observer is not None:
            try:
                self.observer(env, self.mem)
            except Exception as exc:  # keep the rollout alive; report once
                self.errors.setdefault("phase_observer", repr(exc))
        view = EnvView(env)
        scores: Dict[str, Optional[float]] = {}
        for c in self.channels:
            try:
                q = c.fn(view, self.mem)
            except Exception as exc:
                self.errors.setdefault(c.name, repr(exc))
                q = None
            scores[c.name] = q
        self.acc.update(scores)
        return scores

    def finish(self, task_success: bool, **extra) -> Dict:
        rec = finalize_episode(self.acc, task_success, self.lam, task_name=self.spec.name,
                               safe_core_suite=self.spec.suite, **extra)
        rec["channel_category"] = {c.name: c.category for c in self.channels}
        if self.errors:
            rec["monitor_errors"] = dict(self.errors)
        if self.keep_trace:
            rec["trace"] = self.acc.trace
            rec["eps"] = {c.name: c.eps for c in self.channels}
        return rec


CSV_FIELDS = ["episode", "seed", "task_name", "policy_name", "safe_core_suite", "structure_level", "language_level",
              "visual_level", "embodiment", "instruction", "steps", "task_success", "hard_violation", "outcome",
              "SSR", "USR", "SFR", "UFR", "CC", "CC_raw", "RET", "RET_steps", "SASR"]


class EpisodeWriter:
    """Writes ``foresight_episodes.jsonl``/``.csv`` and ``foresight_summary.json`` into a run directory."""

    def __init__(self, save_dir, run_info: Optional[Dict] = None):
        self.dir = Path(save_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "foresight_episodes.jsonl"
        self.csv = self.dir / "foresight_episodes.csv"
        self.summary = self.dir / "foresight_summary.json"
        self.run_info = dict(run_info or {})
        self.records: List[Dict] = []

    def add(self, record: Dict):
        self.records.append(record)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        with self.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in self.records:
                w.writerow(r)
        self.write_summary()

    def write_summary(self) -> Dict:
        summary = {
            "benchmark": "ForesightSafety-VLA",
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.run_info,
            "overall": aggregate(self.records),
            "by_seed": group_by(self.records, "seed"),
        }
        self.summary.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        return summary


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
