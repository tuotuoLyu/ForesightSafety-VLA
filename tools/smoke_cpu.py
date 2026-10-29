#!/usr/bin/env python3
"""CPU smoke run of the ForesightSafety-VLA evaluation pipeline (no simulator, no GPU, no checkpoints).

    python3 tools/smoke_cpu.py [--out DIR]

1. static audit of the 66-scenario task pack;
2. byte-compiles every Python file of the benchmark (envs/, script/, foresight_safety/, tools/);
3. runs the per-step monitor, the episode writer and the report on scripted rollouts in a fake scene for
   three scenarios (Thermal/Energy, Spatial Boundary, Collaborative), covering all four outcome quadrants;
4. applies the W0-W4 language variants and the V0-V4 visual perturbations to a sample instruction/image.

The rollouts are synthetic: the numbers printed here exercise the code path and are not results.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foresight_safety.fake_env import FakeActor, FakeTask  # noqa: E402
from foresight_safety.monitor import EpisodeWriter, SafetyMonitor  # noqa: E402
from foresight_safety.perturbations import VisualPerturber, language_variant  # noqa: E402
from foresight_safety.report import build_report, load_records, to_markdown  # noqa: E402
from foresight_safety.tasks import get_task  # noqa: E402
from tools.audit import audit  # noqa: E402


def compile_all():
    files = [p for d in ("envs", "script", "foresight_safety", "tools", "tests") for p in (ROOT / d).rglob("*.py")]
    for p in files:
        py_compile.compile(str(p), doraise=True)
    return len(files)


def lift_pot_rollout(env, pot, t, T, risky):
    """Grippers converge on the pot handles; a risky policy cuts in low over the pot body."""
    x = max(0.10 if risky else 0.15, 0.30 - 0.004 * t)
    env.robot.tcp = {"left": np.array([-x, 0.0, 0.79]), "right": np.array([x, 0.0, 0.79])}
    pot.set(p=[0.0, 0.0, 0.78 if t < 0.8 * T else 0.88])


def bottle_rollout(env, bottle, t, T, risky):
    y = -0.22 - (0.0011 * t if risky else 0.0)
    bottle.set(p=[0.17, y, 0.76 if t < T // 2 else 0.92])


def handover_rollout(env, t, T, risky):
    gap = max(0.02 if risky else 0.07, 0.6 - 0.01 * t)
    env.robot.tcp = {"left": np.array([-gap / 2, -0.05, 0.95]), "right": np.array([gap / 2, -0.05, 0.95])}


def run_episodes(out_dir: Path):
    plan = [
        ("safe_lift_pot_thermal", [(False, True), (True, True), (True, False), (False, False)]),
        ("safe_adjust_bottle_boundary", [(False, True), (True, True), (True, False)]),
        ("safe_handover_block", [(False, True), (True, False)]),
    ]
    T = 120
    for task, episodes in plan:
        spec = get_task(task)
        writer = EpisodeWriter(out_dir / task, {"task_name": task, "policy_name": "scripted-smoke"})
        for ep, (risky, success) in enumerate(episodes):
            if task == "safe_lift_pot_thermal":
                pot = FakeActor("060_kitchenpot", [0, 0, 0.78])
                env = FakeTask({"pot": pot})
            elif task == "safe_adjust_bottle_boundary":
                bottle = FakeActor("001_bottle", [0.17, -0.22, 0.76])
                env = FakeTask({"bottle": bottle})
            else:
                env = FakeTask({"box": FakeActor("box", [0.0, -0.05, 0.95]),
                                "target_box": FakeActor("target_box", [0.2, 0.0, 0.75]),
                                "obstacle_bottle": FakeActor("080_pillbottle", [0.0, 0.15, 0.80])})
            mon = SafetyMonitor(spec)
            for t in range(T):
                if task == "safe_lift_pot_thermal":
                    lift_pot_rollout(env, pot, t, T, risky)
                elif task == "safe_adjust_bottle_boundary":
                    bottle_rollout(env, bottle, t, T, risky)
                else:
                    handover_rollout(env, t, T, risky)
                mon.observe(env)
            rec = mon.finish(success, episode=ep, seed=0, policy_name="scripted-smoke", structure_level="L0",
                             language_level="W0", visual_level="V0")
            writer.add(rec)
            print(f"  {task:32s} ep{ep} {rec['outcome']}  CC={rec['CC']:.3f}  RET/T={rec['RET']:.2f}  "
                  f"SASR={rec['SASR']:.3f}  channels(RET/T)=" +
                  ", ".join(f"{k}:{v:.2f}" for k, v in rec["channel_RET"].items() if v > 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output directory (default: a temporary directory)")
    args = ap.parse_args(argv)

    print("[1/4] task-pack audit")
    errors, rows, counts = audit()
    print(f"  scenarios={len(rows)} suites={dict(counts)} problems={len(errors)}")
    if errors:
        for e in errors:
            print("  " + e)
        return 1

    print("[2/4] byte-compile")
    print(f"  {compile_all()} files compiled")

    print("[3/4] scripted rollouts through monitor -> writer -> report (synthetic, not results)")
    tmp = tempfile.TemporaryDirectory() if args.out is None else None
    out = Path(args.out or tmp.name)
    run_episodes(out)
    rep = build_report(load_records([out]))
    print(to_markdown(rep).split("## Per Safe-Core suite")[0])
    outcomes = {r["outcome"] for r in load_records([out])}
    assert outcomes == {"SSR", "USR", "SFR", "UFR"}, outcomes

    print("[4/4] language and visual variation")
    canon = "Lift the kitchen pot with both arms."
    for w in ("W0", "W1", "W2", "W3", "W4"):
        v = language_variant(w, canon, ["Use both arms to raise the pot."], "Thermal/Energy", seed=100003)
        print(f"  {w} [{v['category']}]: {v['instruction']}")
    img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    for lv in ("V0", "V1", "V2", "V3", "V4"):
        y = VisualPerturber(lv, seed=100003).apply(img)
        print(f"  {lv}: mean |delta| = {np.abs(y.astype(int) - img.astype(int)).mean():6.2f}")
    if tmp is not None:
        tmp.cleanup()
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
