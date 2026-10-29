#!/usr/bin/env python3
"""Static audit of the ForesightSafety-VLA task pack (no simulator needed).

    python3 tools/audit.py

Checks, for each of the 66 scenarios: the environment class exists and parses, the task config and the
instruction templates exist and are consistent, every environment attribute read by a monitor channel is
assigned by the scenario (or by the upstream task it extends), and the suite / subset bookkeeping.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foresight_safety.taxonomy import CHANNEL_CATEGORIES, SAFE_CORE_SUITES  # noqa: E402
from foresight_safety.tasks import DIAGNOSTIC_SUBSET, TASKS  # noqa: E402

ENVS = ROOT / "envs"


def class_info(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    top_funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assigned = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    assigned.add(sub.attr)
    return classes, top_funcs, assigned


def assigned_attrs(task: str):
    classes, _, assigned = class_info(ENVS / f"{task}.py")
    for base in [ast.unparse(b) for c in classes for b in c.bases]:
        if base != "Base_Task" and (ENVS / f"{base}.py").exists():
            assigned |= class_info(ENVS / f"{base}.py")[2]
    assigned |= class_info(ENVS / "_base_task.py")[2]
    return assigned


def audit():
    errors, rows = [], []
    env_files = {p.stem for p in ENVS.glob("safe_*.py")}
    cfg_files = {p.stem for p in (ROOT / "task_config").glob("safe_*.yml")}
    ins_files = {p.stem for p in (ROOT / "description" / "task_instruction").glob("safe_*.json")}
    names = set(TASKS)
    if len(names) != 66:
        errors.append(f"expected 66 scenarios, registry has {len(names)}")
    for label, a, b in (("env files", env_files, names), ("task configs", cfg_files, names),
                        ("instruction files", ins_files, names)):
        if a != b:
            errors.append(f"{label} differ from the registry: extra={sorted(a - b)} missing={sorted(b - a)}")
    for name, spec in sorted(TASKS.items()):
        path = ENVS / f"{name}.py"
        try:
            classes, top_funcs, _ = class_info(path)
        except SyntaxError as exc:
            errors.append(f"{path.name}: does not parse ({exc})")
            continue
        if name not in [c.name for c in classes]:
            errors.append(f"{path.name}: class {name} not found")
        if top_funcs:
            errors.append(f"{path.name}: module-level functions {top_funcs} (method indentation?)")
        cfg = (ROOT / "task_config" / f"{name}.yml").read_text(encoding="utf-8")
        m = re.search(r"^task_name:\s*(\S+)", cfg, re.M)
        if not m or m.group(1) != name:
            errors.append(f"task_config/{name}.yml: task_name mismatch")
        if not (ROOT / "task_config" / f"{spec.collect_config}.yml").exists():
            errors.append(f"{name}: collect_config {spec.collect_config} missing")
        data = json.loads((ROOT / "description" / "task_instruction" / f"{name}.json").read_text(encoding="utf-8"))
        if not (data.get("full_description") and data.get("seen") and data.get("unseen")):
            errors.append(f"{name}.json: needs full_description, seen and unseen")
        if spec.suite not in SAFE_CORE_SUITES:
            errors.append(f"{name}: unknown suite {spec.suite}")
        assigned = assigned_attrs(name)
        for ch in spec.channels:
            if ch.category not in CHANNEL_CATEGORIES:
                errors.append(f"{name}/{ch.name}: unknown category {ch.category}")
            if not ch.eps > 0:
                errors.append(f"{name}/{ch.name}: eps must be > 0")
            for attr in ch.attrs:
                if attr not in assigned:
                    errors.append(f"{name}/{ch.name}: reads env.{attr}, which the scenario never assigns")
        if len(set(spec.channel_names())) != len(spec.channels):
            errors.append(f"{name}: duplicate channel names")
        rows.append((name, spec.suite, spec.kind, ", ".join(spec.channel_names())))
    counts = Counter(t.suite for t in TASKS.values())
    if set(counts) != set(SAFE_CORE_SUITES):
        errors.append(f"not all five Safe-Core suites are instantiated: {dict(counts)}")
    sub = Counter(TASKS[t].suite for t in DIAGNOSTIC_SUBSET)
    if len(DIAGNOSTIC_SUBSET) != 10 or set(sub) != set(SAFE_CORE_SUITES):
        errors.append(f"diagnostic subset must have 10 scenarios covering all suites: {dict(sub)}")
    return errors, rows, counts


def main():
    errors, rows, counts = audit()
    print(f"scenarios={len(rows)}")
    for s in SAFE_CORE_SUITES:
        print(f"suite[{s}]={counts.get(s, 0)}")
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print("  " + e)
        return 1
    print("audit: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
