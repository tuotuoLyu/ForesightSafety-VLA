"""Aggregate ``foresight_episodes.jsonl`` files into the paper's tables.

    python -m foresight_safety.report eval_result --out eval_result/foresight_report

Writes ``summary.json`` and ``tables.md`` with: Table II-style aggregates per policy (L0/W0/V0),
Table III-style per-suite breakdowns, Fig. 4-style SSR/CC per level of each axis, and Table IV-style
channel exposure for any episode recorded with ``--foresight_keep_trace``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .metrics import aggregate, group_by
from .taxonomy import SAFE_CORE_SUITES


def load_records(roots: Iterable[str]) -> List[Dict]:
    rows = []
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("foresight_episodes.jsonl"))
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _canonical(r) -> bool:
    return (r.get("structure_level"), r.get("language_level"), r.get("visual_level")) == ("L0", "W0", "V0")


def build_report(records: List[Dict]) -> Dict:
    scored = [r for r in records if not r.get("warmup")]
    canon = [r for r in scored if _canonical(r)]
    report = {
        "num_episodes": len(scored),
        "table_ii": group_by(canon, "policy_name"),
        "table_iii": {},
        "diagnostics": {},
    }
    by_policy: Dict[str, List[Dict]] = {}
    for r in canon:
        by_policy.setdefault(str(r.get("policy_name")), []).append(r)
    for pol, rows in sorted(by_policy.items()):
        report["table_iii"][pol] = group_by(rows, "safe_core_suite")
    for axis, key, fixed in (("structure", "structure_level", ("language_level", "visual_level")),
                             ("language", "language_level", ("structure_level", "visual_level")),
                             ("vision", "visual_level", ("structure_level", "language_level"))):
        rows = [r for r in scored if all(str(r.get(k, ""))[1:] == "0" for k in fixed)]
        per = {}
        for pol in sorted({str(r.get("policy_name")) for r in rows}):
            per[pol] = group_by([r for r in rows if str(r.get("policy_name")) == pol], key)
        report["diagnostics"][axis] = per
    traced = [r for r in scored if r.get("channel_RET")]
    report["exposure_examples"] = [
        {"task_name": r.get("task_name"), "policy_name": r.get("policy_name"), "episode": r.get("episode"),
         "outcome": r.get("outcome"), "RET/T": r.get("RET"), "channel_RET/T": r.get("channel_RET"),
         "channel_hard": r.get("channel_hard")}
        for r in traced if r.get("outcome") == "USR"][:20]
    return report


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def to_markdown(report: Dict) -> str:
    lines = ["# ForesightSafety-VLA report", "", f"Scored episodes: {report['num_episodes']}", "",
             "## Aggregate (L0/W0/V0), cf. Table II", "",
             "| Policy | N | SSR | USR | SFR | UFR | CC | RET/T | SASR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pol, a in report["table_ii"].items():
        lines.append(f"| {pol} | {a['N']} | " + " | ".join(_fmt(a[k]) for k in
                                                           ("SSR", "USR", "SFR", "UFR", "CC", "RET", "SASR")) + " |")
    lines += ["", "## Per Safe-Core suite, cf. Table III", ""]
    for pol, suites in report["table_iii"].items():
        lines += [f"### {pol}", "", "| Suite | N | SSR | CC | SASR |", "|---|---:|---:|---:|---:|"]
        for s in SAFE_CORE_SUITES:
            if s in suites:
                a = suites[s]
                lines.append(f"| {s} | {a['N']} | {_fmt(a['SSR'])} | {_fmt(a['CC'])} | {_fmt(a['SASR'])} |")
        lines.append("")
    lines += ["## Diagnostics by level, cf. Fig. 4 (SSR / CC)", ""]
    for axis, per in report["diagnostics"].items():
        for pol, levels in per.items():
            cells = ", ".join(f"{lv}: {_fmt(a['SSR'])} / {_fmt(a['CC'])}" for lv, a in levels.items())
            lines.append(f"- {axis}, {pol}: {cells}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="run directories or foresight_episodes.jsonl files")
    ap.add_argument("--out", default="eval_result/foresight_report")
    args = ap.parse_args(argv)
    report = build_report(load_records(args.roots))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "tables.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return report


if __name__ == "__main__":
    main()
