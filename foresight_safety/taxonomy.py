"""Safety taxonomy (paper Table I) and the three diagnostic dimensions (paper Sec. III-C, Fig. 1)."""

from __future__ import annotations

# Table I: 13 categories in three families; "unsafe when ..." copied from the paper.
TAXONOMY = {
    "Safe-Core": {
        "Force/Torque": "Contact force or torque exceeds allowable threshold",
        "Thermal/Energy": "End-effector or object enters heated/energized zone",
        "Spatial Boundary": "Clearance to obstacle, edge, or no-go zone falls below minimum",
        "Collaborative": "Dual-arm separation drops below safe threshold",
        "Temporal Sequence": "Action precondition violated or ordering constraint breached",
    },
    "Safe-Lang": {
        "Unsafe Instruction": "Instruction explicitly requests hazardous behavior",
        "Contradictory Instr.": "Safety constraints within command conflict",
        "Ambiguous Instr.": "Goal, constraint, or referent is underspecified",
        "Goal Hijacking": "Injected suffix overrides intended objective",
    },
    "Safe-Vis": {
        "Lighting & Material": "Illumination or texture change obscures hazard",
        "Perspective & Pose": "Viewpoint shift causes misjudged spatial relation",
        "Occlusion & Visibility": "Partial occlusion hides boundary or hazard",
        "Adversarial Patch": "Overlay induces unsafe downstream action",
    },
}

SAFE_CORE_SUITES = tuple(TAXONOMY["Safe-Core"])  # the five physical suites of the 66 base scenarios

# Channel categories used by the monitor (Sec. IV-A and IV-C): the five Safe-Core categories plus the
# negative-side-effect channels that are "treated as additional safety channels under the same
# dual-threshold framework".
CHANNEL_CATEGORIES = {
    "force_torque": "Force/Torque",
    "thermal_energy": "Thermal/Energy",
    "spatial_boundary": "Spatial Boundary",
    "collaborative": "Collaborative",
    "temporal_sequence": "Temporal Sequence",
    "side_effect": "Negative side effect",
}

# Scenario construction operators (Sec. III-B).
OPERATORS = ("hazard_injection", "constraint_tightening", "temporal_precondition")

# Diagnostic dimensions (Sec. III-C, Fig. 1). Level 0 is canonical on every axis.
STRUCTURE_LEVELS = {
    "L0": "Canonical",
    "L1": "Moderate: added clutter, reduced clearance",
    "L2": "Strong: dense clutter, tight clearance",
}
LANGUAGE_LEVELS = {
    "W0": "Canonical",
    "W1": "Paraphrase",
    "W2": "Compositional",
    "W3": "Ambiguous / unsafe request",
    "W4": "Injection",
}
VISUAL_LEVELS = {
    "V0": "Canonical",
    "V1": "Lighting",
    "V2": "Appearance",
    "V3": "Viewpoint + noise / occlusion",
    "V4": "Adversarial patch",
}

# Which Safe-Lang / Safe-Vis categories each evaluation-time overlay exercises (Sec. III-D: Safe-Lang and
# Safe-Vis are overlays on the base scenarios, analysed through the W and V dimensions).
LANGUAGE_LEVEL_CATEGORIES = {
    "W0": (),
    "W1": (),
    "W2": (),
    "W3": ("Ambiguous Instr.", "Contradictory Instr.", "Unsafe Instruction"),
    "W4": ("Goal Hijacking", "Unsafe Instruction"),
}
VISUAL_LEVEL_CATEGORIES = {
    "V0": (),
    "V1": ("Lighting & Material",),
    "V2": ("Lighting & Material",),
    "V3": ("Lighting & Material", "Perspective & Pose", "Occlusion & Visibility"),
    "V4": ("Lighting & Material", "Perspective & Pose", "Occlusion & Visibility", "Adversarial Patch"),
}


def normalise_level(value, axis: str) -> str:
    """Accept ``2``, ``"2"``, ``"l2"`` or ``"L2"`` and return ``"L2"`` (axis is ``"L"``, ``"W"`` or ``"V"``)."""
    table = {"L": STRUCTURE_LEVELS, "W": LANGUAGE_LEVELS, "V": VISUAL_LEVELS}[axis]
    text = str(value if value is not None else 0).strip().upper()
    if not text.startswith(axis):
        text = axis + text
    if text not in table:
        raise ValueError(f"unknown {axis}-level {value!r}; expected one of {sorted(table)}")
    return text


def count_categories() -> int:
    return sum(len(v) for v in TAXONOMY.values())
