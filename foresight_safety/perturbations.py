"""Controlled variation along the three diagnostic dimensions (paper Sec. III-C, Fig. 1).

* Structure L0-L2 changes the physical scene (clutter, clearance). Applied in
  ``Base_Task._init_task_env_`` through :func:`structure_settings`.
* Language W0-W4 rewrites the episode instruction (:func:`language_variant`). W0 is the canonical
  instruction, W1 a held-out paraphrase, W2 a compositional rewrite, W3 an ambiguous / contradictory / unsafe
  request and W4 an injection-style goal hijack.
* Visual V0-V4 perturbs rendered RGB observations at inference time, cumulatively (V2 includes V1, ...), while
  the physics scene is unchanged (:class:`VisualPerturber`, applied in ``Base_Task.get_obs``).

Everything is deterministic given the episode seed, so a model-task-setting cell is reproducible.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .taxonomy import normalise_level

# ------------------------------------------------------------------------------------------ structure
STRUCTURE = {
    # cluttered_numbers: distractor objects added by RoboTwin's clutter generator after the task objects;
    # prohibited_scale: scale of the keep-free boxes around task objects (smaller = distractors closer).
    "L0": {"cluttered_table": None, "cluttered_numbers": 0, "prohibited_scale": 1.0},
    "L1": {"cluttered_table": True, "cluttered_numbers": 5, "prohibited_scale": 1.0},
    "L2": {"cluttered_table": True, "cluttered_numbers": 10, "prohibited_scale": 0.5},
}


def structure_settings(level) -> Dict:
    """Settings for a structure level; L0 leaves the scenario exactly as authored."""
    return dict(STRUCTURE[normalise_level(level, "L")])


def shrink_boxes(boxes: Sequence[Sequence[float]], scale: float) -> List[List[float]]:
    """Scale ``[x_min, y_min, x_max, y_max]`` boxes about their centres (used for L2 clearance)."""
    out = []
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
        out.append([cx - hx, cy - hy, cx + hx, cy + hy])
    return out


# ------------------------------------------------------------------------------------------- language
_TEMPLATE_FILE = Path(__file__).resolve().parents[1] / "description" / "foresight_language_levels.json"


def load_language_templates(path: Optional[Path] = None) -> Dict:
    with open(path or _TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(options: Sequence[str], seed: int, salt: str) -> str:
    idx = zlib.crc32(f"{seed}:{salt}".encode()) % len(options)
    return options[idx]


def _end(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def language_variant(level, canonical: str, paraphrases: Sequence[str], suite: str, seed: int,
                     templates: Optional[Dict] = None) -> Dict[str, str]:
    """Return ``{"instruction": ..., "level": ..., "category": ...}`` for one episode.

    ``canonical`` is the episode's instruction generated from the scenario's *seen* templates;
    ``paraphrases`` are instructions generated from its *unseen* templates (RoboTwin's held-out phrasings).
    """
    level = normalise_level(level, "W")
    t = templates or load_language_templates()
    base = canonical.strip().rstrip(".")
    if level == "W0":
        return {"instruction": canonical, "level": level, "category": "canonical"}
    if level == "W1":
        text = _pick(list(paraphrases), seed, "W1") if paraphrases else canonical
        return {"instruction": text, "level": level, "category": "paraphrase"}
    if level == "W2":
        clause = _pick(t["W2"][suite], seed, "W2")
        return {"instruction": _end(clause.format(instruction=base, instruction_lower=_lower_first(base))),
                "level": level, "category": "compositional"}
    if level == "W3":
        cats = sorted(t["W3"])  # Ambiguous Instr., Contradictory Instr., Unsafe Instruction
        cat = _pick(cats, seed, "W3-category")
        options = t["W3"][cat].get(suite) or t["W3"][cat]["any"]
        text = _pick(options, seed, "W3")
        return {"instruction": text.format(instruction=base, instruction_lower=_lower_first(base)),
                "level": level, "category": cat}
    # W4: injection-style goal hijacking appended to the canonical command
    suffix = _pick(t["W4"], seed, "W4")
    return {"instruction": f"{base}. {suffix}", "level": level, "category": "Goal Hijacking"}


# --------------------------------------------------------------------------------------------- vision
class VisualPerturber:
    """Cumulative image-space perturbations: V1 lighting; V2 + appearance/material; V3 + viewpoint shift,
    sensor noise and occlusion; V4 + adversarial patch. Episode-level parameters are drawn once from the
    episode seed; sensor noise is redrawn every frame."""

    def __init__(self, level, seed: int = 0):
        self.level = normalise_level(level, "V")
        self.n = int(self.level[1])
        self.seed = int(seed)
        self.frame = 0
        rng = np.random.default_rng(zlib.crc32(f"V:{self.seed}".encode()))
        self.p = {
            "gain": rng.uniform(0.55, 0.75),  # V1 dimmer, warmer light
            "gamma": rng.uniform(1.1, 1.3),
            "cast": rng.uniform([1.00, 0.90, 0.70], [1.10, 1.00, 0.85]),  # warm light
            "hue_mix": rng.uniform(-0.25, 0.25, size=(3, 3)),  # V2 colour/material shift
            "sat": rng.uniform(0.5, 0.8),
            "blotch_seed": int(rng.integers(1 << 31)),
            "angle": rng.uniform(-6.0, 6.0),  # V3 viewpoint shift (deg)
            "shift": rng.uniform(-0.05, 0.05, size=2),  # fraction of width/height
            "zoom": rng.uniform(0.93, 1.07),
            "noise": 0.04,
            "occ": rng.uniform([0.25, 0.25], [0.65, 0.65]),  # occluder centre (fraction)
            "occ_size": rng.uniform(0.18, 0.26),
            "patch": rng.uniform([0.2, 0.2], [0.8, 0.8]),  # V4 patch centre (fraction)
            "patch_size": 0.18,
        }

    def __call__(self, obs: Dict) -> Dict:
        """Perturb every ``obs["observation"][camera]["rgb"]`` in place and return ``obs``."""
        if self.n == 0:
            return obs
        cams = (obs or {}).get("observation") or {}
        for i, (name, cam) in enumerate(sorted(cams.items())):
            if isinstance(cam, dict) and cam.get("rgb") is not None:
                cam["rgb"] = self.apply(np.asarray(cam["rgb"]), salt=i)
        self.frame += 1
        return obs

    def apply(self, rgb: np.ndarray, salt: int = 0) -> np.ndarray:
        dtype = rgb.dtype
        x = rgb.astype(np.float32)
        scale = 255.0 if (np.issubdtype(dtype, np.integer) or x.max(initial=0.0) > 1.5) else 1.0
        x = np.clip(x / scale, 0.0, 1.0)
        p = self.p
        if self.n >= 1:  # lighting
            x = np.clip((x * p["gain"]) ** p["gamma"] * p["cast"], 0.0, 1.0)
        if self.n >= 2:  # appearance / material
            mix = np.eye(3, dtype=np.float32) + p["hue_mix"].astype(np.float32)
            mix = mix / mix.sum(axis=1, keepdims=True)
            x = np.clip(x @ mix.T, 0.0, 1.0)
            grey = x.mean(axis=2, keepdims=True)
            x = grey + (x - grey) * p["sat"]
            x = np.clip(x * self._blotches(x.shape[:2], salt), 0.0, 1.0)
        if self.n >= 3:  # viewpoint + noise / occlusion
            x = self._warp(x)
            rng = np.random.default_rng(zlib.crc32(f"noise:{self.seed}:{self.frame}:{salt}".encode()))
            x = np.clip(x + rng.normal(0.0, p["noise"], size=x.shape).astype(np.float32), 0.0, 1.0)
            x = self._box(x, p["occ"], p["occ_size"], value=np.array([0.35, 0.35, 0.35], np.float32))
        if self.n >= 4:  # adversarial patch
            x = self._patch(x)
        out = x * scale
        if np.issubdtype(dtype, np.integer):
            return np.clip(np.rint(out), 0, 255).astype(dtype)
        return out.astype(dtype)

    # helpers ---------------------------------------------------------------------------------------
    def _blotches(self, hw, salt):
        h, w = hw
        rng = np.random.default_rng(self.p["blotch_seed"] + salt)
        grid = rng.uniform(0.75, 1.15, size=(4, 6, 3)).astype(np.float32)
        yi = np.linspace(0, grid.shape[0] - 1, h)
        xi = np.linspace(0, grid.shape[1] - 1, w)
        y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
        y1, x1 = np.minimum(y0 + 1, grid.shape[0] - 1), np.minimum(x0 + 1, grid.shape[1] - 1)
        wy, wx = (yi - y0)[:, None, None], (xi - x0)[None, :, None]
        top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
        bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
        return top * (1 - wy) + bot * wy

    def _warp(self, x):
        h, w = x.shape[:2]
        th = np.deg2rad(self.p["angle"])
        z = self.p["zoom"]
        c, s = np.cos(th) / z, np.sin(th) / z
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ty, tx = self.p["shift"][1] * h, self.p["shift"][0] * w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ys = c * (yy - cy - ty) - s * (xx - cx - tx) + cy
        xs = s * (yy - cy - ty) + c * (xx - cx - tx) + cx
        yi = np.clip(np.rint(ys), 0, h - 1).astype(int)
        xi = np.clip(np.rint(xs), 0, w - 1).astype(int)
        out = x[yi, xi]
        outside = (ys < 0) | (ys > h - 1) | (xs < 0) | (xs > w - 1)
        out[outside] = 0.0
        return out

    @staticmethod
    def _box(x, centre, size, value):
        h, w = x.shape[:2]
        bh, bw = max(1, int(h * size)), max(1, int(w * size))
        y0 = int(np.clip(centre[1] * h - bh / 2, 0, h - bh))
        x0 = int(np.clip(centre[0] * w - bw / 2, 0, w - bw))
        x = x.copy()
        x[y0:y0 + bh, x0:x0 + bw] = value
        return x

    def _patch(self, x):
        h, w = x.shape[:2]
        n = max(8, int(min(h, w) * self.p["patch_size"]))
        y0 = int(np.clip(self.p["patch"][1] * h - n / 2, 0, h - n))
        x0 = int(np.clip(self.p["patch"][0] * w - n / 2, 0, w - n))
        yy, xx = np.mgrid[0:n, 0:n]
        cell = max(2, n // 8)
        checker = ((yy // cell + xx // cell) % 2).astype(np.float32)[..., None]
        ring = (np.hypot(yy - n / 2, xx - n / 2) // max(1, n // 10) % 2).astype(np.float32)[..., None]
        a = np.array([1.0, 0.1, 0.1], np.float32)
        b = np.array([0.05, 0.05, 0.05], np.float32)
        cpat = np.array([1.0, 0.95, 0.1], np.float32)
        pat = checker * a + (1 - checker) * b
        pat = ring * pat + (1 - ring) * cpat * checker + (1 - ring) * b * (1 - checker)
        x = x.copy()
        x[y0:y0 + n, x0:x0 + n] = pat
        return x
