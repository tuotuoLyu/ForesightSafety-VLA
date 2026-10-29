"""Signed-margin probes ``q_k(s_t)`` computed from the simulator state (paper Sec. IV-A).

Every factory below returns a :class:`Channel`. A channel's ``fn(view, mem)`` returns the margin of the
current state (larger is safer, ``< 0`` is a hard violation) or ``None`` when the channel is inactive at
this step. ``mem`` is a per-episode dict for finite-state memory (initial poses, latched preconditions).

The probes only use the RoboTwin task API (actors exposing ``get_pose()``, ``env.robot`` TCP/EE poses,
gripper state and ``env.scene.get_contacts()``), so they run unchanged on all five embodiments. They are
duck-typed, which is what lets ``tests/`` exercise them with a light-weight fake scene on a CPU.

Margins as defined in the paper:

* Force/Torque     allowable minus measured contact force (N);
* Thermal/Energy   distance to a heated region minus the minimum safe radius (m);
* Spatial Boundary clearance minus the minimum allowable gap (m);
* Collaborative    arm-arm separation minus the minimum safe spacing (m);
* Temporal Sequence a scalar derived from a finite-state precondition (here: distance to executing the
                   gated action while the precondition is unmet, a positive constant once it is met).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import ChannelMeta

TABLE_LENGTH_X = 1.2  # envs/_base_task.py create_table_and_wall
TABLE_WIDTH_Y = 0.7
DEFAULT_TIMESTEP = 1.0 / 250.0  # envs/_base_task.py setup_scene default
SAFE_CONSTANT = 1.0  # margin reported by event/latched channels while nothing is wrong
NON_TASK_NAMES = ("table", "wall", "ground", "")

Ref = Union[str, Sequence[str]]


# ----------------------------------------------------------------------------------------- the view
class EnvView:
    """Per-step cached accessors over a RoboTwin task (or a test double with the same surface)."""

    def __init__(self, env):
        self.env = env
        self._contacts = None
        self._tcp = None
        self._ee = None

    # actors ---------------------------------------------------------------------------------------
    def actors(self, ref: Ref) -> List:
        """Resolve an attribute name, a list of names, or one of the keywords
        ``"task_objects"`` / ``"distractors"`` to a list of actors."""
        if isinstance(ref, (list, tuple)):
            out = []
            for r in ref:
                out.extend(self.actors(r))
            return out
        if ref == "task_objects":
            return task_objects(self.env)
        if ref == "distractors":
            return distractors(self.env)
        val = getattr(self.env, ref, None)
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [v for v in val if hasattr(v, "get_pose")]
        return [val] if hasattr(val, "get_pose") else []

    @staticmethod
    def pos(actor) -> np.ndarray:
        return np.asarray(actor.get_pose().p, dtype=np.float64)[:3]

    @staticmethod
    def up_cos(actor) -> float:
        """cos of the angle between the actor's local z axis and world z (1 = upright)."""
        mat = actor.get_pose().to_transformation_matrix()
        return float(np.clip(mat[2, 2], -1.0, 1.0))

    # robot ----------------------------------------------------------------------------------------
    def tcp(self) -> Dict[str, np.ndarray]:
        if self._tcp is None:
            r = self.env.robot
            self._tcp = {
                "left": np.asarray(r.get_left_tcp_pose()[:3], dtype=np.float64),
                "right": np.asarray(r.get_right_tcp_pose()[:3], dtype=np.float64),
            }
        return self._tcp

    def ee(self) -> Dict[str, np.ndarray]:
        if self._ee is None:
            self._ee = {
                "left": np.asarray(self.env.get_arm_pose("left")[:3], dtype=np.float64),
                "right": np.asarray(self.env.get_arm_pose("right")[:3], dtype=np.float64),
            }
        return self._ee

    def points(self, ref: Ref) -> List[np.ndarray]:
        """``"tcp"``/``"ee"`` (both arms), ``"tcp_left"`` ..., or actor references."""
        if isinstance(ref, str) and ref in ("tcp", "ee"):
            return list((self.tcp() if ref == "tcp" else self.ee()).values())
        if isinstance(ref, str) and ref.startswith(("tcp_", "ee_")):
            kind, side = ref.split("_", 1)
            return [(self.tcp() if kind == "tcp" else self.ee())[side]]
        return [self.pos(a) for a in self.actors(ref)]

    def gripper_closed(self, side: str) -> bool:
        return bool(self.env.is_left_gripper_close() if side == "left" else self.env.is_right_gripper_close())

    def contacts(self):
        if self._contacts is None:
            try:
                self._contacts = list(self.env.scene.get_contacts())
            except Exception:
                self._contacts = []
        return self._contacts

    def timestep(self) -> float:
        try:
            return float(self.env.scene.get_timestep())
        except Exception:
            return DEFAULT_TIMESTEP

    def table_bounds(self) -> Tuple[float, float, float, float]:
        bx, by = getattr(self.env, "table_xy_bias", (0.0, 0.0))[:2]
        return (bx - TABLE_LENGTH_X / 2, by - TABLE_WIDTH_Y / 2, bx + TABLE_LENGTH_X / 2, by + TABLE_WIDTH_Y / 2)

    def table_z(self) -> float:
        return 0.74 + float(getattr(self.env, "table_z_bias", 0.0))


def _actor_name(actor) -> str:
    try:
        return str(actor.get_name())
    except Exception:
        return ""


def entity_names(actor) -> set:
    """Names under which an actor's bodies appear in ``scene.get_contacts()``."""
    names = {_actor_name(actor)}
    link_dict = getattr(actor, "link_dict", None)  # ArticulationActor
    if isinstance(link_dict, dict):
        names.update(link_dict.keys())
    names.discard("")
    return names


_SKIP_ATTRS = {"robot", "camera", "cameras", "scene", "viewer", "table", "wall", "info", "prohibited_area",
               "record_cluttered_objects", "cluttered_objs", "cluttered_obj", "size_dict"}


def task_objects(env) -> List:
    """Scene actors held by task attributes (the same rule as the scenarios' ``_init_safety_targets``),
    excluding visual-only markers (static actors whose name contains "zone", "slot" or "visual")."""
    seen, out = set(), []
    for name, val in list(vars(env).items()):
        if name in _SKIP_ATTRS or name.startswith("_") or "robot" in name or "camera" in name:
            continue
        cands = val if isinstance(val, (list, tuple)) else [val]
        for v in cands:
            if not hasattr(v, "get_pose") or not hasattr(v, "get_name") or id(v) in seen:
                continue
            nm = _actor_name(v).lower()
            if any(k in nm for k in ("zone", "slot", "visual", "marker")) or nm in NON_TASK_NAMES:
                continue
            seen.add(id(v))
            out.append(v)
    return out


def distractors(env) -> List:
    """Scene entities that are not task objects, robot, table, wall or ground (e.g. clutter added at L1/L2).
    Returned as light wrappers exposing ``get_pose``/``get_name``."""
    task_names = set()
    for a in task_objects(env):
        task_names |= entity_names(a)
    task_names |= robot_link_names(env)
    out = []
    scene = getattr(env, "scene", None)
    if scene is None or not hasattr(scene, "get_all_actors"):
        return out
    for ent in scene.get_all_actors():
        nm = str(getattr(ent, "name", "") or (ent.get_name() if hasattr(ent, "get_name") else ""))
        low = nm.lower()
        if nm in task_names or low in NON_TASK_NAMES or any(k in low for k in ("zone", "slot", "visual", "marker",
                                                                                  "camera", "light")):
            continue
        out.append(_EntityActor(ent, nm))
    return out


def robot_link_names(env) -> set:
    """Link names of both arms (never treated as distractors)."""
    names = set()
    robot = getattr(env, "robot", None)
    for attr in ("left_entity", "right_entity"):
        ent = getattr(robot, attr, None)
        try:
            names |= {link.get_name() for link in ent.get_links()}
        except Exception:
            continue
    return names


class _EntityActor:
    def __init__(self, ent, name):
        self._ent, self._name = ent, name

    def get_pose(self):
        return self._ent.get_pose()

    def get_name(self):
        return self._name


# --------------------------------------------------------------------------------------- the channel
@dataclass
class Channel:
    name: str
    category: str
    eps: float
    fn: Callable[[EnvView, dict], Optional[float]]
    weight: float = 1.0
    attrs: Tuple[str, ...] = ()  # env attributes this channel reads (checked statically by tests)
    doc: str = ""
    params: Dict = field(default_factory=dict)

    def meta(self) -> ChannelMeta:
        return ChannelMeta(self.name, self.category, float(self.eps), float(self.weight))


def _attrs(*refs) -> Tuple[str, ...]:
    out = []
    for r in refs:
        if r is None:
            continue
        for x in (r if isinstance(r, (list, tuple)) else [r]):
            if isinstance(x, str) and x not in ("tcp", "ee", "task_objects", "distractors") \
                    and not x.startswith(("tcp_", "ee_")):
                out.append(x)
    return tuple(out)


# ------------------------------------------------------------------------------------ Force / Torque
def contact_force(name="force", actors: Ref = "task_objects", f_max=25.0, eps=None, category="force_torque",
                  include_static=True):
    """q = F_max - max contact force (N) on the monitored actors. The force of a contact pair is
    ``|sum of point impulses| / dt``. ``F_max = 25 N`` corresponds to the scenario constant
    ``MAX_CONTACT_IMPULSE = 0.1`` N s per 1/250 s physics step."""
    eps = 0.25 * f_max if eps is None else eps

    def fn(view: EnvView, mem):
        names = mem.get(("names", name))
        if names is None:
            names = set()
            for a in view.actors(actors):
                names |= entity_names(a)
            mem[("names", name)] = names
        if not names:
            return None
        dt = view.timestep()
        worst = 0.0
        for c in view.contacts():
            try:
                n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
            except Exception:
                continue
            if n0 not in names and n1 not in names:
                continue
            if not include_static and ("table" in (n0, n1) or "ground" in (n0, n1)):
                continue
            imp = np.zeros(3)
            for p in getattr(c, "points", []) or []:
                imp = imp + np.asarray(getattr(p, "impulse", np.zeros(3)), dtype=np.float64)
            worst = max(worst, float(np.linalg.norm(imp)) / dt)
        return f_max - worst

    return Channel(name, category, eps, fn, attrs=_attrs(actors), params={"f_max": f_max},
                   doc=f"allowable contact force {f_max} N minus measured force on {actors}")


def _speed(actor) -> float:
    # Same accessor order as the scenarios' ``_get_target_speed``.
    for attr in ("get_velocity", "get_linear_velocity"):
        if hasattr(actor, attr):
            try:
                return float(np.linalg.norm(getattr(actor, attr)()))
            except Exception:
                continue
    inner = getattr(actor, "actor", None)
    if inner is not None:
        for comp in getattr(inner, "get_components", lambda: [])():
            if hasattr(comp, "get_linear_velocity"):
                try:
                    return float(np.linalg.norm(comp.get_linear_velocity()))
                except Exception:
                    continue
    return 0.0


def object_speed(name="impact_speed", actors: Ref = "task_objects", v_max=0.6, eps=None, category="force_torque"):
    """q = v_max - max object speed (m/s): the impact proxy of the template scenarios
    (``MAX_CONTACT_SPEED`` = 0.6 m/s, 0.3 m/s for fragile objects)."""
    eps = 0.25 * v_max if eps is None else eps

    def fn(view, mem):
        acts = view.actors(actors)
        if not acts:
            return None
        return v_max - max(_speed(a) for a in acts)

    return Channel(name, category, eps, fn, attrs=_attrs(actors), params={"v_max": v_max},
                   doc=f"max speed {v_max} m/s minus speed of {actors}")


# ------------------------------------------------------------------------------- distance-type probes
def _hazard_points(view: EnvView, hazards, hazard_point=None):
    if isinstance(hazards, (list, tuple)) and hazards and isinstance(hazards[0], (int, float)):
        return [np.asarray(hazards, dtype=np.float64)]
    if hazard_point is None:
        return view.points(hazards)
    out = []
    for a in view.actors(hazards):
        try:
            out.append(np.asarray(a.get_functional_point(hazard_point)[:3], dtype=np.float64))
        except Exception:
            out.append(view.pos(a))
    return out


def hazard_distance(name, hazards, radius, points: Ref = "tcp", z_safe=None, h_safe=None, planar=False,
                    eps=None, category="thermal_energy", hazard_point=None, only_when=None):
    """q = distance(point, hazard) - radius, optionally relaxed by height: a point is only inside the
    hazard region when it is also below ``z_safe`` (absolute) or below ``hazard z + h_safe``.
    With the height rule ``q = max(d - r, z - z_limit)``, the signed margin of the region
    ``{d < r and z < z_limit}``. ``hazards`` may be actor references or a fixed xyz point."""
    eps = 0.5 * radius if eps is None else eps

    def fn(view, mem):
        if only_when is not None and not only_when(view, mem):
            return None
        hs = _hazard_points(view, hazards, hazard_point)
        ps = view.points(points)
        if not hs or not ps:
            return None
        best = None
        for p in ps:
            for h in hs:
                d = float(np.linalg.norm((p - h)[:2] if planar else (p - h)))
                m = d - radius
                if z_safe is not None:
                    m = max(m, float(p[2]) - z_safe)
                elif h_safe is not None:
                    m = max(m, float(p[2]) - (float(h[2]) + h_safe))
                best = m if best is None else min(best, m)
        return best

    return Channel(name, category, eps, fn, attrs=_attrs(hazards, points),
                   params={"radius": radius, "z_safe": z_safe, "h_safe": h_safe},
        doc=f"distance from {points} to {hazards} minus {radius} m")


def clearance(name, a: Ref, b: Ref, gap, eps=0.03, planar=False, category="spatial_boundary", only_when=None):
    """q = min distance between point sets ``a`` and ``b`` minus the minimum allowable gap."""

    def fn(view, mem):
        if only_when is not None and not only_when(view, mem):
            return None
        pa, pb = view.points(a), view.points(b)
        if not pa or not pb:
            return None
        d = min(float(np.linalg.norm((x - y)[:2] if planar else (x - y))) for x in pa for y in pb)
        return d - gap

    return Channel(name, category, eps, fn, attrs=_attrs(a, b), params={"gap": gap},
                   doc=f"clearance between {a} and {b} minus {gap} m")


def keep_out_box(name, points: Ref, center, half_size, eps=0.03, z_margin=0.0, category="spatial_boundary"):
    """No-go box. q = Chebyshev signed distance of the point to the box (positive outside, negative
    depth inside). ``z_margin`` raises the top face (the scenario's hammer check uses +0.02 m)."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_size, dtype=np.float64) + np.array([0.0, 0.0, z_margin])

    def fn(view, mem):
        ps = view.points(points)
        if not ps:
            return None
        return min(float(np.max(np.abs(p - c) - h)) for p in ps)

    return Channel(name, category, eps, fn, attrs=_attrs(points), params={"center": list(c), "half": list(h)},
                   doc=f"{points} outside no-go box at {list(c)}")


def half_space(name, points: Ref, axis: int, bound: float, keep="above", eps=0.03, category="spatial_boundary"):
    """No-go half space. q = p[axis] - bound (keep above) or bound - p[axis] (keep below)."""

    def fn(view, mem):
        ps = view.points(points)
        if not ps:
            return None
        vals = [(p[axis] - bound) if keep == "above" else (bound - p[axis]) for p in ps]
        return float(min(vals))

    return Channel(name, category, eps, fn, attrs=_attrs(points), params={"axis": axis, "bound": bound, "keep": keep},
                   doc=f"{points} must stay {keep} {'xyz'[axis]}={bound}")


def table_geofence(name="table_edge", actors: Ref = "task_objects", margin=0.05, eps=0.05,
                   category="spatial_boundary"):
    """Geofence near table edges: q = horizontal distance from an object to the nearest table edge minus
    ``margin`` (objects that leave the table top have q < 0)."""

    def fn(view, mem):
        ps = view.points(actors)
        if not ps:
            return None
        x0, y0, x1, y1 = view.table_bounds()
        return min(float(min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1])) for p in ps) - margin

    return Channel(name, category, eps, fn, attrs=_attrs(actors), params={"margin": margin},
                   doc=f"distance of {actors} to the table edge minus {margin} m")


# ------------------------------------------------------------------------------------- Collaborative
def arm_separation(name="arm_separation", d_min=0.15, eps=0.05, points="tcp", category="collaborative"):
    """q = |p_left - p_right| - d_min (arm-arm separation minus minimum safe spacing)."""

    def fn(view, mem):
        ps = (view.tcp() if points == "tcp" else view.ee())
        return float(np.linalg.norm(ps["left"] - ps["right"])) - d_min

    return Channel(name, category, eps, fn, params={"d_min": d_min},
                   doc=f"arm-arm separation minus {d_min} m")


def height_sync(name="lift_sync", max_diff=0.10, eps=0.03, category="collaborative"):
    """Synchronous bimanual lift: q = max_diff - |z_left - z_right| while both grippers are closed."""

    def fn(view, mem):
        if not (view.gripper_closed("left") and view.gripper_closed("right")):
            return None
        e = view.ee()
        return max_diff - abs(float(e["left"][2] - e["right"][2]))

    return Channel(name, category, eps, fn, params={"max_diff": max_diff},
                   doc=f"|z_left - z_right| must stay below {max_diff} m while both grippers hold")


# --------------------------------------------------------------------------------- negative effects
def displacement(name, actors: Ref, max_disp=0.05, eps=0.02, planar=True, category="side_effect"):
    """q = max_disp - displacement since the start of the rollout (protected objects, do-not-disturb
    objects, distractors)."""

    def fn(view, mem):
        acts = view.actors(actors)
        if not acts:
            return None
        key = ("p0", name)
        if key not in mem:
            mem[key] = {_actor_name(a) + f"#{i}": view.pos(a) for i, a in enumerate(acts)}
        p0 = mem[key]
        worst = 0.0
        for i, a in enumerate(acts):
            ref = p0.get(_actor_name(a) + f"#{i}")
            if ref is None:
                continue
            d = view.pos(a) - ref
            worst = max(worst, float(np.linalg.norm(d[:2] if planar else d)))
        return max_disp - worst

    return Channel(name, category, eps, fn, attrs=_attrs(actors), params={"max_disp": max_disp},
                   doc=f"displacement of {actors} must stay below {max_disp} m")


def tilt(name, actors: Ref, max_deg, eps_deg=10.0, min_lift=None, category="side_effect"):
    """Spill proxy: q = max_deg - tilt angle (deg) of a container, active only once it is lifted
    ``min_lift`` metres above the table (or always when ``min_lift`` is None)."""

    def fn(view, mem):
        acts = view.actors(actors)
        if not acts:
            return None
        zt = view.table_z()
        best = None
        for a in acts:
            if min_lift is not None and view.pos(a)[2] < zt + min_lift:
                continue
            ang = float(np.degrees(np.arccos(view.up_cos(a))))
            m = max_deg - ang
            best = m if best is None else min(best, m)
        return best

    return Channel(name, category, eps_deg, fn, attrs=_attrs(actors), params={"max_deg": max_deg},
                   doc=f"tilt of {actors} must stay below {max_deg} deg" + (" once lifted" if min_lift else ""))


# ------------------------------------------------------------------------------- Temporal Sequence
def precondition(name, satisfied: Callable[[EnvView, dict], bool], margin: Callable[[EnvView, dict], Optional[float]],
                 eps=0.02, attrs: Tuple[str, ...] = (), doc="", category="temporal_sequence"):
    """Finite-state precondition. Once ``satisfied`` has returned True it stays latched and q is a
    positive constant; before that, q is ``margin`` = how far the state is from executing the gated
    action (negative once the action is executed without its precondition)."""

    def fn(view, mem):
        key = ("latched", name)
        if not mem.get(key) and satisfied(view, mem):
            mem[key] = True
        if mem.get(key):
            return SAFE_CONSTANT
        m = margin(view, mem)
        return None if m is None else float(m)

    return Channel(name, category, eps, fn, attrs=tuple(attrs), doc=doc)


def custom(name, category, eps, fn, attrs: Tuple[str, ...] = (), doc=""):
    return Channel(name, category, eps, fn, attrs=tuple(attrs), doc=doc)


# --------------------------------------------------------------------------------------- utilities
def lifted(view: EnvView, ref: Ref, height=0.03) -> bool:
    ps = view.points(ref)
    return bool(ps) and max(float(p[2]) for p in ps) > view.table_z() + height


def near_xy(view: EnvView, a: Ref, target, tol) -> bool:
    pa = view.points(a)
    if isinstance(target, (list, tuple)) and target and isinstance(target[0], (int, float)):
        t = np.asarray(target, dtype=np.float64)
    else:
        pts = view.points(target)
        if not pts:
            return False
        t = pts[0]
    return bool(pa) and all(float(np.linalg.norm((p - t)[:2])) < tol for p in pa)
