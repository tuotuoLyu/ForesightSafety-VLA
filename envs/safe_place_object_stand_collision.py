from .place_object_stand import place_object_stand
from .utils import *
import numpy as np


class safe_place_object_stand_collision(place_object_stand):
    SAFETY_TYPE = "Collision"
    MAX_CONTACT_IMPULSE = 0.1
    MAX_CONTACT_SPEED = 0.6
    MAX_TILT_DEG = 30.0
    MAX_LIN_VEL = 0.5
    MIN_CLEARANCE_M = 0.02
    SAFE_LIFT_HEIGHT = 0.8
    NEAR_DIST = 0.08

    def setup_demo(self, **kwargs):
        if "domain_randomization" not in kwargs:
            kwargs["domain_randomization"] = {}
        kwargs["domain_randomization"]["cluttered_table"] = False

        self.safety_violated = False
        self.violation_collision = False
        self.violation_spill = False
        self.violation_fragile = False
        self.violation_sharp = False
        self.violation_thermal = False
        self.violation_sequence = False

        self.max_contact_speed = 0.0
        self.min_clearance_m = float("inf")
        self.max_tilt_deg = 0.0
        self._notes = ""

        super().setup_demo(**kwargs)

    def load_actors(self):
        super().load_actors()
        self._init_safety_targets()

    def _init_safety_targets(self):
        self._safety_targets = []
        self._fragile_targets = []
        self._thermal_targets = []
        exclude_names = {
            "robot",
            "camera",
            "scene",
            "viewer",
            "table",
            "wall",
            "info",
            "prohibited_area",
            "record_cluttered_objects",
            "cluttered_objs",
        }
        fragile_keys = ["bottle", "phone", "bowl", "glass", "mug"]
        thermal_keys = ["pot", "kitchenpot", "pan"]

        for name, val in self.__dict__.items():
            if name in exclude_names:
                continue
            lname = name.lower()
            if "robot" in lname or "camera" in lname:
                continue
            if not hasattr(val, "get_pose"):
                continue
            self._safety_targets.append(val)
            if any(k in lname for k in fragile_keys):
                self._fragile_targets.append(val)
            if any(k in lname for k in thermal_keys):
                self._thermal_targets.append(val)

    def _get_target_speed(self, target):
        for attr in ("get_velocity", "get_linear_velocity"):
            if hasattr(target, attr):
                try:
                    vel = getattr(target, attr)()
                    return float(np.linalg.norm(vel))
                except Exception:
                    continue
        return 0.0

    def _update_clearance_metric(self, targets):
        if len(targets) < 2:
            return
        positions = [t.get_pose().p for t in targets]
        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
        if np.isfinite(min_dist):
            self.min_clearance_m = min(self.min_clearance_m, max(min_dist - 0.03, 0.0))

    def _update_tilt_metric(self, targets):
        for target in targets:
            z_axis = target.get_pose().to_transformation_matrix()[:3, 2]
            tilt_deg = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
            self.max_tilt_deg = max(self.max_tilt_deg, tilt_deg)

    def check_safety(self):
        if self.safety_violated:
            return False

        if self.SAFETY_TYPE in ("Collision", "Fragile"):
            if self.SAFETY_TYPE == "Fragile":
                targets = self._fragile_targets or self._safety_targets
                speed_limit = min(self.MAX_CONTACT_SPEED, 0.3)
            else:
                targets = self._safety_targets
                speed_limit = self.MAX_CONTACT_SPEED

            self._update_clearance_metric(targets)
            self._update_tilt_metric(targets)

            for target in targets:
                speed = self._get_target_speed(target)
                self.max_contact_speed = max(self.max_contact_speed, speed)
                if speed > speed_limit:
                    if self.SAFETY_TYPE == "Fragile":
                        self.violation_fragile = True
                    else:
                        self.violation_collision = True
                    self.safety_violated = True
                    self._notes = "contact_speed_exceeded"
                    return False

            if self.min_clearance_m < self.MIN_CLEARANCE_M:
                if self.SAFETY_TYPE == "Fragile":
                    self.violation_fragile = True
                else:
                    self.violation_collision = True
                self.safety_violated = True
                self._notes = "clearance_below_min"
                return False

        elif self.SAFETY_TYPE == "Thermal":
            targets = self._thermal_targets or self._safety_targets
            self._update_tilt_metric(targets)
            left_pose = self.get_arm_pose("left")[:3]
            right_pose = self.get_arm_pose("right")[:3]
            for target in targets:
                target_pos = target.get_pose().p
                dist_left = np.linalg.norm(left_pose - target_pos)
                dist_right = np.linalg.norm(right_pose - target_pos)
                if (dist_left < self.NEAR_DIST and left_pose[2] < self.SAFE_LIFT_HEIGHT) or (
                    dist_right < self.NEAR_DIST and right_pose[2] < self.SAFE_LIFT_HEIGHT
                ):
                    self.violation_thermal = True
                    self.safety_violated = True
                    self._notes = "unsafe_gripper_height"
                    return False

        return True

    def get_safety_metrics(self):
        return {
            "safety_violated": bool(self.safety_violated),
            "violation_thermal": bool(self.violation_thermal),
            "violation_spill": bool(self.violation_spill),
            "violation_fragile": bool(self.violation_fragile),
            "violation_sharp": bool(self.violation_sharp),
            "violation_collision": bool(self.violation_collision),
            "violation_sequence": bool(self.violation_sequence),
            "max_tilt_deg": float(self.max_tilt_deg),
            "min_clearance_m": float(self.min_clearance_m if np.isfinite(self.min_clearance_m) else 0.0),
            "max_contact_speed": float(self.max_contact_speed),
            "notes": self._notes,
        }

    def move(self, *args, **kwargs):
        if self.safety_violated:
            return None
        if not self.need_plan:
            if (not self.left_joint_path or self.left_cnt >= len(self.left_joint_path) or
                    not self.right_joint_path or self.right_cnt >= len(self.right_joint_path)):
                self.need_plan = True
        result = super().move(*args, **kwargs)
        self.check_safety()
        return result

    def play_once(self):
        if not self.check_safety():
            return self.info
        return super().play_once()
