"""A light-weight stand-in for a RoboTwin task, used by the CPU tests and the CPU smoke run.

It exposes the subset of the task API the probes read (actor poses and velocities, TCP/EE poses, gripper
state, contacts, table geometry). It does no physics; callers move things by setting poses.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class FakePose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)

    def to_transformation_matrix(self):
        m = np.eye(4)
        m[:3, :3] = _quat_to_mat(self.q)
        m[:3, 3] = self.p
        return m


def tilt_quat(deg: float):
    """Quaternion (w, x, y, z) of a rotation by ``deg`` about the x axis."""
    h = np.deg2rad(deg) / 2
    return (np.cos(h), np.sin(h), 0.0, 0.0)


class FakeActor:
    def __init__(self, name, p, q=(1.0, 0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
        self.name = name
        self.pose = FakePose(p, q)
        self.velocity = np.asarray(velocity, dtype=np.float64)

    def get_pose(self):
        return self.pose

    def get_name(self):
        return self.name

    def get_linear_velocity(self):
        return self.velocity

    def get_functional_point(self, idx, ret="list"):
        return list(self.pose.p) + list(self.pose.q)

    def set(self, p=None, q=None, velocity=None):
        if p is not None:
            self.pose = FakePose(p, self.pose.q if q is None else q)
        elif q is not None:
            self.pose = FakePose(self.pose.p, q)
        if velocity is not None:
            self.velocity = np.asarray(velocity, dtype=np.float64)


class _Body:
    def __init__(self, name):
        self.entity = type("Entity", (), {"name": name})()


class _Point:
    def __init__(self, impulse):
        self.impulse = np.asarray(impulse, dtype=np.float64)


class FakeContact:
    def __init__(self, name0, name1, impulses):
        self.bodies = [_Body(name0), _Body(name1)]
        self.points = [_Point(i) for i in impulses]


class FakeScene:
    def __init__(self, timestep=1 / 250):
        self.timestep = timestep
        self.contacts: List[FakeContact] = []
        self.extra_entities: List[FakeActor] = []

    def get_timestep(self):
        return self.timestep

    def get_contacts(self):
        return list(self.contacts)

    def get_all_actors(self):
        return list(self.extra_entities)


class FakeRobot:
    def __init__(self):
        self.tcp = {"left": np.array([-0.3, -0.2, 0.95]), "right": np.array([0.3, -0.2, 0.95])}
        self.gripper = {"left": 1.0, "right": 1.0}

    def get_left_tcp_pose(self):
        return list(self.tcp["left"]) + [1, 0, 0, 0]

    def get_right_tcp_pose(self):
        return list(self.tcp["right"]) + [1, 0, 0, 0]

    def get_left_gripper_val(self):
        return self.gripper["left"]

    def get_right_gripper_val(self):
        return self.gripper["right"]


class FakeTask:
    """Attributes mirror ``Base_Task``: actors are plain attributes, ``robot``, ``scene``, table biases."""

    def __init__(self, actors: Optional[Dict[str, object]] = None, table_xy_bias=(0.0, 0.0), table_z_bias=0.0):
        self.scene = FakeScene()
        self.robot = FakeRobot()
        self.table_xy_bias = list(table_xy_bias)
        self.table_z_bias = table_z_bias
        for k, v in (actors or {}).items():
            setattr(self, k, v)

    def get_arm_pose(self, arm_tag):
        return list(self.robot.tcp[str(arm_tag)]) + [1, 0, 0, 0]

    def is_left_gripper_close(self):
        return self.robot.gripper["left"] < 0.2

    def is_right_gripper_close(self):
        return self.robot.gripper["right"] < 0.2

    def is_left_gripper_open(self):
        return self.robot.gripper["left"] > 0.8

    def is_right_gripper_open(self):
        return self.robot.gripper["right"] > 0.8
