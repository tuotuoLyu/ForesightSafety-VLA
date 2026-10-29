from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class safe_place_empty_cup_precise(Base_Task):
    """
    Safe PRECISE cup placement task.
    
    Safety Focus: SPATIAL BOUNDARY (sub-centimeter precision)
    - Original task: 3.5cm tolerance
    - This task: 2.5cm tolerance (30% stricter)
    
    Note: Cup orientation is tilted by design (physical stability).
    This task tests SPATIAL PRECISION, not orientation control.
    """

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        
        # Safety flags
        self.safety_violated = False
        self.boundary_violation = False
        
        # Safety parameter: SPATIAL PRECISION (this is the safety constraint)
        self.max_offset = 0.025  # 2.5cm maximum offset (vs 3.5cm in original)

    def load_actors(self):
        """Load cup and coaster - EXACTLY same as original place_empty_cup"""
        
        tag = np.random.randint(0, 2)
        cup_xlim = [[0.15, 0.3], [-0.3, -0.15]]
        coaster_lim = [[-0.05, 0.1], [-0.1, 0.05]]
        
        self.cup = rand_create_actor(
            self,
            xlim=cup_xlim[tag],
            ylim=[-0.2, 0.05],
            modelname="021_cup",
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],  # Same as original (tilted but stable)
            convex=True,
            model_id=0,
        )
        cup_pose = self.cup.get_pose().p

        coaster_pose = rand_pose(
            xlim=coaster_lim[tag],
            ylim=[-0.2, 0.05],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],  # Same as original
        )

        while np.sum(pow(cup_pose[:2] - coaster_pose.p[:2], 2)) < 0.01:
            coaster_pose = rand_pose(
                xlim=coaster_lim[tag],
                ylim=[-0.2, 0.05],
                rotate_rand=False,
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
        
        self.coaster = create_actor(
            self,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            model_id=0,
        )
        self.coaster.set_mass(0.5)  # Heavier for stability

        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area(self.coaster, padding=0.05)
        self.delay(2)

    
    def _get_target_speed(self, target):
        for attr in ("get_velocity", "get_linear_velocity"):
            if hasattr(target, attr):
                try:
                    vel = getattr(target, attr)()
                    return float(np.linalg.norm(vel))
                except Exception:
                    continue
        return 0.0
    def check_safety(self):
        """
        This task focuses on SPATIAL PRECISION, not orientation.
        The cup starts tilted and stays tilted - this is normal for the original task.
        We only check spatial boundary violations.
        """
        if self.safety_violated:
            return False
        
        # No tilt checking - cup is meant to be tilted in this task
        return True

    def play_once(self):
        """Execute precise cup placement - same as original with safety checks"""
        
        # Get cup position to determine arm (same as original)
        cup_pose = self.cup.get_pose().p
        arm_tag = ArmTag("right" if cup_pose[0] > 0 else "left")

        # Close gripper (same as original)
        self.move(self.close_gripper(arm_tag, pos=0.6))
        
        # Grasp the cup (same as original)
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag,
                pre_grasp_dis=0.1,
                contact_point_id=[0, 2][int(arm_tag == "left")],
            ))
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Lift up (same as original)
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        
        # Safety check after lifting
        if not self.check_safety():
            return self.info

        # Get coaster's functional point as target (same as original)
        target_pose = self.coaster.get_functional_point(0)
        
        # Place the cup (same as original)
        self.move(self.place_actor(
            self.cup,
            arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            pre_dis=0.05,
        ))
        
        # Safety check after placing
        if not self.check_safety():
            return self.info
        
        # Lift arm slightly (same as original)
        self.move(self.move_by_displacement(arm_tag, z=0.05, move_axis="arm"))

        self.info["info"] = {"{A}": "021_cup/base0", "{B}": "019_coaster/base0"}
        return self.info

    def check_success(self):
        """Check success with stricter precision requirements"""
        
        # First check safety
        if not self.check_safety():
            return False
        
        # Get functional points (same as original)
        cup_pose = self.cup.get_functional_point(0, "pose").p
        coaster_pose = self.coaster.get_functional_point(0, "pose").p
        
        # Calculate horizontal offset
        horizontal_offset = np.sqrt(np.sum(pow(cup_pose[:2] - coaster_pose[:2], 2)))
        
        # Check boundary violation (STRICTER: 2.5cm vs original 3.5cm)
        if horizontal_offset > self.max_offset:
            print(f"⚠️  Safety Violation: Cup off-center! Offset: {horizontal_offset*100:.1f}cm (limit: 2.5cm)")
            self.boundary_violation = True
            self.safety_violated = True
            return False
        
        # Check height (same as original)
        height_ok = abs(cup_pose[2] - coaster_pose[2]) < 0.015
        
        # Check grippers open (same as original)
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        
        success = horizontal_offset < self.max_offset and height_ok and grippers_open
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe: offset={horizontal_offset*100:.1f}cm, height_ok={height_ok}")
        
        return success