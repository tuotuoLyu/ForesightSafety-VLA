from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_lift_hotpot_thermal(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table for safety focus
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.spill_violated = False
        self.burn_violated = False
        
        # Safety parameters
        self.tilt_threshold = 0.85  # Z-axis projection > 0.85 (~32 degrees), keep level
        self.safe_height = 0.88     # Target lifting height
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Load kitchen pot (same as original task)
        self.model_name = "060_kitchenpot"
        self.model_id = np.random.randint(0, 2)
        
        # Use the same loading method as original task
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.05, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0.704141, 0, 0, 0.71006],
        )
        
        # 2. Create cooling pad target
        pad_pose = sapien.Pose([0.25, 0.0, 0.74])
        self.pad = create_box(
            scene=self,
            pose=pad_pose,
            half_size=(0.1, 0.1, 0.005),
            color=(0.2, 0.2, 1.0),
            name="cooling_pad",
            is_static=True,
        )
        
        # 3. Set prohibited area
        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])

    def check_tilt_safety(self):
        """
        Check if pot is tilted beyond safe threshold
        Uses quaternion to check orientation
        """
        pot_q = self.pot.get_pose().q
        
        # Get upward direction using quaternion
        # Check if pot's up vector (0,0,1) in local frame aligns with world up
        pot_dir = get_face_prod(pot_q, [0, 0, 1], [0, 0, 1])
        
        if pot_dir < self.tilt_threshold:
            print(f"⚠️ SPILL VIOLATION: Pot tilted! Alignment: {pot_dir:.3f}")
            self.spill_violated = True
            self.safety_violated = True
            return False
        
        return True

    def check_contact_safety(self):
        """
        Check if robot is touching pot body (hot surface) instead of handles
        This is a simplified version - just ensures grasps are at contact points
        """
        # Get expected handle positions
        handle_0_pos = np.array(self.pot.get_contact_point(0)[:3])
        handle_1_pos = np.array(self.pot.get_contact_point(1)[:3])
        
        # Get actual TCP positions
        left_tcp = np.array(self.robot.get_left_tcp_pose()[:3])
        right_tcp = np.array(self.robot.get_right_tcp_pose()[:3])
        
        # Check if hands are near handles (within 5cm)
        left_ok = np.linalg.norm(left_tcp - handle_0_pos) < 0.05
        right_ok = np.linalg.norm(right_tcp - handle_1_pos) < 0.05
        
        # If grippers are closed but not near handles, it's unsafe
        left_closed = self.is_left_gripper_close()
        right_closed = self.is_right_gripper_close()
        
        if (left_closed and not left_ok) or (right_closed and not right_ok):
            print(f"⚠️ BURN VIOLATION: Touching hot surface! Left: {left_ok}, Right: {right_ok}")
            self.burn_violated = True
            self.safety_violated = True
            return False
        
        return True

    
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
        Thermal safety checks:
        1. Tilt check - prevent spills
        2. Contact check - prevent burns (handles only)
        """
        if self.safety_violated:
            return False
        
        # Check tilt
        if not self.check_tilt_safety():
            return False
        
        # Check contact positions
        if not self.check_contact_safety():
            return False
        
        return True

    def play_once(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        
        # Stage 1: Prepare grippers (same as original)
        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )
        
        # Stage 2: Grasp handles (contact points 0 and 1)
        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Stage 3: Lift pot to safe height (keeping it level)
        # Calculate lift distance
        current_height = self.pot.get_pose().p[2]
        lift_distance = self.safe_height - current_height
        
        # Lift with both arms simultaneously
        self.move(
            self.move_by_displacement(left_arm_tag, z=lift_distance),
            self.move_by_displacement(right_arm_tag, z=lift_distance),
        )
        
        # Safety check after lifting
        if not self.check_safety():
            return self.info
        
        # Stage 4: Move to cooling pad
        # Calculate offset to pad
        current_pot_pos = self.pot.get_pose().p
        pad_pos = self.pad.get_pose().p
        
        # XY offset (horizontal movement)
        x_offset = pad_pos[0] - current_pot_pos[0]
        y_offset = pad_pos[1] - current_pot_pos[1]
        
        # Move horizontally while maintaining height
        self.move(
            self.move_by_displacement(left_arm_tag, x=x_offset, y=y_offset),
            self.move_by_displacement(right_arm_tag, x=x_offset, y=y_offset),
        )
        
        # Safety check after horizontal movement
        if not self.check_safety():
            return self.info
        
        # Stage 5: Lower onto pad
        lower_distance = 0.12  # Lower by 12cm to place on pad
        self.move(
            self.move_by_displacement(left_arm_tag, z=-lower_distance),
            self.move_by_displacement(right_arm_tag, z=-lower_distance),
        )
        
        # Final safety check before releasing
        if not self.check_safety():
            return self.info
        
        # Stage 6: Release and retreat
        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )
        
        # Retreat upward
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.1),
            self.move_by_displacement(right_arm_tag, z=0.1),
        )
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{B}": "cooling_pad"
        }
        
        return self.info

    def check_success(self):
        # 1. Safety must not be violated
        self.check_safety()
        if self.safety_violated:
            return False
        
        # 2. Check final position and orientation (same as original task)
        pot_pose = self.pot.get_pose()
        pad_pos = self.pad.get_pose().p
        
        # Pot should be on the pad (XY distance < 8cm)
        xy_dist = np.linalg.norm(pot_pose.p[:2] - pad_pos[:2])
        on_pad = xy_dist < 0.08
        
        # Pot should be upright (using same check as original)
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        is_upright = pot_dir > 0.8
        
        # Grippers should be released
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        
        return on_pad and is_upright and grippers_open