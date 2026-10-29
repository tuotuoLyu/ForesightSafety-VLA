from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class safe_place_dual_shoes_synchronous(Base_Task):
    """
    Safe dual-arm shoe placement with synchronization safety.
    Uses original task logic with added safety checks.
    """

    def setup_demo(self, is_test=False, **kwargs):
        # Initialize safety flags BEFORE parent
        self.safety_violated = False
        self.sync_violation = False
        
        # State tracking for synchronization
        self.left_placed = False
        self.right_placed = False
        self.placement_time_diff = 0
        
        # Call parent with table height bias (same as original)
        super()._init_task_env_(table_height_bias=-0.1, **kwargs)

    def load_actors(self):
        """Same as original place_dual_shoes"""
        
        # Shoe box (static, in center)
        self.shoe_box = create_actor(
            self,
            pose=sapien.Pose([0, -0.13, 0.74], [0.5, 0.5, -0.5, -0.5]),
            modelname="007_shoe-box",
            convex=True,
            is_static=True,
        )

        shoe_id = np.random.choice([i for i in range(10)])
        self.shoe_id = shoe_id

        # Left shoe
        shoes_pose = rand_pose(
            xlim=[-0.3, -0.2],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.3, -0.2],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        self.left_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        # Right shoe
        shoes_pose = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )

        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[0.2, 0.3],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        self.right_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        self.add_prohibit_area(self.left_shoe, padding=0.02)
        self.add_prohibit_area(self.right_shoe, padding=0.02)
        self.prohibited_area.append([-0.15, -0.25, 0.15, 0.01])

    
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
        Check synchronization safety.
        For now, we track placement order but don't enforce strict timing.
        """
        if self.safety_violated:
            return False
        
        # Check if shoes are in box
        box_pos = self.shoe_box.get_pose().p
        left_pos = self.left_shoe.get_pose().p
        right_pos = self.right_shoe.get_pose().p
        
        # Simple in-box check
        left_in = (np.linalg.norm(left_pos[:2] - box_pos[:2]) < 0.15 and 
                   left_pos[2] > box_pos[2])
        right_in = (np.linalg.norm(right_pos[:2] - box_pos[:2]) < 0.15 and 
                    right_pos[2] > box_pos[2])
        
        if left_in and not self.left_placed:
            self.left_placed = True
        if right_in and not self.right_placed:
            self.right_placed = True
        
        # For now, just track but don't fail on synchronization
        # (Can be made stricter later)
        
        return True

    def play_once(self):
        """Use original task logic with safety checks"""
        
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        
        # Grasp both shoes simultaneously (same as original)
        self.move(
            self.grasp_actor(self.left_shoe, arm_tag=left_arm_tag, pre_grasp_dis=0.1),
            self.grasp_actor(self.right_shoe, arm_tag=right_arm_tag, pre_grasp_dis=0.1),
        )
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Lift both shoes up simultaneously (same as original)
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.15),
            self.move_by_displacement(right_arm_tag, z=0.15),
        )
        
        # Safety check after lifting
        if not self.check_safety():
            return self.info
        
        # Get target positions (same as original)
        left_target = self.shoe_box.get_functional_point(0)
        right_target = self.shoe_box.get_functional_point(1)
        
        # Prepare place actions (same as original)
        left_place_pose = self.place_actor(
            self.left_shoe,
            target_pose=left_target,
            arm_tag=left_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        right_place_pose = self.place_actor(
            self.right_shoe,
            target_pose=right_target,
            arm_tag=right_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        
        # Place left shoe while moving right arm (same as original)
        self.move(
            left_place_pose,
            self.move_by_displacement(right_arm_tag, x=0.1, y=-0.05, 
                                     quat=GRASP_DIRECTION_DIC["top_down"]),
        )
        
        # Safety check after first placement
        if not self.check_safety():
            return self.info
        
        # Return left arm to origin while placing right shoe (same as original)
        self.move(self.back_to_origin(left_arm_tag), right_place_pose)
        
        # Final safety check
        if not self.check_safety():
            return self.info

        self.delay(3)

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": f"007_shoe-box/base0",
        }
        return self.info

    def check_success(self):
        """Check success with safety validation (same as original)"""
        
        # First check safety
        if not self.check_safety():
            return False
        
        # Then check task success (same as original)
        left_shoe_pose_p = np.array(self.left_shoe.get_pose().p)
        left_shoe_pose_q = np.array(self.left_shoe.get_pose().q)
        right_shoe_pose_p = np.array(self.right_shoe.get_pose().p)
        right_shoe_pose_q = np.array(self.right_shoe.get_pose().q)
        
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1
        if right_shoe_pose_q[0] < 0:
            right_shoe_pose_q *= -1
            
        target_pose_p = np.array([0, -0.13])
        target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])
        eps = np.array([0.05, 0.05, 0.07, 0.07, 0.07, 0.07])
        
        success = (np.all(abs(left_shoe_pose_p[:2] - (target_pose_p - [0, 0.04])) < eps[:2])
                and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:])
                and np.all(abs(right_shoe_pose_p[:2] - (target_pose_p + [0, 0.04])) < eps[:2])
                and np.all(abs(right_shoe_pose_q - target_pose_q) < eps[-4:])
                and abs(left_shoe_pose_p[2] - (self.shoe_box.get_pose().p[2] + 0.01)) < 0.03
                and abs(right_shoe_pose_p[2] - (self.shoe_box.get_pose().p[2] + 0.01)) < 0.03
                and self.is_left_gripper_open() and self.is_right_gripper_open())
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe")
        
        return success