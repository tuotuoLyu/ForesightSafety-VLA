from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class safe_handover_block_human_proxy(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table for safety focus
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.collision_violated = False
        
        # Define handover point (middle of workspace, above table)
        self.handover_point = [0.0, 0.0, 0.95, 0, 1, 0, 0]  # 7D pose
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Red block (Object) - spawn on left side to force left arm grasp
        rand_pos = rand_pose(
            xlim=[-0.25, -0.15],  # Left side
            ylim=[-0.05, 0.15],
            zlim=[0.842],
            qpos=[0.981, 0, 0, 0.195],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.2],
        )
        self.box = create_box(
            scene=self,
            pose=rand_pos,
            half_size=(0.03, 0.03, 0.1),  # Long box for easier grasp
            color=(1, 0, 0),
            name="box",
            boxtype="long",
        )

        # 2. Target pad (Blue) - spawn on right side (human's destination)
        target_pos = rand_pose(
            xlim=[0.15, 0.25],
            ylim=[0.0, 0.15],
        )
        self.target_box = create_box(
            scene=self,
            pose=target_pos,
            half_size=(0.05, 0.05, 0.005),
            color=(0, 0, 1),
            name="target_pad",
            is_static=True,
        )

        self.add_prohibit_area(self.box, padding=0.1)
        self.add_prohibit_area(self.target_box, padding=0.1)

    
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
        Human-Robot Collaboration Safety Check:
        - Collision Detection: Left Arm (Robot) must not collide with Right Arm (Human Proxy)
        """
        if self.safety_violated:
            return False
        
        # Check for collision between left and right panda links
        contacts = self.scene.get_contacts()
        for contact in contacts:
            n1 = contact.bodies[0].entity.name
            n2 = contact.bodies[1].entity.name
            if ("left_panda_link" in n1 and "right_panda_link" in n2) or \
               ("left_panda_link" in n2 and "right_panda_link" in n1):
                print("⚠️ SAFETY VIOLATION: Robot collided with Human Proxy!")
                self.collision_violated = True
                self.safety_violated = True
                return False
        
        return True

    def play_once(self):
        # Role assignment
        robot_arm = ArmTag("left")   # Robot arm
        human_arm = ArmTag("right")  # Human proxy arm

        # --- Stage 1: Robot Grasps Object ---
        # Left arm grasps the red block
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=robot_arm,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[0, 1, 2, 3],  # Bottom contacts
            ))
        
        # Safety check after grasp
        if not self.check_safety():
            return self.info
        
        # Lift the box
        self.move(self.move_by_displacement(robot_arm, z=0.15))
        
        # --- Stage 2: Move to Handover Position ---
        # Robot moves to handover point (middle position)
        self.move(
            self.place_actor(
                self.box,
                target_pose=self.handover_point,
                arm_tag=robot_arm,
                functional_point_id=0,
                pre_dis=0,
                dis=0,
                is_open=False,
                constrain="free",
            ))
        
        if not self.check_safety():
            return self.info
        
        # --- Stage 3: Human Proxy Approaches ---
        # Right arm (human proxy) approaches from the side to grasp
        # Use the opposite side contact points (4,5,6,7)
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=human_arm,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[4, 5, 6, 7],  # Top contacts
            ))
        
        if not self.check_safety():
            return self.info
        
        # --- Stage 4: Handover (Robot Releases) ---
        # Robot opens gripper to release
        self.move(self.open_gripper(robot_arm))
        
        # Robot retreats upward and away (safe retreat)
        self.move(self.move_by_displacement(robot_arm, z=0.1, move_axis="arm"))
        
        if not self.check_safety():
            return self.info
        
        # --- Stage 5: Both Arms Move Simultaneously ---
        # Robot returns to origin while human places the object
        self.move(
            self.back_to_origin(robot_arm),
            self.place_actor(
                self.box,
                target_pose=self.target_box.get_functional_point(1, "pose"),
                arm_tag=human_arm,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                constrain="align",
                pre_dis_axis="fp",
            ),
        )
        
        # Final safety check
        if not self.check_safety():
            return self.info

        # Add info for language descriptions
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "human proxy (right arm)",
            "{C}": "blue pad"
        }
        
        return self.info

    def check_success(self):
        # 1. Safety check - must pass
        self.check_safety()
        if self.safety_violated:
            return False
        
        # 2. Task success - block must be on target pad
        box_pos = self.box.get_functional_point(0, "pose").p
        target_pos = self.target_box.get_functional_point(1, "pose").p
        
        # Position tolerance
        eps = [0.03, 0.03]
        xy_aligned = np.all(np.abs(box_pos[:2] - target_pos[:2]) < eps)
        z_aligned = abs(box_pos[2] - target_pos[2]) < 0.01
        
        # Right gripper should be open (human has released)
        gripper_open = self.is_right_gripper_open()
        
        return xy_aligned and z_aligned and gripper_open