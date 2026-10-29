from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class safe_handover_mic_collaborative(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.dropped = False
        self.zone_violation = False
        
        # Safe handover zone (center position, elevated)
        # Define as full 7D pose for consistency
        self.handover_zone_pose = [0.0, -0.05, 0.98, 0, 1, 0, 0]
        self.handover_zone_radius = 0.12  # 12cm radius sphere
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Microphone - random position
        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.05, 0.0],
            qpos=[0.707, 0.707, 0, 0],  # Lying flat
            rotate_rand=False,
        )
        # Ensure not in the center (force robot to move)
        while abs(rand_pos.p[0]) < 0.15:
            rand_pos = rand_pose(
                xlim=[-0.2, 0.2],
                ylim=[-0.05, 0.0],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
            
        self.microphone_id = np.random.choice([0, 4, 5], 1)[0]
        self.microphone = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="018_microphone",
            convex=True,
            model_id=self.microphone_id,
        )

        self.add_prohibit_area(self.microphone, padding=0.07)
        
        # Store initial position for determining grasp arm
        self.initial_mic_x = self.microphone.get_pose().p[0]
        self.grasp_arm_tag = ArmTag("right" if self.initial_mic_x > 0 else "left")
        self.handover_arm_tag = self.grasp_arm_tag.opposite

    
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
        Collaborative safety checks:
        1. Collision between arms
        2. Drop detection
        """
        if self.safety_violated:
            return False
        
        # Check for arm collision
        contacts = self.scene.get_contacts()
        for contact in contacts:
            n1 = contact.bodies[0].entity.name
            n2 = contact.bodies[1].entity.name
            if ("left_panda_link" in n1 and "right_panda_link" in n2) or \
               ("left_panda_link" in n2 and "right_panda_link" in n1):
                print("⚠️ SAFETY VIOLATION: Arm collision detected!")
                self.safety_violated = True
                return False
        
        return True

    def check_handover_zone(self):
        """Check if object is within the safe handover zone"""
        mic_pos = self.microphone.get_pose().p
        zone_center = np.array(self.handover_zone_pose[:3])
        dist = np.linalg.norm(mic_pos - zone_center)
        
        if dist > self.handover_zone_radius:
            print(f"⚠️ ZONE VIOLATION: Handover outside safe zone! Distance: {dist:.3f}m")
            self.zone_violation = True
            self.safety_violated = True
            return False
        return True

    def play_once(self):
        # Determine arms based on microphone position
        grasp_arm = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        handover_arm = grasp_arm.opposite

        # --- Stage 1: Grasp & Lift ---
        # Grasp the microphone handle with the first arm
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=grasp_arm,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],  # Handle points
                pre_grasp_dis=0.1,
            ))
        
        if not self.check_safety():
            return self.info
        
        # Lift and reorient for handover
        # Adjust pose to make it easier for the other arm to grasp
        self.move(
            self.move_by_displacement(
                grasp_arm,
                z=0.12,
                quat=(GRASP_DIRECTION_DIC["front_right"] 
                      if grasp_arm == "left" 
                      else GRASP_DIRECTION_DIC["front_left"]),
                move_axis="arm",
            ))

        # --- Stage 2: Move to Safe Handover Zone ---
        # Place microphone in the designated handover zone (but keep holding)
        self.move(
            self.place_actor(
                self.microphone,
                arm_tag=grasp_arm,
                target_pose=self.handover_zone_pose,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,  # DO NOT release yet
                constrain="free",
            ))
        
        # Verify we're in the safe zone
        if not self.check_handover_zone():
            return self.info
        
        if not self.check_safety():
            return self.info

        # --- Stage 3: Synchronized Dual Grasp ---
        # Second arm grasps the microphone head (different contact points)
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=handover_arm,
                contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],  # Head points
                pre_grasp_dis=0.1,
            ))
        
        if not self.check_safety():
            return self.info
        
        # --- Stage 4: Confirmation Phase ---
        # Brief pause to ensure stable dual-arm grasp
        # Both grippers should be closed at this point
        self.delay(3)
        
        # Verify still in safe zone during dual grasp
        if not self.check_handover_zone():
            return self.info

        # --- Stage 5: First Arm Release ---
        # Only after confirming dual grasp, release the first arm
        self.move(self.open_gripper(grasp_arm))
        
        # --- Stage 6: Safe Retreat & Transport ---
        # First arm retreats, second arm moves the microphone
        self.move(
            self.move_by_displacement(grasp_arm, z=0.07, move_axis="arm"),
            self.move_by_displacement(
                handover_arm, 
                x=0.05 if handover_arm == "right" else -0.05
            ),
        )
        
        # Final safety check
        if not self.check_safety():
            return self.info

        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"018_microphone/base{self.microphone_id}",
            "{a}": str(grasp_arm),
            "{b}": str(handover_arm),
        }
        
        return self.info

    def check_success(self):
        # 1. Safety checks must pass
        self.check_safety()
        if self.safety_violated:
            return False
        
        # 2. Microphone must not have dropped
        mic_pose = self.microphone.get_functional_point(0)
        if mic_pose[2] < 0.92:  # Below safe height
            return False
        
        # 3. Verify grip status
        # Get contact to ensure microphone is held
        contact = self.get_gripper_actor_contact_position("018_microphone")
        if len(contact) == 0:
            return False
        
        # 4. Verify correct arm is holding (handover arm closed, grasp arm open)
        close_gripper_func = (self.is_left_gripper_close 
                             if self.handover_arm_tag == "left" 
                             else self.is_right_gripper_close)
        open_gripper_func = (self.is_left_gripper_open 
                            if self.grasp_arm_tag == "left" 
                            else self.is_right_gripper_open)
        
        # 5. Verify microphone position relative to handover arm
        # If handover arm is left, mic should be on left side (x < 0)
        # If handover arm is right, mic should be on right side (x > 0)
        position_check = (mic_pose[0] < 0 if self.handover_arm_tag == "left" 
                         else mic_pose[0] > 0)
        
        return (close_gripper_func() and 
                open_gripper_func() and 
                mic_pose[2] > 0.92 and 
                position_check)