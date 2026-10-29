from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_place_can_basket_transport(Base_Task):
    """
    Smart safety checks:
    - Tilt: Only check when basket is BOTH lifted AND grasped (transport phase)
    - Spill: Only check after can has been loaded
    """

    def setup_demo(self, is_test=False, **kwargs):
        super()._init_task_env_(**kwargs)
        
        # Safety flags
        self.safety_violated = False
        self.spill_violation = False
        self.tilt_violation = False
        
        # State tracking
        self.can_was_loaded = False
        self.basket_transport_started = False
        
        # Safety parameters
        self.max_tilt_cos = 0.50  # ~60 degrees
        self.can_inside_radius = 0.15

    def load_actors(self):
        """Same as original"""
        self.arm_tag = ArmTag({0: "left", 1: "right"}[np.random.randint(0, 2)])
        self.basket_name = "110_basket"
        self.basket_id = [0, 1][np.random.randint(0, 2)]
        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = can_dict[self.can_name][np.random.randint(0, len(can_dict[self.can_name]))]

        if self.arm_tag == "left":
            self.basket = rand_create_actor(
                scene=self, modelname=self.basket_name, model_id=self.basket_id,
                xlim=[0.02, 0.02], ylim=[-0.08, -0.05],
                qpos=[0.5, 0.5, 0.5, 0.5], convex=True,
            )
            self.can = rand_create_actor(
                scene=self, modelname=self.can_name, model_id=self.can_id,
                xlim=[-0.25, -0.2], ylim=[0.0, 0.1],
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061], convex=True,
            )
        else:
            self.basket = rand_create_actor(
                scene=self, modelname=self.basket_name, model_id=self.basket_id,
                xlim=[-0.02, -0.02], ylim=[-0.08, -0.05],
                qpos=[0.5, 0.5, 0.5, 0.5], convex=True,
            )
            self.can = rand_create_actor(
                scene=self, modelname=self.can_name, model_id=self.can_id,
                xlim=[0.2, 0.25], ylim=[0.0, 0.1],
                qpos=[0.707225, 0.706849, -0.0100455, -0.00982061], convex=True,
            )
        
        self.start_height = self.basket.get_pose().p[2]
        self.object_start_height = self.can.get_pose().p[2]
        self.basket.set_mass(0.5)
        self.can.set_mass(0.01)
        self.add_prohibit_area(self.can, padding=0.1)
        self.add_prohibit_area(self.basket, padding=0.05)

    
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
        """Smart safety checking based on task phase"""
        if self.safety_violated:
            return False
        
        can_p = self.can.get_pose().p
        basket_p = self.basket.get_pose().p
        
        # Check if can is in basket
        horizontal_dist = np.linalg.norm(can_p[:2] - basket_p[:2])
        can_in_basket = horizontal_dist < 0.12 and can_p[2] >= basket_p[2] - 0.02
        
        if can_in_basket and not self.can_was_loaded:
            self.can_was_loaded = True
        
        # Check if basket is lifted
        basket_lifted = basket_p[2] - self.start_height > 0.05  # 5cm threshold
        
        if basket_lifted and self.can_was_loaded:
            self.basket_transport_started = True
        
        # Only check violations during transport phase
        if self.basket_transport_started:
            # 1. Spill check
            if not can_in_basket:
                print(f"⚠️  Safety Violation: Can spilled out during transport! Distance: {horizontal_dist:.3f}m")
                self.spill_violation = True
                self.safety_violated = True
                return False
            
            # 2. Tilt check - only during stable transport (not during initial grasp)
            # Additional check: basket must be significantly lifted (> 8cm)
            if basket_p[2] - self.start_height > 0.08:
                basket_transform = self.basket.get_pose().to_transformation_matrix()
                basket_z_axis = basket_transform[:3, 2]
                cos_angle = basket_z_axis[2]
                
                if cos_angle < self.max_tilt_cos:
                    print(f"⚠️  Safety Violation: Basket tilted too much during transport! (cos={cos_angle:.2f})")
                    self.tilt_violation = True
                    self.safety_violated = True
                    return False
        
        return True

    def play_once(self):
        """Same as original with smart safety checks"""
        # Grasp the can
        self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05))

        # Determine placement pose
        place_pose = self.get_arm_pose(arm_tag=self.arm_tag)
        f0 = np.array(self.basket.get_functional_point(0))
        f1 = np.array(self.basket.get_functional_point(1))
        
        if np.linalg.norm(f0[:2] - place_pose[:2]) < np.linalg.norm(f1[:2] - place_pose[:2]):
            place_pose = f0
            place_pose[:2] = f0[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99))
        else:
            place_pose = f1
            place_pose[:2] = f1[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag == "left" else (0.05, 0, 0, 0.99))

        # Place the can
        self.move(
            self.place_actor(self.can, arm_tag=self.arm_tag, target_pose=place_pose,
                           dis=0.02, is_open=False, constrain="free"))

        # Safety check after placement
        if not self.check_safety():
            return self.info

        # Handle planning failure
        if self.plan_success is False:
            self.plan_success = True
            place_pose[0] += -0.15 if self.arm_tag == "left" else 0.15
            place_pose[2] += 0.15
            self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=place_pose))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=-0.1))
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(
                self.back_to_origin(arm_tag=self.arm_tag),
                self.grasp_actor(self.basket, arm_tag=self.arm_tag.opposite, pre_grasp_dis=0.02),
            )
        else:
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
            self.move(
                self.back_to_origin(arm_tag=self.arm_tag),
                self.grasp_actor(self.basket, arm_tag=self.arm_tag.opposite, pre_grasp_dis=0.08),
            )

        # Safety check before grasping basket
        if not self.check_safety():
            return self.info

        # Grasp and lift basket
        self.move(self.close_gripper(arm_tag=self.arm_tag.opposite))
        
        # Lift basket
        self.move(
            self.move_by_displacement(arm_tag=self.arm_tag.opposite,
                                      x=-0.02 if self.arm_tag.opposite == "left" else 0.02,
                                      z=0.05))
        
        # Safety check during transport
        if not self.check_safety():
            return self.info

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        """Check success with safety validation"""
        # First check safety
        if not self.check_safety():
            return False
        
        # Then check task success
        can_p = self.can.get_pose().p
        basket_p = self.basket.get_pose().p
        basket_axis = (self.basket.get_pose().to_transformation_matrix()[:3, :3] @ 
                      np.array([[0, 1, 0]]).T)
        can_contact_table = not self.check_actors_contact("071_can", "table")
        can_contact_basket = self.check_actors_contact("071_can", "110_basket")
        
        success = (basket_p[2] - self.start_height > 0.02 and \
                can_p[2] - self.object_start_height > 0.02 and \
                np.dot(basket_axis.reshape(3), [0, 0, 1]) > 0.5 and \
                np.sum(np.sqrt(np.power(can_p - basket_p, 2))) < 0.15 and \
                can_contact_table and can_contact_basket)
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe")
        
        return success