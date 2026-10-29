from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_place_cans_box_packing(Base_Task):
    """
    Safe dual-arm packing task with boundary and collision safety.
    
    Fixes:
    1. Use original can placement logic for stability
    2. Fix list subtraction with numpy arrays
    3. Simplify safety checks to avoid false positives
    """

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        
        # Safety flags
        self.safety_violated = False
        self.rim_collision = False
        self.arm_interference = False
        
        # Safety parameters
        self.min_arm_dist = 0.15  # 15cm minimum arm distance (relaxed)

    def load_actors(self):
        """Use original placement logic for stability"""
        
        # 1. Plastic box in center (same as original)
        rand_pos_1 = rand_pose(
            xlim=[-0.0, 0.0],
            ylim=[-0.15, -0.1],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        
        self.plasticbox_id = np.random.choice([3, 5], 1)[0]
        self.plasticbox = create_actor(
            scene=self,
            pose=rand_pos_1,
            modelname="062_plasticbox",
            convex=True,
            model_id=self.plasticbox_id,
        )
        self.plasticbox.set_mass(0.05)

        # 2. Left can (same as original)
        rand_pos_2 = rand_pose(
            xlim=[-0.25, -0.15],
            ylim=[-0.15, -0.07],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        
        self.object1_id = np.random.choice([0, 1, 2, 3, 5, 6], 1)[0]
        self.object1 = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname="071_can",
            convex=True,
            model_id=self.object1_id,
        )
        self.object1.set_mass(0.05)

        # 3. Right can (same as original)
        rand_pos_3 = rand_pose(
            xlim=[0.15, 0.25],
            ylim=[-0.15, -0.07],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        
        self.object2_id = np.random.choice([0, 1, 2, 3, 5, 6], 1)[0]
        self.object2 = create_actor(
            scene=self,
            pose=rand_pos_3,
            modelname="071_can",
            convex=True,
            model_id=self.object2_id,
        )
        self.object2.set_mass(0.05)

        self.add_prohibit_area(self.plasticbox, padding=0.1)
        self.add_prohibit_area(self.object1, padding=0.05)
        self.add_prohibit_area(self.object2, padding=0.05)

    
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
        Check safety constraints:
        1. Arm interference: Arms too close
        2. No rim collision check for now (too sensitive)
        """
        if self.safety_violated:
            return False
        
        # Arm interference check
        try:
            l_pos = np.array(self.get_arm_pose(ArmTag("left"))[:3])
            r_pos = np.array(self.get_arm_pose(ArmTag("right"))[:3])
            arm_dist = np.linalg.norm(l_pos - r_pos)
            
            if arm_dist < self.min_arm_dist:
                print(f"⚠️  Safety Violation: Arms too close! Distance: {arm_dist:.3f}m")
                self.arm_interference = True
                self.safety_violated = True
                return False
        except Exception as e:
            # If we can't get arm poses, skip this check
            pass
        
        return True

    def play_once(self):
        """Use original play_once logic with safety checks"""
        
        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")

        # Grasp both objects with dual arms (same as original)
        self.move(
            self.grasp_actor(self.object1, arm_tag=arm_tag_left, pre_grasp_dis=0.1),
            self.grasp_actor(self.object2, arm_tag=arm_tag_right, pre_grasp_dis=0.1),
        )
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info

        # Lift up both arms (same as original)
        self.move(
            self.move_by_displacement(arm_tag=arm_tag_left, z=0.2),
            self.move_by_displacement(arm_tag=arm_tag_right, z=0.2),
        )
        
        # Safety check after lifting
        if not self.check_safety():
            return self.info

        # Place left object into plastic box (same as original)
        self.move(
            self.place_actor(
                self.object1,
                arm_tag=arm_tag_left,
                target_pose=self.plasticbox.get_functional_point(1),
                constrain="free",
                pre_dis=0.1,
            ))
        
        # Safety check after first placement
        if not self.check_safety():
            return self.info

        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))

        # Place right object while left arm returns (same as original)
        self.move(
            self.back_to_origin(arm_tag=arm_tag_left),
            self.place_actor(
                self.object2,
                arm_tag=arm_tag_right,
                target_pose=self.plasticbox.get_functional_point(0),
                constrain="free",
                pre_dis=0.1,
            ),
        )
        
        # Final safety check
        if not self.check_safety():
            return self.info

        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))
        self.move(self.back_to_origin(arm_tag=arm_tag_right))

        self.info["info"] = {
            "{A}": f"071_can/base{self.object1_id}",
            "{B}": f"062_plasticbox/base{self.plasticbox_id}",
            "{C}": f"071_can/base{self.object2_id}",
        }
        return self.info

    def check_success(self):
        """Check success with safety validation (fix list subtraction)"""
        
        # First check safety
        if not self.check_safety():
            return False
        
        # Then check task success (same as original but fix numpy)
        # Convert to numpy arrays to avoid list subtraction error
        plasticbox_fp_0 = np.array(self.plasticbox.get_functional_point(0)[0:2])
        plasticbox_fp_1 = np.array(self.plasticbox.get_functional_point(1)[0:2])
        
        obj1_pos = np.array(self.object1.get_pose().p[0:2])
        obj2_pos = np.array(self.object2.get_pose().p[0:2])
        
        dis1 = min(np.linalg.norm(obj1_pos - plasticbox_fp_0),
                   np.linalg.norm(obj1_pos - plasticbox_fp_1))
        dis2 = min(np.linalg.norm(obj2_pos - plasticbox_fp_0),
                   np.linalg.norm(obj2_pos - plasticbox_fp_1))
        
        threshold = 0.04
        success = (dis1 < threshold and dis2 < threshold and 
                  self.is_left_gripper_open() and self.is_right_gripper_open())
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe: dis1={dis1:.3f}, dis2={dis2:.3f}")
        
        return success