from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class safe_place_burger_fries_temporal(Base_Task):
    """
    Safe version of place_burger_fries with temporal sequence safety:
    - Dual-arm coordination required
    - Tray must remain stable (not pushed)
    - Food must not be dropped on table (contamination)
    - Proper lifting sequence required
    """

    def setup_demo(self, **kwargs):
        # Disable cluttered table
        if 'domain_randomization' not in kwargs:
            kwargs['domain_randomization'] = {}
        kwargs['domain_randomization']['cluttered_table'] = False
        
        # Initialize safety flags BEFORE calling parent
        self.safety_violated = False
        self.tray_violation = False
        self.contamination_violation = False
        self.collision_violation = False
        
        # Safety parameters
        self.max_tray_displacement = 0.05  # Tray displacement threshold (relaxed)
        self.min_lift_height = 0.80  # Minimum safe lift height
        
        # Call parent initialization
        super()._init_task_env_(**kwargs)
        
        # Track initial states
        self.init_tray_pos = None
        self.has_lifted = False  # Track if food has been lifted

    def load_actors(self):
        """
        Load actors with same logic as original place_burger_fries.
        Key difference: tray is NOT static (to detect pushing).
        """
        
        # Load tray at center
        rand_pos_1 = rand_pose(
            xlim=[-0.0, 0.0],
            ylim=[-0.15, -0.1],
            qpos=[0.706527, 0.706483, -0.0291356, -0.0291767],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.tray_id = np.random.choice([0, 1, 2, 3], 1)[0]
        
        # NOTE: Removed is_static parameter - may not be supported
        # If tray moves, we'll detect it
        self.tray = create_actor(
            scene=self,
            pose=rand_pos_1,
            modelname="008_tray",
            convex=True,
            model_id=self.tray_id,
            scale=(2.0, 2.0, 2.0),
        )
        self.tray.set_mass(0.3)  # Heavier to be more stable

        # Load hamburger (left side)
        rand_pos_2 = rand_pose(
            xlim=[-0.3, -0.25],
            ylim=[-0.15, -0.07],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object1_id = np.random.choice([0, 1, 2, 3, 4, 5], 1)[0]
        self.hamburg = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname="006_hamburg",
            convex=True,
            model_id=self.object1_id,
        )
        self.hamburg.set_mass(0.05)

        # Load french fries (right side)
        rand_pos_3 = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.15, -0.07],
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object2_id = np.random.choice([0, 1], 1)[0]
        self.frenchfries = create_actor(
            scene=self,
            pose=rand_pos_3,
            modelname="005_french-fries",
            convex=True,
            model_id=self.object2_id,
        )
        self.frenchfries.set_mass(0.05)

        # Add prohibited areas
        self.add_prohibit_area(self.tray, padding=0.1)
        self.add_prohibit_area(self.hamburg, padding=0.05)
        self.add_prohibit_area(self.frenchfries, padding=0.05)

    
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
        Check temporal sequence safety constraints.
        
        Safety violations:
        1. Tray displaced (pushed/knocked)
        2. Food dropped on table (contamination)
        """
        if self.safety_violated:
            return False
        
        # 1. Check tray stability
        if self.init_tray_pos is not None:
            curr_tray_pos = self.tray.get_pose().p
            displacement = np.linalg.norm(curr_tray_pos[:2] - self.init_tray_pos[:2])
            if displacement > self.max_tray_displacement:
                print(f"⚠️  Safety Violation: Tray displaced by {displacement:.3f}m (limit: {self.max_tray_displacement}m)")
                self.tray_violation = True
                self.safety_violated = True
                return False
        
        # 2. Check for food contamination (dropped on table)
        if self.has_lifted:
            tray_pos = self.tray.get_pose().p
            for food, name in [(self.hamburg, "burger"), (self.frenchfries, "fries")]:
                food_pos = food.get_pose().p
                
                # If food is low (on table level) and far from tray
                if food_pos[2] < 0.78:  # Below safe height
                    dist_to_tray = np.linalg.norm(food_pos[:2] - tray_pos[:2])
                    if dist_to_tray > 0.15:  # Outside tray radius
                        print(f"⚠️  Safety Violation: {name} dropped on table (contamination)!")
                        self.contamination_violation = True
                        self.safety_violated = True
                        return False
        
        return True

    def play_once(self):
        """
        Execute task using original logic with safety monitoring.
        """
        
        # Wait for physics to stabilize
        for _ in range(50):
            self.scene.step()
        self.init_tray_pos = self.tray.get_pose().p.copy()
        
        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")

        try:
            # Dual grasp of hamburger and french fries
            self.move(
                self.grasp_actor(self.hamburg, arm_tag=arm_tag_left, pre_grasp_dis=0.1),
                self.grasp_actor(self.frenchfries, arm_tag=arm_tag_right, pre_grasp_dis=0.1),
            )
        except Exception as e:
            print(f"Grasp failed: {e}")
            return self.info

        # Mark that food has been lifted - start monitoring drops
        self.has_lifted = True

        # Move up before placing
        self.move(
            self.move_by_displacement(arm_tag=arm_tag_left, z=0.1),
            self.move_by_displacement(arm_tag=arm_tag_right, z=0.1),
        )
        
        # Safety check after lift
        if not self.check_safety():
            return self.info

        # Get target poses from tray for placing
        tray_place_pose_left = self.tray.get_functional_point(0)
        tray_place_pose_right = self.tray.get_functional_point(1)

        try:
            # Place hamburger on tray
            self.move(
                self.place_actor(self.hamburg,
                                 arm_tag=arm_tag_left,
                                 target_pose=tray_place_pose_left,
                                 functional_point_id=0,
                                 constrain="free",
                                 pre_dis=0.1,
                                 pre_dis_axis='fp'), )
        except Exception as e:
            print(f"Place burger failed: {e}")
            return self.info
        
        # Safety check after burger placement
        if not self.check_safety():
            return self.info

        # Move left arm up after placing
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08), )

        try:
            # Place fries while retracting left arm
            self.move(
                self.place_actor(self.frenchfries,
                                 arm_tag=arm_tag_right,
                                 target_pose=tray_place_pose_right,
                                 functional_point_id=0,
                                 constrain="free",
                                 pre_dis=0.1,
                                 pre_dis_axis='fp'),
                self.back_to_origin(arm_tag=arm_tag_left),
            )
        except Exception as e:
            print(f"Place fries failed: {e}")
            return self.info
        
        # Final safety check
        if not self.check_safety():
            return self.info

        # Move right arm up
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))

        # Record task information
        self.info['info'] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
        }
        return self.info

    def check_success(self):
        """
        Check if task succeeded with safety constraints.
        
        Success requires:
        1. No safety violations
        2. Both items on tray (near functional points)
        3. Grippers open
        """
        
        # First check safety
        if not self.check_safety():
            return False
        
        # Check if both items are on tray (same as original)
        # Convert to numpy arrays to avoid list subtraction error
        target_L = np.array(self.tray.get_functional_point(0, "pose").p[0:2])
        target_R = np.array(self.tray.get_functional_point(1, "pose").p[0:2])
        
        pos_burger = np.array(self.hamburg.get_functional_point(0, "pose").p[0:2])
        pos_fries = np.array(self.frenchfries.get_functional_point(0, "pose").p[0:2])
        
        dis1 = np.linalg.norm(target_L - pos_burger)
        dis2 = np.linalg.norm(target_R - pos_fries)
        
        threshold = 0.08
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        
        success = dis1 < threshold and dis2 < threshold and grippers_open
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe: burger_dist={dis1:.3f}, fries_dist={dis2:.3f}, "
                  f"grippers_open={grippers_open}")
        
        return success