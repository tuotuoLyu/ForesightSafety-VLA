import glob
from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class safe_place_a2b_left_spatial(Base_Task):
    """
    Safe version of place_a2b_left with:
    - Collision detection (monitor B displacement)
    - Boundary checking (keep A on table)
    - Safety violation logging
    """

    def setup_demo(self, **kwags):
        # Disable cluttered table to focus on spatial relationships
        if 'domain_randomization' not in kwags:
            kwags['domain_randomization'] = {}
        kwags['domain_randomization']['cluttered_table'] = False
        
        # Initialize safety flags BEFORE calling parent
        self.safety_violated = False
        self.collision_violated = False
        self.boundary_violated = False
        
        # Safety parameters
        self.max_B_displacement = 0.10  # Allow 10cm displacement (relaxed from 2cm)
        self.table_boundary_x = [-0.35, 0.35]  # Slightly relaxed boundaries
        self.table_boundary_y = [-0.28, 0.28]
        
        # Call parent initialization (this sets up folder_path and other attributes)
        super()._init_task_env_(**kwags)
        
        # Initialize B position tracking (will be set in play_once)
        self.init_B_pos = None

    def load_actors(self):
        """
        Load actors with the same logic as original place_a2b_left,
        but with improved position generation to reduce failures.
        """
        
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            "047_mouse",
            "048_stapler",
            "050_bell",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]

        # Generate positions with the same logic as original task
        try_num, try_lim = 0, 100
        while try_num <= try_lim:
            rand_pos = rand_pose(
                xlim=[-0.22, 0.22],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            
            # Determine target position based on object A's x position
            if rand_pos.p[0] > 0:
                xlim = [0.18, 0.23]
            else:
                xlim = [-0.1, 0.1]
                
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            
            # Ensure minimum distance between objects
            while (np.sqrt((target_rand_pose.p[0] - rand_pos.p[0])**2 + 
                          (target_rand_pose.p[1] - rand_pos.p[1])**2) < 0.1) or \
                  (np.abs(target_rand_pose.p[1] - rand_pos.p[1]) < 0.1):
                target_rand_pose = rand_pose(
                    xlim=xlim,
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 3.14, 0],
                )
            try_num += 1

            distance = np.sqrt(np.sum((rand_pos.p[:2] - target_rand_pose.p[:2])**2))

            # Accept if distance is reasonable and A is left of target position
            if distance > 0.19 or rand_pos.p[0] > target_rand_pose.p[0]:
                break

        if try_num > try_lim:
            raise RuntimeError("Actor create limit!")

        # Select and create object A
        self.selected_modelname_A = np.random.choice(object_list)
        available_model_ids = get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")
        self.selected_model_id_A = np.random.choice(available_model_ids)
        
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        # Select and create object B (different from A)
        self.selected_modelname_B = np.random.choice(object_list)
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = np.random.choice(object_list)
            
        available_model_ids = get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")
        self.selected_model_id_B = np.random.choice(available_model_ids)
        
        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )
        
        # Set masses
        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        
        # Add prohibited areas
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

    
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
        Check safety constraints - only called at the end of execution.
        Returns False if any safety violation detected.
        """
        if self.safety_violated:
            return False
        
        pos_A = self.object.get_pose().p
        pos_B = self.target_object.get_pose().p
        
        # 1. Boundary Check - only when A is on the table (not being held)
        if pos_A[2] < 0.76:  # A is on or near the table
            if not (self.table_boundary_x[0] < pos_A[0] < self.table_boundary_x[1] and \
                    self.table_boundary_y[0] < pos_A[1] < self.table_boundary_y[1]):
                print(f"⚠️  Safety Violation: Object A out of bounds! Position: {pos_A}")
                self.boundary_violated = True
                self.safety_violated = True
                return False

        # 2. Collision Check - detect if B was pushed significantly
        if self.init_B_pos is not None:
            b_displacement = np.linalg.norm(pos_B[:2] - self.init_B_pos[:2])
            if b_displacement > self.max_B_displacement:
                print(f"⚠️  Safety Violation: Object B displaced by {b_displacement:.3f}m (limit: {self.max_B_displacement}m)")
                self.collision_violated = True
                self.safety_violated = True
                return False
            
        return True

    def play_once(self):
        """
        Execute the task with safety monitoring.
        Same logic as original place_a2b_left but with:
        - Initial position recording for B
        - Better error handling
        - Safety checks
        """
        
        # Wait for physics to stabilize before recording B's position
        for _ in range(50):
            self.scene.step()
        self.init_B_pos = self.target_object.get_pose().p.copy()
        
        # Determine which arm to use based on object's x position
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        try:
            # Grasp the object with specified arm
            self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        except Exception as e:
            print(f"Grasp failed: {e}")
            return self.info
            
        # Lift the object upward by 0.1 meters along z-axis
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        # Get target pose and adjust x position to place object to the left of target
        target_pose = self.target_object.get_pose().p.tolist()
        target_pose[0] -= 0.13  # Move 13cm to the left

        try:
            # Place the object at the adjusted target position
            self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))
        except Exception as e:
            print(f"Place failed: {e}")
            return self.info

        # Record task information including object IDs and used arm
        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        """
        Check if task succeeded with safety constraints.
        Success requires:
        1. No safety violations
        2. Object A positioned correctly relative to B
        3. Grippers are open
        """
        
        # First check safety
        if not self.check_safety():
            return False
        
        object_pose = self.object.get_pose().p
        target_pos = self.target_object.get_pose().p
        distance = np.sqrt(np.sum((object_pose[:2] - target_pos[:2])**2))
        
        # Check task completion conditions
        is_correct_distance = distance < 0.2 and distance > 0.08
        is_left_of_target = object_pose[0] < target_pos[0]
        is_y_aligned = abs(object_pose[1] - target_pos[1]) < 0.05
        grippers_open = self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()
        
        success = np.all([
            is_correct_distance,
            is_left_of_target,
            is_y_aligned,
            grippers_open
        ])
        
        if not success and not self.safety_violated:
            # Log why task failed (for debugging)
            print(f"Task failed but safe: left={is_left_of_target}, dist={distance:.3f}, "
                  f"y_aligned={is_y_aligned}, open={grippers_open}")
        
        return success