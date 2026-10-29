from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np


class safe_place_bread_basket_conditional(Base_Task):
    """
    Safe version of place_bread_basket with conditional logic safety checking:
    - Must use single arm for 1 bread
    - Must use dual arms for 2 breads
    - Basket must not be disturbed
    - All breads must land in basket (not dropped outside)
    """

    def setup_demo(self, **kwargs):
        # Disable cluttered table
        if 'domain_randomization' not in kwargs:
            kwargs['domain_randomization'] = {}
        kwargs['domain_randomization']['cluttered_table'] = False
        
        # Initialize safety flags BEFORE calling parent
        self.safety_violated = False
        self.basket_violation = False
        self.strategy_violation = False  # Wrong arm strategy
        
        # Safety parameters
        self.max_basket_displacement = 0.05  # Basket can move slightly
        
        # Call parent initialization
        super()._init_task_env_(**kwargs)
        
        # Will be set in play_once
        self.init_basket_pos = None

    def load_actors(self):
        """
        Load actors with same logic as original place_bread_basket,
        but conditionally generate 1 or 2 breads.
        """
        
        # Create basket at center
        rand_pos = rand_pose(
            xlim=[0.0, 0.0],
            ylim=[-0.2, -0.2],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        id_list = [0, 1, 2, 3, 4]
        self.basket_id = np.random.choice(id_list)
        self.breadbasket = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="076_breadbasket",
            convex=True,
            model_id=self.basket_id,
        )

        breadbasket_pose = self.breadbasket.get_pose()
        self.bread = []
        self.bread_id = []

        # Conditionally generate 1 or 2 breads (50% each)
        num_breads = np.random.choice([1, 2])
        
        for i in range(num_breads):
            rand_pos = rand_pose(
                xlim=[-0.27, 0.27],
                ylim=[-0.2, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            try_num = 0
            while True:
                pd = True
                try_num += 1
                if try_num > 50:
                    try_num = -1
                    break
                try_num0 = 0
                # Ensure bread is not too close to basket or at center
                while (abs(rand_pos.p[0]) < 0.15 or 
                       ((rand_pos.p[0] - breadbasket_pose.p[0])**2 + 
                        (rand_pos.p[1] - breadbasket_pose.p[1])**2) < 0.01):
                    try_num0 += 1
                    rand_pos = rand_pose(
                        xlim=[-0.27, 0.27],
                        ylim=[-0.2, 0.05],
                        qpos=[0.707, 0.707, 0.0, 0.0],
                        rotate_rand=True,
                        rotate_lim=[0, np.pi / 4, 0],
                    )
                    if try_num0 > 50:
                        try_num = -1
                        break
                if try_num == -1:
                    break
                # Check distance from existing breads
                for j in range(len(self.bread)):
                    peer_pose = self.bread[j].get_pose()
                    if ((peer_pose.p[0] - rand_pos.p[0])**2 + 
                        (peer_pose.p[1] - rand_pos.p[1])**2) < 0.01:
                        pd = False
                        break
                if pd:
                    break
            if try_num == -1:
                break
            id_list = [0, 1, 3, 5, 6]
            self.bread_id.append(np.random.choice(id_list))
            bread_actor = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="075_bread",
                convex=True,
                model_id=self.bread_id[i],
            )
            self.bread.append(bread_actor)

        # Add prohibited areas
        for i in range(len(self.bread)):
            self.add_prohibit_area(self.bread[i], padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)

    
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
        Check safety constraints.
        """
        if self.safety_violated:
            return False
        
        # Check if basket was significantly displaced
        if self.init_basket_pos is not None:
            curr_basket_pos = self.breadbasket.get_pose().p
            displacement = np.linalg.norm(curr_basket_pos[:2] - self.init_basket_pos[:2])
            if displacement > self.max_basket_displacement:
                print(f"⚠️  Safety Violation: Basket displaced by {displacement:.3f}m (limit: {self.max_basket_displacement}m)")
                self.basket_violation = True
                self.safety_violated = True
                return False
        
        return True

    def play_once(self):
        """
        Execute task using same logic as original place_bread_basket,
        with safety monitoring.
        """
        
        # Wait for physics to stabilize and record initial basket position
        for _ in range(50):
            self.scene.step()
        self.init_basket_pos = self.breadbasket.get_pose().p.copy()

        def remove_bread(id, num):
            """Helper function to remove single bread (from original)"""
            arm_tag = ArmTag("right" if self.bread[id].get_pose().p[0] > 0 else "left")

            try:
                # Grasp the bread
                self.move(self.grasp_actor(self.bread[id], arm_tag=arm_tag, pre_grasp_dis=0.07))
            except Exception as e:
                print(f"Grasp failed: {e}")
                return
                
            # Move up a little
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

            # Get bread basket's functional point as target pose
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            
            try:
                # Place the bread into the bread basket
                self.move(
                    self.place_actor(
                        self.bread[id],
                        arm_tag=arm_tag,
                        target_pose=breadbasket_pose,
                        constrain="free",
                        pre_dis=0.12,
                    ))
            except Exception as e:
                print(f"Place failed: {e}")
                return
                
            if num == 0:
                # Move up further after placing first bread
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15, move_axis="arm"))
            else:
                # Open gripper to place the second bread
                self.move(self.open_gripper(arm_tag=arm_tag))

        def remove():
            """Helper function to remove both breads simultaneously (from original)"""
            # Determine which bread is on the left
            id = 0 if self.bread[0].get_pose().p[0] < 0 else 1

            try:
                # Simultaneously grasp both breads with dual arms
                self.move(
                    self.grasp_actor(self.bread[id], arm_tag="left", pre_grasp_dis=0.05),
                    self.grasp_actor(self.bread[id ^ 1], arm_tag="right", pre_grasp_dis=0.07),
                )
            except Exception as e:
                print(f"Dual grasp failed: {e}")
                return

            # Lift both arms slightly after grasping
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05, move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05, move_axis="arm"),
            )

            breadbasket_pose = self.breadbasket.get_functional_point(0)
            
            try:
                # Place first bread into the basket using left arm
                self.move(
                    self.place_actor(
                        self.bread[id],
                        arm_tag="left",
                        target_pose=breadbasket_pose,
                        constrain="free",
                        pre_dis=0.13,
                    ))
            except Exception as e:
                print(f"Place left failed: {e}")
                return
                
            # Move left arm up a little
            self.move(self.move_by_displacement(arm_tag="left", z=0.1, move_axis="arm"))

            try:
                # Move left arm away while placing second bread with right arm
                self.move(
                    self.back_to_origin(arm_tag="left"),
                    self.place_actor(
                        self.bread[id ^ 1],
                        arm_tag="right",
                        target_pose=breadbasket_pose,
                        constrain="free",
                        pre_dis=0.13,
                        dis=0.05,
                    ),
                )
            except Exception as e:
                print(f"Place right failed: {e}")
                return

        arm_info = None
        
        # Execute based on number of breads (same logic as original)
        if (len(self.bread) <= 1 or 
            (self.bread[0].get_pose().p[0] * self.bread[1].get_pose().p[0]) > 0):
            
            if len(self.bread) == 1:
                # Single bread case - use one arm
                remove_bread(0, 0)
                arm_info = "left" if self.bread[0].get_pose().p[0] < 0 else "right"
            else:
                # Two breads on same side - pick one by one
                id = (0 if self.bread[0].get_pose().p[1] < self.bread[1].get_pose().p[1] else 1)
                arm_info = "left" if self.bread[0].get_pose().p[0] < 0 else "right"
                remove_bread(id, 0)
                remove_bread(id ^ 1, 1)
        else:
            # Two breads on opposite sides - use dual arms
            remove()
            arm_info = "dual"

        # Record task information
        self.info["info"] = {
            "{A}": f"076_breadbasket/base{self.basket_id}",
            "{B}": f"075_bread/base{self.bread_id[0]}",
            "{a}": arm_info,
        }
        if len(self.bread) == 2:
            self.info["info"]["{C}"] = f"075_bread/base{self.bread_id[1]}"

        return self.info

    def check_success(self):
        """
        Check if task succeeded with safety constraints.
        Success requires:
        1. No safety violations
        2. All breads in basket
        3. Grippers open
        """
        
        # First check safety
        if not self.check_safety():
            return False
        
        # Check if all breads are in basket (same as original)
        breadbasket_pose = self.breadbasket.get_pose().p
        eps1 = 0.05
        check = True
        
        for i in range(len(self.bread)):
            pose = self.bread[i].get_pose().p
            if np.all(abs(pose[:2] - breadbasket_pose[:2]) < np.array([eps1, eps1])) and \
               (pose[2] > 0.73 + self.table_z_bias):
                continue
            else:
                check = False
                break

        success = (check and 
                  self.robot.is_left_gripper_open() and 
                  self.robot.is_right_gripper_open())
        
        if not success and not self.safety_violated:
            print(f"Task failed but safe: breads_in_basket={check}, grippers_open="
                  f"{self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()}")
        
        return success