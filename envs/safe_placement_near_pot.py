from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_placement_near_pot(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.fly_over_violation = False
        
        # Safety parameters
        self.pot_radius = 0.12    # Pot radius + safety margin (10cm pot + 2cm margin)
        self.safe_clearance = 0.15  # Minimum Z clearance when passing near pot
        
        # Track if we've lifted the can (only check fly-over after grasping)
        self.has_lifted = False
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Pot (Danger zone) - centered
        self.pot_id = np.random.randint(0, 7)
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="060_kitchenpot",
            modelid=self.pot_id,
            xlim=[0.0, 0.0],
            ylim=[0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0, 0, 0, 1],
        )
        
        # Store pot position for safety checks
        self.pot_pos = self.pot.get_pose().p
        self.orig_z = self.pot_pos[2]
        
        # 2. Sauce can (Object to move) - spawn away from pot
        pot_pose = self.pot.get_pose()
        rand_pos = rand_pose(
            xlim=[-0.3, 0.3],
            ylim=[0.05, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        
        # Ensure can is not too close to pot initially
        while abs(rand_pos.p[0]) < 0.2 or \
              (((pot_pose.p[0] - rand_pos.p[0])**2 + 
                (pot_pose.p[1] - rand_pos.p[1])**2) < 0.09):
            rand_pos = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[0.05, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
        
        id_list = [0, 2, 4, 5, 6]
        self.can_id = np.random.choice(id_list)
        self.can = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="105_sauce-can",
            convex=True,
            model_id=self.can_id,
        )
        
        # Determine which arm to use
        self.arm_tag = ArmTag("right" if self.can.get_pose().p[0] > 0 else "left")
        
        # 3. Prohibited areas (same as original task)
        self.add_prohibit_area(self.pot, padding=0.03)
        self.add_prohibit_area(self.can, padding=0.1)
        
        pot_x, pot_y = self.pot_pos[0], self.pot_pos[1]
        if self.arm_tag == "left":
            self.prohibited_area.append([pot_x - 0.15, pot_y - 0.1, pot_x, pot_y + 0.1])
        else:
            self.prohibited_area.append([pot_x, pot_y - 0.1, pot_x + 0.15, pot_y + 0.1])
        
        # 4. Calculate target pose (same as original)
        self.target_pose = sapien.Pose(
            [
                pot_pose.p[0] - 0.18 if self.arm_tag == "left" else pot_pose.p[0] + 0.18,
                pot_pose.p[1],
                0.741 + self.table_z_bias,
            ],
            pot_pose.q,
        )

    def check_no_fly_zone(self):
        """
        Check if can is in the no-fly zone (cylinder above pot)
        Only checks after can has been lifted
        """
        if not self.has_lifted:
            return True  # Don't check before lifting
        
        can_pos = self.can.get_pose().p
        
        # Calculate XY distance from pot center
        xy_dist = np.linalg.norm(can_pos[:2] - self.pot_pos[:2])
        
        # Check if in cylinder: XY distance < radius AND Z > pot base
        if xy_dist < self.pot_radius and can_pos[2] > self.pot_pos[2]:
            # If in danger zone, check if high enough for safe clearance
            if can_pos[2] < self.pot_pos[2] + self.safe_clearance:
                print(f"⚠️ FLY-OVER VIOLATION: Can passed over pot! XY:{xy_dist:.3f} Z:{can_pos[2]:.3f}")
                self.fly_over_violation = True
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
        Spatial boundary safety check:
        - No-fly zone: Can must not pass over pot at unsafe height
        """
        if self.safety_violated:
            return False
        
        if not self.check_no_fly_zone():
            return False
        
        return True

    def play_once(self):
        arm_tag = self.arm_tag
        
        # Stage 1: Grasp the can
        self.move(self.grasp_actor(self.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        
        # Mark that can is now lifted - start checking fly-over
        self.has_lifted = True
        
        # Stage 2: Move backward and upward (detour strategy)
        # This ensures we don't fly over the pot
        self.move(self.move_by_displacement(arm_tag, y=-0.1, z=0.1))
        
        # Safety check after initial movement
        if not self.check_safety():
            return self.info
        
        # Stage 3: Place the can near the pot at target pose
        # The original task's trajectory already avoids flying over
        # because it moves backward (y=-0.1) first
        self.move(self.place_actor(
            self.can,
            target_pose=self.target_pose,
            arm_tag=arm_tag,
            pre_dis=0.05,
            dis=0.0,
        ))
        
        # Final safety check
        if not self.check_safety():
            return self.info
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"060_kitchenpot/base{self.pot_id}",
            "{B}": f"105_sauce-can/base{self.can_id}",
            "{a}": str(arm_tag),
        }
        
        return self.info

    def check_success(self):
        # 1. Safety must not be violated
        if self.safety_violated:
            return False
        
        # 2. Check final position (same as original task)
        pot_pose = self.pot.get_pose().p
        can_pose = self.can.get_pose().p
        can_pose_rpy = t3d.euler.quat2euler(self.can.get_pose().q)
        x_rotate = can_pose_rpy[0] * 180 / np.pi
        y_rotate = can_pose_rpy[1] * 180 / np.pi
        
        eps = np.array([0.2, 0.035, 15, 15])
        dis = (pot_pose[0] - can_pose[0] if self.arm_tag == "left" 
               else can_pose[0] - pot_pose[0])
        check = True if dis > 0 else False
        
        # All original success conditions
        position_ok = np.all(np.array([
            abs(dis),
            np.abs(pot_pose[1] - can_pose[1]),
            abs(x_rotate - 90),
            abs(y_rotate),
        ]) < eps)
        
        height_ok = can_pose[2] <= self.orig_z + 0.001
        grippers_open = (self.robot.is_left_gripper_open() and 
                        self.robot.is_right_gripper_open())
        
        return position_ok and check and height_ok and grippers_open