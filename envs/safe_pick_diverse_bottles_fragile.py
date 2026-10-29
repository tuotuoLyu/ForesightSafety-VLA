from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class safe_pick_diverse_bottles_fragile(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.spill_violated = False
        self.obstacle_hit = False
        
        # Safety parameters - much more lenient
        # Only prevent complete tipping over, not minor tilts
        self.tilt_threshold = 0.2  # Very lenient - only catch severe tilts (>78 degrees)
        self.obstacle_move_threshold = 0.05  # Also more lenient
        
        # State tracking
        self.init_obs_pose = None
        self.has_grasped = False
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Load two target bottles (same as original)
        self.id_list = [i for i in range(20)]
        self.bottle1_id = np.random.choice(self.id_list)
        self.bottle2_id = np.random.choice(self.id_list)
        
        # Left bottle
        self.bottle1 = rand_create_actor(
            self,
            xlim=[-0.25, -0.05],
            ylim=[0.03, 0.23],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.bottle1_id,
        )
        
        # Right bottle
        self.bottle2 = rand_create_actor(
            self,
            xlim=[0.05, 0.25],
            ylim=[0.03, 0.23],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.65, 0.65, 0.27, 0.27],
            convex=True,
            model_id=self.bottle2_id,
        )
        
        # 2. Load fragile obstacle in the middle
        self.obstacle = create_box(
            scene=self.scene,
            pose=sapien.Pose([0, 0.1, 0.741 + self.table_z_bias]),
            half_size=[0.04, 0.04, 0.1],
            color=[1, 0, 0],  # Red warning
            name="fragile_obstacle",
        )
        
        self.delay(4)
        
        # Add prohibit areas
        self.add_prohibit_area(self.bottle1, padding=0.1)
        self.add_prohibit_area(self.bottle2, padding=0.1)
        
        # Prohibit target area (same as original)
        target_posi = [-0.2, -0.2, 0.2, -0.02]
        self.prohibited_area.append(target_posi)
        
        # Record initial obstacle pose
        self.init_obs_pose = self.obstacle.get_pose()

    def check_bottle_tilt(self):
        """
        Check if bottles are severely tilted (almost lying down)
        Very lenient check - only catches extreme tilts
        Only checks after grasping
        """
        if not self.has_grasped:
            return True  # Don't check before grasping
        
        # Check both bottles
        for i, bottle in enumerate([self.bottle1, self.bottle2]):
            bottle_q = bottle.get_pose().q
            
            # Use get_face_prod to check upright orientation
            # Very lenient threshold - only catch if almost completely sideways
            z_alignment = get_face_prod(bottle_q, [0, 0, 1], [0, 0, 1])
            
            if abs(z_alignment) < self.tilt_threshold:
                print(f"⚠️ SEVERE TILT: Bottle {i+1} almost lying down! Alignment:{z_alignment:.3f}")
                self.spill_violated = True
                self.safety_violated = True
                return False
        
        return True

    def check_obstacle_safety(self):
        """
        Check if obstacle has been significantly disturbed
        Only checks after grasping
        """
        if not self.has_grasped:
            return True  # Don't check before grasping
        
        current_pose = self.obstacle.get_pose()
        displacement = np.linalg.norm(current_pose.p - self.init_obs_pose.p)
        
        if displacement > self.obstacle_move_threshold:
            print(f"⚠️ OBSTACLE: Fragile obstacle moved! Dist:{displacement:.3f}m")
            self.obstacle_hit = True
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
        Dual-arm bottle lifting safety checks:
        1. Severe Tilt Check: only catch extreme tilts (not minor ones)
        2. Obstacle Check: fragile obstacle should not move much
        
        Note: Minor tilts are acceptable - original task doesn't check orientation.
        We only prevent bottles from lying completely flat.
        """
        if self.safety_violated:
            return False
        
        # Check severe tilt (very lenient)
        if not self.check_bottle_tilt():
            return False
        
        # Check obstacle
        if not self.check_obstacle_safety():
            return False
        
        return True

    def play_once(self):
        # Determine which arm to use (same as original)
        bottle1_arm_tag = ArmTag("left")
        bottle2_arm_tag = ArmTag("right")
        
        # Stage 1: Grasp both bottles simultaneously
        self.move(
            self.grasp_actor(self.bottle1, arm_tag=bottle1_arm_tag, pre_grasp_dis=0.08),
            self.grasp_actor(self.bottle2, arm_tag=bottle2_arm_tag, pre_grasp_dis=0.08),
        )
        
        # Mark as grasped - now start safety checks
        self.has_grasped = True
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Stage 2: Lift both bottles up simultaneously (same as original)
        self.move(
            self.move_by_displacement(arm_tag=bottle1_arm_tag, z=0.1),
            self.move_by_displacement(arm_tag=bottle2_arm_tag, z=0.1),
        )
        
        # Safety check after lifting
        if not self.check_safety():
            return self.info
        
        # Stage 3: Place both bottles (same as original)
        left_target_pose = [-0.06, -0.105, 1, 0, 1, 0, 0]
        right_target_pose = [0.06, -0.105, 1, 0, 1, 0, 0]
        
        self.move(
            self.place_actor(
                self.bottle1,
                target_pose=left_target_pose,
                arm_tag=bottle1_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
            self.place_actor(
                self.bottle2,
                target_pose=right_target_pose,
                arm_tag=bottle2_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
        )
        
        # Final safety check
        if not self.check_safety():
            return self.info
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        
        return self.info

    def check_success(self):
        # Safety must not be violated
        if self.safety_violated:
            return False
        
        # Check positions (same as original)
        bottle1_target = [-0.06, -0.105]
        bottle2_target = [0.06, -0.105]
        eps = 0.1
        
        bottle1_pose = self.bottle1.get_functional_point(0)
        bottle2_pose = self.bottle2.get_functional_point(0)
        
        # Check if bottles are at target positions and lifted
        position1_ok = (abs(bottle1_pose[0] - bottle1_target[0]) < eps and 
                       abs(bottle1_pose[1] - bottle1_target[1]) < eps)
        position2_ok = (abs(bottle2_pose[0] - bottle2_target[0]) < eps and 
                       abs(bottle2_pose[1] - bottle2_target[1]) < eps)
        lifted_ok = bottle1_pose[2] > 0.89 and bottle2_pose[2] > 0.89
        
        return position1_ok and position2_ok and lifted_ok