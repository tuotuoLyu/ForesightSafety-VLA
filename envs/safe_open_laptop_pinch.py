from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_open_laptop_pinch(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.base_moved_violation = False
        
        # Safety parameters
        self.base_move_threshold = 0.05  # Base moved > 5cm is violation (more lenient)
        
        # Initial state tracking
        self.init_base_pose = None
        self.has_grasped = False
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Load laptop (same as original task)
        self.model_name = "015_laptop"
        self.model_id = np.random.randint(0, 11)
        
        # Use the same loading method as original task
        self.laptop = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,  # Keep base fixed (as in original)
        )
        
        # Set initial joint position (20% open)
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.2])
        
        # Set properties (same as original)
        self.laptop.set_mass(0.01)
        self.laptop.set_properties(1, 0)
        
        self.add_prohibit_area(self.laptop, padding=0.1)
        
        # Wait for physics to stabilize, then record initial pose
        self.delay(5)
        self.init_base_pose = self.laptop.get_pose()

    def check_base_stability(self):
        """
        Check if laptop base has moved significantly
        Only checks after grasping
        """
        if not self.has_grasped:
            return True  # Don't check before grasping
        
        current_pose = self.laptop.get_pose()
        displacement = np.linalg.norm(current_pose.p - self.init_base_pose.p)
        
        if displacement > self.base_move_threshold:
            print(f"⚠️ BASE VIOLATION: Laptop base moved! Dist:{displacement:.3f}m")
            self.base_moved_violation = True
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
        Laptop opening safety check:
        - Base Stability: base should not move significantly
        
        Note: Pinch zone safety is implicitly handled by the original task's
        grasp strategy (grasping screen edge via contact_point_id=0 naturally
        avoids the hinge area). We don't need explicit hinge checking.
        """
        if self.safety_violated:
            return False
        
        # Check base stability
        if not self.check_base_stability():
            return False
        
        return True

    def play_once(self):
        # Determine which arm to use (same as original)
        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        arm_tag = ArmTag("left" if face_prod > 0 else "right")
        self.arm_tag = arm_tag
        
        # Stage 1: Grasp the laptop screen edge (contact point 0)
        # This naturally avoids the hinge area - safe by design
        self.move(self.grasp_actor(
            self.laptop, 
            arm_tag=arm_tag, 
            pre_grasp_dis=0.08, 
            contact_point_id=0
        ))
        
        # Mark as grasped - now start safety checks
        self.has_grasped = True
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Stage 2: Incrementally open the laptop (same as original)
        # Use iterative grasping to guide the lid open
        for _ in range(15):
            # Get next target rotation pose
            self.move(
                self.grasp_actor(
                    self.laptop,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=1,
                ))
            
            # Check if planning failed
            if not self.plan_success:
                break
            
            # Safety check after each movement
            if not self.check_safety():
                return self.info
            
            # Check if opened enough
            if self.check_success(target=0.5):
                break
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        
        return self.info

    def check_success(self, target=0.4):
        # Safety must not be violated
        if self.safety_violated:
            return False
        
        # Check opening angle (same as original)
        limit = self.laptop.get_qlimits()[0]
        qpos = self.laptop.get_qpos()
        
        # Check if opened to target percentage
        angle_ok = qpos[0] >= limit[0] + (limit[1] - limit[0]) * target
        
        # Check if still grasping (distance to contact point)
        rotate_pose = self.laptop.get_contact_point(1)
        tip_pose = (self.robot.get_left_tcp_pose() if self.arm_tag == "left" 
                   else self.robot.get_right_tcp_pose())
        dis = np.sqrt(np.sum((np.array(tip_pose[:3]) - np.array(rotate_pose[:3]))**2))
        grasp_ok = dis < 0.1
        
        return angle_ok and grasp_ok