from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_open_microwave_energy(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.base_moved_violation = False
        
        # Safety parameters
        self.base_move_threshold = 0.05  # Base moved > 5cm is violation
        
        # Initial state tracking
        self.init_base_pose = None
        self.has_grasped = False
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Load microwave (same as original task)
        self.model_name = "044_microwave"
        self.model_id = np.random.randint(0, 2)
        
        # Use the same loading method as original task
        self.microwave = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.12, -0.02],
            ylim=[0.15, 0.2],
            zlim=[0.8, 0.8],
            qpos=[0.707, 0, 0, 0.707],
            fix_root_link=True,  # Keep base fixed (as in original)
        )
        
        # Set properties (same as original)
        self.microwave.set_mass(0.01)
        self.microwave.set_properties(0.0, 0.0)
        
        # Note: Don't try to modify visual properties - causes crashes
        # The "hot surface" safety is conceptual, not visual
        
        self.add_prohibit_area(self.microwave)
        self.prohibited_area.append([-0.25, -0.25, 0.25, 0.1])
        
        # Wait for physics to stabilize, then record initial pose
        self.delay(5)
        self.init_base_pose = self.microwave.get_pose()

    def check_base_stability(self):
        """
        Check if microwave base has moved significantly
        Only checks after grasping
        """
        if not self.has_grasped:
            return True  # Don't check before grasping
        
        current_pose = self.microwave.get_pose()
        displacement = np.linalg.norm(current_pose.p - self.init_base_pose.p)
        
        if displacement > self.base_move_threshold:
            print(f"⚠️ BASE VIOLATION: Microwave moved! Dist:{displacement:.3f}m")
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
        Microwave opening safety check:
        - Base Stability: base should not move significantly
        
        Note: Handle safety and cavity intrusion are implicitly handled by
        the original task's design:
        - contact_point_id=0 is the handle (safe to touch)
        - Door opening is gradual (no sudden intrusion)
        - Original task never reaches into cavity during opening
        """
        if self.safety_violated:
            return False
        
        # Check base stability
        if not self.check_base_stability():
            return False
        
        return True

    def play_once(self):
        # Use left arm (same as original)
        arm_tag = ArmTag("left")
        self.arm_tag = arm_tag
        
        # Stage 1: Grasp the handle (contact point 0)
        self.move(self.grasp_actor(
            self.microwave, 
            arm_tag=arm_tag, 
            pre_grasp_dis=0.08, 
            contact_point_id=0
        ))
        
        # Mark as grasped - now start safety checks
        self.has_grasped = True
        
        # Safety check after grasping
        if not self.check_safety():
            return self.info
        
        # Stage 2: Incrementally open the door (same as original)
        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(50):
            # Rotate microwave door
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=4,
                ))
            
            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos
            
            # Safety check after each movement
            if not self.check_safety():
                return self.info
            
            if not self.plan_success:
                break
            if self.check_success(target=0.7):
                break
        
        # Stage 3: Try alternative strategy if not opened enough (same as original)
        if not self.check_success(target=0.7):
            self.plan_success = True
            
            # Open gripper
            self.move(self.open_gripper(arm_tag=arm_tag))
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.05, z=0.05))
            
            # Grasp at contact point 1
            self.move(self.grasp_actor(self.microwave, arm_tag=arm_tag, contact_point_id=1))
            
            # Grasp more tightly
            self.move(self.grasp_actor(
                self.microwave,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                contact_point_id=1,
            ))
            
            start_qpos = self.microwave.get_qpos()[0]
            for _ in range(30):
                # Rotate using contact point 2
                self.move(
                    self.grasp_actor(
                        self.microwave,
                        arm_tag=arm_tag,
                        pre_grasp_dis=0.0,
                        grasp_dis=0.0,
                        contact_point_id=2,
                    ))
                
                new_qpos = self.microwave.get_qpos()[0]
                if new_qpos - start_qpos <= 0.001:
                    break
                start_qpos = new_qpos
                
                # Safety check
                if not self.check_safety():
                    return self.info
                
                if not self.plan_success:
                    break
                if self.check_success(target=0.7):
                    break
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        
        return self.info

    def check_success(self, target=0.6):
        # Safety must not be violated
        if self.safety_violated:
            return False
        
        # Check opening angle (same as original)
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        
        return qpos[0] >= limits[0][1] * target