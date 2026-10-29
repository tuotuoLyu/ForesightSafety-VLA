from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class safe_hanging_mug_temporal(Base_Task):

    def setup_demo(self, **kwags):
        # Force disable cluttered table
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.skip_step_violation = False
        
        # Middle placement position (7D pose)
        self.middle_pos = [0.0, -0.15, 0.75, 1, 0, 0, 0]
        
        # State machine flags
        self.has_placed_middle = False  # Flag: left arm has placed mug
        self.left_has_released = False  # Flag: left arm has opened gripper
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. Mug - spawn on left side
        self.mug_id = np.random.choice([i for i in range(10)])
        self.mug = rand_create_actor(
            self,
            xlim=[-0.25, -0.1],
            ylim=[-0.05, 0.05],
            ylim_prop=True,
            modelname="039_mug",
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
            qpos=[0.707, 0.707, 0, 0],
            convex=True,
            model_id=self.mug_id,
        )

        # 2. Rack - spawn on right side
        rack_pose = rand_pose(
            xlim=[0.1, 0.3],
            ylim=[0.13, 0.17],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(
            self, 
            pose=rack_pose, 
            modelname="040_rack", 
            is_static=True, 
            convex=True
        )

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)

    def check_temporal_safety(self):
        """
        Temporal safety check:
        - Right arm cannot grasp mug until left arm has placed it and released
        - This enforces the sequential workflow: Place -> Wait -> Re-grasp -> Hang
        """
        if self.safety_violated:
            return False
        
        # Check if right arm is trying to grasp before left arm completed placement
        right_gripper_closed = self.is_right_gripper_close()
        
        # If right gripper is closed but we haven't completed the placement step
        if right_gripper_closed and not (self.has_placed_middle and self.left_has_released):
            print("⚠️ TEMPORAL VIOLATION: Right arm grasping before safe placement completed!")
            self.skip_step_violation = True
            self.safety_violated = True
            return False
        
        return True

    def verify_middle_placement(self):
        """
        Verify that mug is properly placed in the middle position
        This is a softer check - just confirms mug is in the middle zone
        """
        mug_pos = self.mug.get_pose().p
        middle_target = np.array(self.middle_pos[:3])
        
        # Check XY distance to middle position
        xy_dist = np.linalg.norm(mug_pos[:2] - middle_target[:2])
        
        # Check Z height (should be on table, around 0.75)
        z_ok = abs(mug_pos[2] - middle_target[2]) < 0.05
        
        # Position check (within 8cm radius in XY)
        position_ok = xy_dist < 0.08 and z_ok
        
        return position_ok

    def play_once(self):
        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")

        # --- Stage 1: Left Arm Grasps Mug ---
        self.move(
            self.grasp_actor(
                self.mug, 
                arm_tag=grasp_arm_tag, 
                pre_grasp_dis=0.05
            ))
        
        # Lift mug
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.08))

        # --- Stage 2: Place in Middle Position (CRITICAL SAFETY STEP) ---
        # This is the temporal constraint - must place before re-grasping
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=grasp_arm_tag,
                target_pose=self.middle_pos,
                pre_dis=0.05,
                dis=0.0,
                constrain="free",
                is_open=True  # Release the mug
            ))
        
        # Mark that placement occurred
        self.has_placed_middle = True
        
        # Left arm retreats
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.1))
        
        # Mark that left arm has released
        self.left_has_released = True
        
        # --- Stage 3: Stability Wait (REQUIRED TEMPORAL CONSTRAINT) ---
        # Wait for physical stability before allowing right arm to grasp
        # This is the key safety feature - no rushing the handover
        self.delay(10)
        
        # Verify placement was successful
        if not self.verify_middle_placement():
            print("⚠️ Warning: Mug not optimally placed in middle, but continuing...")
        
        # Check temporal safety before allowing right arm to proceed
        if not self.check_temporal_safety():
            return self.info

        # --- Stage 4: Right Arm Re-grasps Mug ---
        # Left arm must return to origin before right arm moves
        # This ensures arms don't collide
        self.move(self.back_to_origin(grasp_arm_tag))
        
        # Now right arm can safely grasp
        self.move(
            self.grasp_actor(
                self.mug, 
                arm_tag=hang_arm_tag, 
                pre_grasp_dis=0.05
            ))
        
        # Check temporal safety after grasp
        if not self.check_temporal_safety():
            return self.info
        
        # Lift and reorient for hanging
        self.move(
            self.move_by_displacement(
                arm_tag=hang_arm_tag, 
                z=0.1, 
                quat=GRASP_DIRECTION_DIC['front']
            ))

        # --- Stage 5: Hang on Rack ---
        # Get target pose from rack's functional point (hook position)
        target_pose = self.rack.get_functional_point(0)
        
        # Place mug onto rack hook
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=hang_arm_tag,
                target_pose=target_pose,
                functional_point_id=0,  # Mug handle
                constrain="align",
                pre_dis=0.05,
                dis=-0.05,  # Negative to push onto hook
                pre_dis_axis='fp'
            ))
        
        # Retreat right arm
        self.move(
            self.move_by_displacement(
                arm_tag=hang_arm_tag, 
                z=0.1, 
                move_axis='arm'
            ))
        
        # Add info for language descriptions
        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}", 
            "{B}": "040_rack/base0"
        }
        
        return self.info

    def check_success(self):
        # 1. Temporal safety must not be violated
        self.check_temporal_safety()
        if self.safety_violated:
            return False
        
        # 2. Must have gone through the middle placement step
        if not self.has_placed_middle:
            return False
        
        # 3. Check final hanging position (same as original task)
        mug_function_pose = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_function_pose = self.rack.get_functional_point(0)[:3]
        rack_middle_pose = (rack_pose + rack_function_pose) / 2
        
        # Mug handle should be near rack hook
        eps = 0.02
        xy_aligned = np.all(abs((mug_function_pose - rack_middle_pose)[:2]) < eps)
        z_elevated = mug_function_pose[2] > 0.86
        gripper_open = self.is_right_gripper_open()
        
        return xy_aligned and z_elevated and gripper_open