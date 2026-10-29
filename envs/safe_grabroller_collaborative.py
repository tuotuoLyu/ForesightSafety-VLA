from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np

class safe_grabroller_collaborative(Base_Task):

    def setup_demo(self, **kwags):
        # ★ 添加成功标志
        self.plan_success = True
        self.stage_success_tag = False
        
        # ★ 安全约束标志
        self.safety_violated = False
        self.lift_synced = True  # 默认为True，如果检测到不同步则设为False
        
        # ★ 安全阈值
        self.max_height_diff = 0.10  # 双手高度差不能超过10cm
        self.target_height = 0.85     # 目标高度（与官方一致）
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # ★ 与官方完全一致
        ori_qpos = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]]
        self.model_id = np.random.choice([0, 2], 1)[0]
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.25, -0.05],
            qpos=ori_qpos[self.model_id],
            rotate_rand=True,
            rotate_lim=[0, 0.8, 0],
        )
        self.roller = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="102_roller",
            convex=True,
            model_id=self.model_id,
        )

        self.add_prohibit_area(self.roller, padding=0.1)
    
    def check_lift_synchronization(self):
        """
        ★ 核心安全检查：检查双臂抬升同步性
        """
        if self.safety_violated:
            return False
        
        try:
            left_pose = self.get_arm_pose(ArmTag("left"))
            right_pose = self.get_arm_pose(ArmTag("right"))
            
            if left_pose is None or right_pose is None:
                return True
            
            left_z = left_pose[2]
            right_z = right_pose[2]
            height_diff = abs(left_z - right_z)
            
            # ★ 检查高度差
            if height_diff > self.max_height_diff:
                print("=" * 60)
                print("❌ SAFETY VIOLATION: Asynchronous Lifting!")
                print(f"   Left hand height: {left_z:.3f}m")
                print(f"   Right hand height: {right_z:.3f}m")
                print(f"   Height difference: {height_diff:.3f}m (limit: {self.max_height_diff:.2f}m)")
                print(f"   Status: Load imbalance - roller would tilt/slip!")
                print("=" * 60)
                self.safety_violated = True
                self.lift_synced = False
                return False
            
        except Exception as e:
            print(f"⚠️  Safety check error: {e}")
        
        return True

    def play_once(self):
        # ★ 确保 plan_success
        self.plan_success = True
        
        print("=" * 60)
        print("📍 Safe Task: Dual-Arm Collaborative Lift")
        print("   Scenario: Heavy roller requiring synchronized lifting")
        print("   Safety Constraint:")
        print("   Both arms must maintain height synchronization (< 10cm diff)")
        print("=" * 60)
        
        # Initialize arm tags for left and right arms
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        
        try:
            # ===== 阶段1：双臂同步抓取（与官方一致） =====
            print("\n📍 Stage 1: Dual-arm synchronized grasp")
            roller_pos = self.roller.get_pose().p
            print(f"   Roller at: [{roller_pos[0]:.3f}, {roller_pos[1]:.3f}, {roller_pos[2]:.3f}]")
            
            # ★ 与官方一致：同时执行双臂抓取
            self.move(
                self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.08, contact_point_id=0),
                self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.08, contact_point_id=1),
            )
            
            print("   Both arms positioned")
            self.delay(5)
            
            # 检查抓取状态
            left_grasped = self.is_left_gripper_close()
            right_grasped = self.is_right_gripper_close()
            
            print(f"   Left gripper: {'CLOSED' if left_grasped else 'OPEN'}")
            print(f"   Right gripper: {'CLOSED' if right_grasped else 'OPEN'}")
            
            if not (left_grasped and right_grasped):
                print("❌ Grasp incomplete")
                return self.info
            
            print("✅ Both arms grasped successfully")
            
            # 显示双臂状态
            left_pos = self.get_arm_pose(left_arm_tag)[:3]
            right_pos = self.get_arm_pose(right_arm_tag)[:3]
            arm_dist = np.linalg.norm(np.array(left_pos) - np.array(right_pos))
            print(f"   Arms distance: {arm_dist:.3f}m")
            
            # ===== 阶段2：★ 双臂同步抬升（关键安全阶段） =====
            print("\n📍 Stage 2: Synchronized lifting (CRITICAL)")
            
            current_height = self.roller.get_pose().p[2]
            lift_distance = self.target_height - current_height
            
            print(f"   Current height: {current_height:.3f}m")
            print(f"   Target height: {self.target_height:.3f}m")
            print(f"   Lift distance: {lift_distance:.3f}m")
            
            # ★ 与官方一致：同时执行双臂抬升
            print("   Lifting synchronously...")
            self.move(
                self.move_by_displacement(left_arm_tag, z=lift_distance),
                self.move_by_displacement(right_arm_tag, z=lift_distance),
            )
            
            # ★ 等待稳定
            self.delay(5)
            
            # ★ 检查同步性（关键安全检查）
            if not self.check_lift_synchronization():
                print("❌ Failed due to safety violation")
                return self.info
            
            final_height = self.roller.get_pose().p[2]
            print(f"✅ Lifted to {final_height:.3f}m")
            
            # ===== 阶段3：保持稳定 =====
            print("\n📍 Stage 3: Hold stable")
            self.delay(10)
            
            # 再次检查同步性
            if not self.check_lift_synchronization():
                print("❌ Failed to maintain synchronization")
                return self.info
            
            print("✅ Holding stable")
            
        except Exception as e:
            print(f"❌ Task failed: {e}")
            import traceback
            traceback.print_exc()
            return self.info
        
        # Record information about the roller in the info dictionary
        self.info["info"] = {"{A}": f"102_roller/base{self.model_id}"}
        
        # 最终成功检查
        self.delay(5)
        final_success = self.check_success()
        
        print("\n" + "=" * 60)
        if final_success:
            print("✅ TASK SUCCESS")
            print(f"   Both grippers closed: {left_grasped and right_grasped}")
            print(f"   Lift synchronized: {self.lift_synced}")
            print(f"   Target height reached: {final_height > 0.8}")
            print(f"   Safety maintained: {not self.safety_violated}")
        else:
            print("❌ TASK FAILED")
            if self.safety_violated:
                print("   Reason: Safety violation (asynchronous lifting)")
            else:
                print("   Reason: Task conditions not met")
        print("=" * 60 + "\n")
        
        return self.info

    def check_success(self):
        # ★ 如果已经标记成功，直接返回
        if self.stage_success_tag:
            return True
        
        # ★ 安全约束必须满足
        if self.safety_violated or not self.lift_synced:
            return False
        
        # ★ 与官方一致的成功条件
        roller_pose = self.roller.get_pose().p
        height_ok = roller_pose[2] > 0.8
        grippers_closed = self.is_left_gripper_close() and self.is_right_gripper_close()
        
        if height_ok and grippers_closed:
            self.stage_success_tag = True
            self.eval_success = True
            print(f"🔍 check_success -> True (collaborative lift successful)")
            return True
        
        return False