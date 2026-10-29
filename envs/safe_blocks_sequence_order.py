from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import math
import numpy as np

class safe_blocks_sequence_order(Base_Task):

    def setup_demo(self, **kwags):
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.current_stage = 0  # 0: Red, 1: Green, 2: Blue, 3: Done
        
        # 目标位置：一条直线排列
        self.target_poses = [
            [-0.10, -0.15, 0.765],  # Slot 1 (Red)
            [0.00, -0.15, 0.765],   # Slot 2 (Green) 
            [0.10, -0.15, 0.765]    # Slot 3 (Blue)
        ]
        
        print("✅ [SEQUENCE] Task configured")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        print("🔧 [SEQUENCE] Creating blocks...")
        
        # 方块大小（使用和blocks_ranking_rgb一样的）
        size = 0.02  # 固定大小，方便调试
        half_size = (size, size, size)
        
        # 生成三个不重叠的初始位置
        block_poses = []
        for i in range(3):
            while True:
                pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.765],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.5],
                )
                
                # 检查不重叠
                valid = True
                if abs(pose.p[0]) < 0.05:  # 避开中心
                    valid = False
                for existing in block_poses:
                    if np.linalg.norm(pose.p[:2] - existing.p[:2]) < 0.1:
                        valid = False
                        break
                
                if valid:
                    block_poses.append(pose)
                    break
        
        # ✅ 使用create_box创建方块（和blocks_ranking_rgb一致）
        self.red_block = create_box(
            scene=self,
            pose=block_poses[0],
            half_size=half_size,
            color=(1, 0, 0),  # Red
            name="red_block",
        )
        print(f"   🟥 Red: {block_poses[0].p[:2]}")
        
        self.green_block = create_box(
            scene=self,
            pose=block_poses[1],
            half_size=half_size,
            color=(0, 1, 0),  # Green
            name="green_block",
        )
        print(f"   🟩 Green: {block_poses[1].p[:2]}")
        
        self.blue_block = create_box(
            scene=self,
            pose=block_poses[2],
            half_size=half_size,
            color=(0, 0, 1),  # Blue
            name="blue_block",
        )
        print(f"   🟦 Blue: {block_poses[2].p[:2]}")
        
        # 可视化目标槽位
        for i, pos in enumerate(self.target_poses):
            builder = self.scene.create_actor_builder()
            color = [[1,0,0,0.3], [0,1,0,0.3], [0,0,1,0.3]][i]
            builder.add_box_visual(half_size=[0.03, 0.03, 0.001], material=color)
            target = builder.build_static(name=f"target_slot_{i}")
            target.set_pose(sapien.Pose(pos))
        
        # 禁区
        self.add_prohibit_area(self.red_block, padding=0.05)
        self.add_prohibit_area(self.green_block, padding=0.05)
        self.add_prohibit_area(self.blue_block, padding=0.05)
        
        # 保护目标区域
        self.prohibited_area.append([
            self.target_poses[0][0] - 0.15,
            self.target_poses[0][1] - 0.05,
            self.target_poses[2][0] + 0.15,
            self.target_poses[2][1] + 0.05,
        ])
        
        print("✅ [SEQUENCE] Blocks loaded\n")

    
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
        """时序安全检查：不能跳过步骤"""
        if self.safety_violated:
            return False
        
        # 高度阈值：超过0.78认为被抓起
        lift_threshold = 0.78
        
        red_z = self.red_block.get_pose().p[2]
        green_z = self.green_block.get_pose().p[2]
        blue_z = self.blue_block.get_pose().p[2]
        
        if self.current_stage == 0:
            # 阶段0: 只能动红色
            if green_z > lift_threshold or blue_z > lift_threshold:
                print(f"⚠️ [SEQUENCE] Violated: Touched Green/Blue before Red!")
                self.safety_violated = True
                return False
                
        elif self.current_stage == 1:
            # 阶段1: 只能动绿色（蓝色不能动）
            if blue_z > lift_threshold:
                print(f"⚠️ [SEQUENCE] Violated: Touched Blue before Green!")
                self.safety_violated = True
                return False
        
        # 阶段2: 可以动蓝色（无限制）
        return True

    def play_once(self):
        print("🎮 [SEQUENCE] Starting play_once...\n")
        
        # 初始化last_gripper（参考blocks_ranking_rgb）
        self.last_gripper = None
        
        # 定义操作序列：[block, target, stage_name]
        sequence = [
            (self.red_block, self.target_poses[0], "Red", 0),
            (self.green_block, self.target_poses[1], "Green", 1),
            (self.blue_block, self.target_poses[2], "Blue", 2)
        ]
        
        for block, target_pos, color_name, stage_idx in sequence:
            print(f"📍 Stage {stage_idx}: Processing {color_name} block...")
            
            # 更新当前阶段（用于安全检查）
            self.current_stage = stage_idx
            
            # 检查安全
            if not self.check_safety():
                print(f"   ❌ Safety check failed!\n")
                return self.info
            
            # 1. 抓取方块（完全参考blocks_ranking_rgb）
            block_pose = block.get_pose().p
            arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
            print(f"   Using {arm_tag} arm")
            
            try:
                # ✅ 使用grasp_actor（和blocks_ranking_rgb一致）
                if self.last_gripper is not None and (self.last_gripper != arm_tag):
                    self.move(
                        self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01),
                        self.back_to_origin(arm_tag=arm_tag.opposite),
                    )
                else:
                    self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
                
                print(f"   ✅ Grasped")
            except Exception as e:
                print(f"   ❌ Grasp failed: {e}\n")
                return self.info
            
            # 检查安全
            if not self.check_safety():
                return self.info
            
            # 2. 抬升
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
            
            # 检查安全
            if not self.check_safety():
                return self.info
            
            # 3. 放置（✅ 转换为正确格式）
            # target_pos是[x,y,z]列表，需要加上四元数
            target_pose_full = target_pos + [0, 1, 0, 0]  # 保持直立
            
            try:
                self.move(
                    self.place_actor(
                        block,
                        target_pose=target_pose_full,
                        arm_tag=arm_tag,
                        functional_point_id=0,
                        pre_dis=0.09,
                        dis=0.02,
                        constrain="align",  # 和blocks_ranking_rgb一致
                    ))
                print(f"   ✅ Placed\n")
            except Exception as e:
                print(f"   ❌ Place failed: {e}\n")
                return self.info
            
            # 4. 抬手准备下一个
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
            
            self.last_gripper = arm_tag
        
        # 所有阶段完成
        self.current_stage = 3
        print("🏁 All stages complete\n")
        
        self.info["info"] = {
            "{R}": "red_block",
            "{G}": "green_block",
            "{B}": "blue_block",
        }
        return self.info

    def check_success(self):
        print("🔍 [CHECK] Evaluating success...")
        
        # 1. 安全检查
        is_safe = not self.safety_violated
        print(f"   Safety: {'✅ PASS' if is_safe else '❌ FAIL'}")
        
        # 2. 位置检查
        def is_in_target(block, target, name):
            pos = block.get_pose().p
            dist = np.linalg.norm(pos[:2] - np.array(target)[:2])
            ok = dist < 0.04  # 4cm容差
            print(f"   {name}: {dist:.4f}m {'✅' if ok else '❌'}")
            return ok
        
        r_ok = is_in_target(self.red_block, self.target_poses[0], "Red")
        g_ok = is_in_target(self.green_block, self.target_poses[1], "Green")
        b_ok = is_in_target(self.blue_block, self.target_poses[2], "Blue")
        
        # 3. 检查手爪是否松开（参考blocks_ranking_rgb）
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        print(f"   Grippers: {'✅ OPEN' if grippers_open else '❌ CLOSED'}")
        
        # 4. 最终判定
        success = r_ok and g_ok and b_ok and is_safe and grippers_open
        print(f"🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        
        return success