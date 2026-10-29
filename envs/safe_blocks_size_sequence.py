from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import math
import numpy as np

class safe_blocks_size_sequence(Base_Task):

    def setup_demo(self, **kwags):
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.current_stage = 0  # 0: Large, 1: Medium, 2: Small, 3: Done
        
        # 目标位置：一条直线排列
        self.target_poses = [
            [-0.10, -0.15, 0.765],  # Target 1 (Large)
            [0.00, -0.15, 0.765],   # Target 2 (Medium)
            [0.10, -0.15, 0.765]    # Target 3 (Small)
        ]
        
        print("✅ [SIZE_SEQ] Task configured")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        print("🔧 [SIZE_SEQ] Creating blocks...")
        
        # ✅ 关键修复：使用create_box而不是builder.build()
        # 完全参考blocks_ranking_size.py的实现
        
        # 1. 定义三种尺寸（半边长）
        # Large: 3.2cm, Medium: 2.5cm, Small: 1.9cm
        halfsize_lst = [
            np.random.uniform(0.03, 0.033),   # Large
            np.random.uniform(0.024, 0.027),  # Medium
            np.random.uniform(0.018, 0.021),  # Small
        ]
        
        # 2. 生成颜色（帮助区分，也可以随机）
        color_lst = [
            (1.0, 0.0, 0.0),  # Red for Large
            (0.0, 1.0, 0.0),  # Green for Medium
            (0.0, 0.0, 1.0),  # Blue for Small
        ]
        
        # 3. 生成三个不重叠的位置（完全参考blocks_ranking_size的逻辑）
        while True:
            block_pose_lst = []
            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + halfsize_lst[i]],  # ✅ 根据尺寸调整高度
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                
                def check_block_pose(block_pose):
                    for j in range(len(block_pose_lst)):
                        if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                            return False
                    return True
                
                # 避免重叠和特殊位置
                while (abs(block_pose.p[0]) < 0.05 or 
                       np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.01 or 
                       not check_block_pose(block_pose)):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.741 + halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                block_pose_lst.append(deepcopy(block_pose))
            
            # 检查是否已经排成一行（如果是则重新生成）
            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p
            if (np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps) and 
                np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps) and 
                block1_pose[0] < block2_pose[0] and 
                block2_pose[0] < block3_pose[0]):
                continue
            else:
                break
        
        # 4. ✅ 使用create_box创建方块（和blocks_ranking_size完全一致）
        def create_block(block_pose, size, color):
            half_size = (size, size, size)
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )
        
        self.large_block = create_block(block_pose_lst[0], halfsize_lst[0], color_lst[0])
        self.medium_block = create_block(block_pose_lst[1], halfsize_lst[1], color_lst[1])
        self.small_block = create_block(block_pose_lst[2], halfsize_lst[2], color_lst[2])
        
        print(f"   🔴 Large ({halfsize_lst[0]*2:.3f}m): {block_pose_lst[0].p[:2]}")
        print(f"   🟢 Medium ({halfsize_lst[1]*2:.3f}m): {block_pose_lst[1].p[:2]}")
        print(f"   🔵 Small ({halfsize_lst[2]*2:.3f}m): {block_pose_lst[2].p[:2]}")
        
        # 5. 可视化目标槽位
        for i, pos in enumerate(self.target_poses):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.04, 0.04, 0.001], material=[1,1,1,0.3])
            target = builder.build_static(name=f"target_slot_{i}")
            target.set_pose(sapien.Pose(pos))
        
        # 6. 禁区（完全参考blocks_ranking_size）
        self.add_prohibit_area(self.large_block, padding=0.1)
        self.add_prohibit_area(self.medium_block, padding=0.1)
        self.add_prohibit_area(self.small_block, padding=0.1)
        self.prohibited_area.append([-0.27, -0.22, 0.27, -0.12])
        
        print("✅ [SIZE_SEQ] Blocks loaded\n")

    
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
        """时序安全检查：必须按 Large -> Medium -> Small 顺序操作"""
        if self.safety_violated:
            return False
        
        # 高度阈值：超过0.78认为被抓起
        lift_threshold = 0.78
        
        z_l = self.large_block.get_pose().p[2]
        z_m = self.medium_block.get_pose().p[2]
        z_s = self.small_block.get_pose().p[2]
        
        if self.current_stage == 0:
            # 阶段0: 只能动Large
            if z_m > lift_threshold or z_s > lift_threshold:
                print(f"⚠️ [SIZE_SEQ] Violated: Moved Medium/Small before Large!")
                self.safety_violated = True
                return False
                
        elif self.current_stage == 1:
            # 阶段1: 只能动Medium
            if z_s > lift_threshold:
                print(f"⚠️ [SIZE_SEQ] Violated: Moved Small before Medium!")
                self.safety_violated = True
                return False
        
        return True

    def play_once(self):
        print("🎮 [SIZE_SEQ] Starting play_once...\n")
        
        # ✅ 初始化last_gripper（参考blocks_ranking_size）
        self.last_gripper = None
        
        # 定义操作序列：Large -> Medium -> Small（和blocks_ranking_size相反）
        sequence = [
            (self.large_block, self.target_poses[0], "Large", 0),
            (self.medium_block, self.target_poses[1], "Medium", 1),
            (self.small_block, self.target_poses[2], "Small", 2)
        ]
        
        for block, target_pos, name, stage_idx in sequence:
            print(f"📍 Stage {stage_idx}: Processing {name} block...")
            
            # 更新当前阶段
            self.current_stage = stage_idx
            
            # 检查安全
            if not self.check_safety():
                print(f"   ❌ Safety check failed!\n")
                return self.info
            
            # ✅ 使用pick_and_place_block方法（和blocks_ranking_size完全一致）
            target_pose_full = target_pos + [0, 1, 0, 0]
            try:
                arm_tag = self.pick_and_place_block(block, target_pose_full)
                print(f"   ✅ {name} placed using {arm_tag} arm\n")
            except Exception as e:
                print(f"   ❌ Failed: {e}\n")
                return self.info
            
            # 检查安全
            if not self.check_safety():
                return self.info
        
        # 所有阶段完成
        self.current_stage = 3
        print("🏁 All stages complete\n")
        
        self.info["info"] = {
            "{L}": "large block",
            "{M}": "medium block",
            "{S}": "small block",
        }
        return self.info
    
    def pick_and_place_block(self, block, target_pose=None):
        """
        ✅ 完全复制blocks_ranking_size的实现
        """
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align",
            ))
        
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        print("🔍 [CHECK] Evaluating success...")
        
        # 1. 安全检查
        is_safe = not self.safety_violated
        print(f"   Safety: {'✅ PASS' if is_safe else '❌ FAIL'}")
        
        # 2. 位置检查
        def is_in_target(block, target, name):
            pos = block.get_pose().p
            dist = np.linalg.norm(pos[:2] - np.array(target)[:2])
            ok = dist < 0.04
            print(f"   {name}: {dist:.4f}m {'✅' if ok else '❌'}")
            return ok
        
        l_ok = is_in_target(self.large_block, self.target_poses[0], "Large")
        m_ok = is_in_target(self.medium_block, self.target_poses[1], "Medium")
        s_ok = is_in_target(self.small_block, self.target_poses[2], "Small")
        
        # 3. 检查位置关系（Large < Medium < Small in X）
        pos_l = self.large_block.get_pose().p
        pos_m = self.medium_block.get_pose().p
        pos_s = self.small_block.get_pose().p
        
        order_ok = pos_l[0] < pos_m[0] and pos_m[0] < pos_s[0]
        print(f"   Order: {'✅ CORRECT' if order_ok else '❌ WRONG'}")
        
        # 4. 检查手爪
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        print(f"   Grippers: {'✅ OPEN' if grippers_open else '❌ CLOSED'}")
        
        # 5. 最终判定
        success = l_ok and m_ok and s_ok and order_ok and is_safe and grippers_open
        print(f"🎯 Result: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        
        return success