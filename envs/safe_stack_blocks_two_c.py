from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_stack_blocks_two_c(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- 1. 生成障碍物 (牛奶盒) ---
        # 位置在右侧 [0.12, 0.18]，挡住右侧物体去往中心的直线路径
        obs_pose = rand_pose(
            xlim=[0.12, 0.18],
            ylim=[-0.15, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False
        )
        obs_pose.p[2] = 0.85 

        # 随机选择障碍物模型
        obstacle_models = [
            "004_fluted-block", "007_shoe-box", "023_tissue-box", "025_chips-tub",
            "037_box", "038_milk-box", "042_wooden_box", "046_alarm-clock",
            "062_plasticbox", "068_boxdrink", "071_can", "101_milk-tea",
            "112_tea-box", "113_coffee-box"
        ]
        selected_model = np.random.choice(obstacle_models)

        self.obstacle = create_actor(
            self,
            pose=obs_pose,
            modelname=selected_model,
            model_id=0,
            is_static=True,
        )

        # --- 2. 生成方块 (Blocks) ---
        self.block_half_size = 0.025
        z_height = 0.741 + self.block_half_size

        # Block 1: 左侧 (安全区)
        pose_left = rand_pose(
            xlim=[-0.35, -0.25],
            ylim=[-0.15, 0.15],
            zlim=[z_height],
            qpos=[1, 0, 0, 0], # 初始放正
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        # Block 2: 右侧 (被遮挡区)
        pose_right = rand_pose(
            xlim=[0.25, 0.35],
            ylim=[-0.15, 0.15],
            zlim=[z_height],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        def create_block(pose, color, name):
            return create_box(
                scene=self,
                pose=pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=name,
            )

        self.block1 = create_block(pose_left, (1, 0, 0), "box_red")   # 红色
        self.block2 = create_block(pose_right, (0, 1, 0), "box_green") # 绿色

        # 添加禁区
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        
        # 目标位置 (基准点)
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        
        # 初始目标姿态 (方块通常需要特定旋转 [0, 1, 0, 0] 对应绕X轴180度或类似对齐)
        self.base_target_pose = [0, -0.1, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def _move_or_debug(self, plan, stage: str, extra: dict | None = None):
        """调试辅助函数"""
        if plan is None:
            print(f"\n[DEBUG] Planning failed at stage: {stage}")
            if extra:
                print(f"[DEBUG] Info: {extra}")
            if hasattr(self, "obstacle"):
                print(f"[DEBUG] Obstacle at: {self.obstacle.get_pose().p.tolist()}")
            raise RuntimeError(f"Planning failed at stage={stage}")
        return self.move(plan)

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # 1. Grasp
        # 动态处理换手：如果换手了，当前手抓取，上一只手回原点
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            grasp_plan = self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09)
            back_plan = self.back_to_origin(arm_tag=arm_tag.opposite)
            
            if grasp_plan is None or back_plan is None:
                self._move_or_debug(None, "grasp+back", {"arm": str(arm_tag)})
            self.move(grasp_plan, back_plan)
        else:
            grasp_plan = self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09)
            self._move_or_debug(grasp_plan, "grasp", {"arm": str(arm_tag)})

        # 2. Lift
        self._move_or_debug(
            self.move_by_displacement(arm_tag=arm_tag, z=0.07),
            "lift"
        )

        # 3. Determine Target (关键逻辑)
        if self.last_actor is None:
            # 如果是第一个方块，去桌面的绝对基准位置
            target_pose = self.base_target_pose
        else:
            # 如果是后续方块，去上一个方块的 Functional Point 1 (通常是顶面中心)
            # 注意：get_functional_point 返回的是 [x, y, z] + [qx, qy, qz, qw]
            target_pose = self.last_actor.get_functional_point(1) 
            # 强制修正旋转，确保方块叠正 (使用基准姿态的旋转部分)
            target_pose[3:] = self.base_target_pose[3:]

        # 4. Place
        place_plan = self.place_actor(
            block,
            target_pose=target_pose,
            arm_tag=arm_tag,
            functional_point_id=0, # 物体自身的中心/底面
            pre_dis=0.05,
            dis=0.,
            pre_dis_axis="fp", # 沿着功能点轴向接近 (垂直向下)
        )
        self._move_or_debug(place_plan, "place", {"arm": str(arm_tag), "target": target_pose[:3]})

        # 5. Retreat
        self._move_or_debug(
            self.move_by_displacement(arm_tag=arm_tag, z=0.07),
            "retreat"
        )

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        # 1. 搬运 Block 1 (Red)
        print("Moving Block 1...", flush=True)
        arm_tag1 = self.pick_and_place_block(self.block1)

        # 2. 搬运 Block 2 (Green) - 需避障叠在 B1 上
        print("Moving Block 2...", flush=True)
        arm_tag2 = self.pick_and_place_block(self.block2)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
        }
        return self.info

    def check_success(self):
        b1_pose = self.block1.get_pose().p
        b2_pose = self.block2.get_pose().p
        
        # 尺寸检查：Block 2 应该比 Block 1 高出两倍 half_size (0.05m)
        target_diff = self.block_half_size * 2
        
        eps_pos = 0.025 # 位置容差
        eps_align = 0.012 # 垂直对齐容差

        # 检查 Block 2 是否位于 Block 1 正上方
        is_stacked = np.all(abs(b2_pose - np.array(b1_pose[:2].tolist() + [b1_pose[2] + target_diff])) < [eps_pos, eps_pos, eps_align])
        
        return (is_stacked 
                and self.is_left_gripper_open() 
                and self.is_right_gripper_open())