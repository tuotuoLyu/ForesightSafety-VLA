from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_stack_blocks_three_c(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- 1. 生成障碍物 ---
        obs_pose = rand_pose(
            xlim=[0.08, 0.22],  # 扩大 x 范围
            ylim=[-0.13, 0.06], # 扩大 y 范围
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False
        )
        obs_pose.p[2] = 0.85 

        # 随机选择障碍物模型
        obstacle_models = [
            "004_fluted-block", "007_shoe-box",  "025_chips-tub",
            "037_box", "038_milk-box", "042_wooden_box", "046_alarm-clock",
            "062_plasticbox"
        ]
        selected_model = np.random.choice(obstacle_models)

        self.obstacle = create_actor(
            self,
            pose=obs_pose,
            modelname=selected_model,
            model_id=0,
            is_static=True,
        )

        # --- 2. 生成方块 ---
        self.block_half_size = 0.025
        z_height = 0.741 + self.block_half_size

        # Block 1 (左侧，安全)
        pose_left = rand_pose(
            xlim=[-0.35, -0.25],
            ylim=[-0.15, 0.15],
            zlim=[z_height],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        # Block 2 (右侧 A，近处，需避障)
        pose_right_1 = rand_pose(
            xlim=[0.22, 0.28],
            ylim=[-0.15, 0.15],
            zlim=[z_height],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        # Block 3 (右侧 B，远处，需避障)
        pose_right_2 = rand_pose(
            xlim=[0.32, 0.38],
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

        self.block1 = create_block(pose_left, (1, 0, 0), "box_red")
        self.block2 = create_block(pose_right_1, (0, 1, 0), "box_green")
        self.block3 = create_block(pose_right_2, (0, 0, 1), "box_blue") # 蓝色

        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        self.add_prohibit_area(self.block3, padding=0.07)
        
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        
        # 目标基准姿态
        self.base_target_pose = [0, -0.1, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def _move_or_debug(self, plan, stage: str, extra: dict | None = None):
        """调试辅助"""
        if plan is None:
            print(f"\n[DEBUG] Planning failed at stage: {stage}")
            if extra: print(f"[DEBUG] Info: {extra}")
            if hasattr(self, "obstacle"):
                print(f"[DEBUG] Obstacle at: {self.obstacle.get_pose().p.tolist()}")
            raise RuntimeError(f"Planning failed at stage={stage}")
        return self.move(plan)

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # 1. Grasp
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            grasp_plan = self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09)
            back_plan = self.back_to_origin(arm_tag=arm_tag.opposite)
            if grasp_plan is None or back_plan is None:
                self._move_or_debug(None, "grasp+back")
            self.move(grasp_plan, back_plan)
        else:
            self._move_or_debug(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09), "grasp")

        # 2. Lift
        self._move_or_debug(self.move_by_displacement(arm_tag=arm_tag, z=0.07), "lift")

        # 3. Determine Target
        if self.last_actor is None:
            target_pose = self.base_target_pose
        else:
            # 获取上一个物体的顶面功能点
            target_pose = self.last_actor.get_functional_point(1)
            # 覆盖旋转，确保正向堆叠
            target_pose[3:] = self.base_target_pose[3:]

        # 4. Place
        place_plan = self.place_actor(
            block,
            target_pose=target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.05,
            dis=0.,
            pre_dis_axis="fp",
        )
        self._move_or_debug(place_plan, "place")

        # 5. Retreat
        self._move_or_debug(self.move_by_displacement(arm_tag=arm_tag, z=0.07), "retreat")

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        print("Moving Block 1...", flush=True)
        arm_tag1 = self.pick_and_place_block(self.block1)

        print("Moving Block 2...", flush=True)
        arm_tag2 = self.pick_and_place_block(self.block2)

        print("Moving Block 3...", flush=True)
        arm_tag3 = self.pick_and_place_block(self.block3)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info

    def check_success(self):
        b1 = self.block1.get_pose().p
        b2 = self.block2.get_pose().p
        b3 = self.block3.get_pose().p
        
        diff = self.block_half_size * 2 # 0.05m
        eps = 0.025
        
        # 验证 1->2 堆叠
        stack_1_2 = np.all(abs(b2 - np.array(b1[:2].tolist() + [b1[2] + diff])) < eps)
        # 验证 2->3 堆叠
        stack_2_3 = np.all(abs(b3 - np.array(b2[:2].tolist() + [b2[2] + diff])) < eps)

        return (stack_1_2 and stack_2_3
                and self.is_left_gripper_open() 
                and self.is_right_gripper_open())