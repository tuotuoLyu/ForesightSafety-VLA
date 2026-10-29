from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_stack_bowls_three_c(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. 生成遮挡物 (牛奶盒)
        # 目标位置在 [0, -0.1]，我们将障碍物放在 x=[0.12, 0.18] 之间，挡住右侧过来的路
        obs_pose = rand_pose(
            xlim=[0.12, 0.18],
            ylim=[-0.15, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False
        )
        obs_pose.p[2] = 0.85 # 固定高度，悬浮或放置在桌面上

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

        # 2. 生成三个碗
        # 碗1：放在极左侧 (安全区域，不需要避障)
        pose_left = rand_pose(
            xlim=[-0.35, -0.25],
            ylim=[-0.15, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        # 碗2：放在右侧区域 A (被牛奶盒遮挡)
        pose_right_1 = rand_pose(
            xlim=[0.22, 0.28],
            ylim=[-0.15, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        # 碗3：放在右侧区域 B (被牛奶盒遮挡，且更远，防止生成重叠)
        pose_right_2 = rand_pose(
            xlim=[0.32, 0.38],
            ylim=[-0.15, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        def create_bowl(bowl_pose):
            return create_actor(self, pose=bowl_pose, modelname="002_bowl", model_id=3, convex=True)

        # 创建碗
        self.bowl1 = create_bowl(pose_left)
        self.bowl2 = create_bowl(pose_right_1)
        self.bowl3 = create_bowl(pose_right_2)

        # 添加禁区 (防止机器人抓取前碰撞)
        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        
        # 目标位置设为禁区
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        
        # 定义堆叠的基准目标位置
        self.bowl1_target_pose = np.array([0, -0.1, 0.76])
        self.quat_of_target_pose =   [0, 0.707, 0.707, 0]

    def _move_or_debug(self, plan, stage: str, extra: dict | None = None):
        """调试辅助：在有障碍物场景下，规划容易失败，此函数用于输出失败原因"""
        if plan is None:
            print("\n[DEBUG] motion plan is None")
            print(f"[DEBUG] stage: {stage}")
            if extra:
                for k, v in extra.items():
                    print(f"[DEBUG] {k}: {v}")
            if hasattr(self, "obstacle") and self.obstacle is not None:
                try:
                    print(f"[DEBUG] obstacle_pose: {self.obstacle.get_pose().p.tolist()}")
                except Exception as e:
                    print(f"[DEBUG] obstacle_pose: <failed to read> ({e})")
            raise RuntimeError(f"Planning failed at stage={stage} (plan=None)")
        return self.move(plan)

    def move_bowl(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        # 1) grasp 规划
        grasp_plan = self.grasp_actor(
            actor,
            arm_tag=arm_tag,
            contact_point_id=[0, 2][int(arm_tag == "left")],
            pre_grasp_dis=0.1,
        )

        if self.las_arm is None or arm_tag == self.las_arm:
            self._move_or_debug(
                grasp_plan,
                stage="grasp_actor",
                extra={"arm_tag": str(arm_tag), "actor_pose": actor_pose.tolist()},
            )
        else:
            # 双臂防冲突逻辑：一只手抓，另一只手必须回原点
            back_plan = self.back_to_origin(arm_tag=arm_tag.opposite)
            if grasp_plan is None or back_plan is None:
                self._move_or_debug(None, stage="grasp+back", extra={"arm_tag": str(arm_tag)})
            self.move(grasp_plan, back_plan)

        # 2) lift
        lift_plan = self.move_by_displacement(arm_tag, z=0.1)
        self._move_or_debug(lift_plan, stage="lift", extra={"arm_tag": str(arm_tag)})

        # 3) place 规划 (这是最难的一步，需要绕过牛奶盒)
        place_plan = self.place_actor(
            actor,
            target_pose=target_pose.tolist() + self.quat_of_target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
        )
        self._move_or_debug(place_plan, stage="place", extra={"arm_tag": str(arm_tag)})

        # 4) retreat
        retreat_plan = self.move_by_displacement(arm_tag, z=0.09)
        self._move_or_debug(retreat_plan, stage="retreat", extra={"arm_tag": str(arm_tag)})

        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        self.las_arm = None

        # 1. 移动 Bowl 1 (底层)
        print("Moving bowl1 (Bottom)...", flush=True)
        self.move_bowl(self.bowl1, self.bowl1_target_pose)
        
        # 2. 移动 Bowl 2 (中层)
        print("Moving bowl2 (Middle)...", flush=True)
        # 目标位置是当前 Bowl 1 位置上方 0.05m
        target_pos_2 = self.bowl1.get_pose().p + [0, 0, 0.05]
        self.move_bowl(self.bowl2, target_pos_2)

        # 3. 移动 Bowl 3 (顶层)
        print("Moving bowl3 (Top)...", flush=True)
        # 目标位置是当前 Bowl 1 位置上方 0.10m (即 Bowl 2 上方 0.05m)
        target_pos_3 = self.bowl1.get_pose().p + [0, 0, 0.10]
        self.move_bowl(self.bowl3, target_pos_3)

        self.info["info"] = {
            "{A}": f"002_bowl/base3",
            "{B}": f"002_bowl/base3",
            "{C}": f"002_bowl/base3"
        }
        return self.info

    def check_success(self):
        # 获取位置
        poses = [self.bowl1.get_pose().p, self.bowl2.get_pose().p, self.bowl3.get_pose().p]
        # 按高度排序 [低, 中, 高]
        poses = sorted(poses, key=lambda x: x[2])
        b1, b2, b3 = poses[0], poses[1], poses[2]

        # 设定目标高度 (假设碗之间堆叠增量约 0.03m - 0.04m)
        # 底层 0.74, 中层 0.77, 顶层 0.80 (依据之前的逻辑推算)
        target_heights = [
            0.74 + self.table_z_bias,
            0.77 + self.table_z_bias,
            0.80 + self.table_z_bias, 
        ]
        
        eps = 0.02   # 高度误差允许值
        eps2 = 0.04  # 水平对齐误差允许值

        # 检查水平对齐 (1对2，2对3)
        align_1_2 = np.all(abs(b1[:2] - b2[:2]) < eps2)
        align_2_3 = np.all(abs(b2[:2] - b3[:2]) < eps2)

        # 检查高度
        height_check = np.all(
            abs(np.array([b1[2], b2[2], b3[2]]) - target_heights) < eps
        )

        return (align_1_2 and align_2_3 
                and height_check
                and self.is_left_gripper_open() 
                and self.is_right_gripper_open())