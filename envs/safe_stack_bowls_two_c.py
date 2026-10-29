from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class safe_stack_bowls_two_c(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. 首先生成遮挡物 (牛奶盒)，位置随机但在右侧路径上，构成阻挡
        # 目标位置大约在 [0, -0.1]，我们将障碍物放在 [0.12, 0.18] 之间，挡住右侧过来的路
        obs_pose = rand_pose(
            xlim=[0.12, 0.18],
            ylim=[-0.15, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False
        )
        obs_pose.p[2] = 0.85 # 固定高度

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
        # 立即添加禁区，确保后续规划能看到它
        #self.add_prohibit_area(self.obstacle, padding=0.05)

        # 2. 生成两个碗，强制拉开距离，并确保一个被遮挡
        # 碗1：放在极左侧 (安全区域)
        pose_left = rand_pose(
            xlim=[-0.35, -0.25],
            ylim=[-0.15, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        # 碗2：放在极右侧 (被遮挡区域)
        # 它的 x > 障碍物 x，去往中心目标点时必须绕过障碍物
        pose_right = rand_pose(
            xlim=[0.25, 0.35],
            ylim=[-0.15, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=False,
        )

        def create_bowl(bowl_pose):
            return create_actor(self, pose=bowl_pose, modelname="002_bowl", model_id=3, convex=True)

        # 创建碗
        self.bowl1 = create_bowl(pose_left)
        self.bowl2 = create_bowl(pose_right)

        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        self.bowl1_target_pose = np.array([0, -0.1, 0.76])
        self.quat_of_target_pose =   [0, 0.707, 0.707, 0]

    def _move_or_debug(self, plan, stage: str, extra: dict | None = None):
        """调试辅助：防止 self.move(None) 导致 NoneType.get，并输出失败上下文。"""
        if plan is None:
            print("\n[DEBUG] motion plan is None")
            print(f"[DEBUG] stage: {stage}")
            if extra:
                for k, v in extra.items():
                    print(f"[DEBUG] {k}: {v}")
            # 尽量输出障碍物位置（如果存在）
            if hasattr(self, "obstacle") and self.obstacle is not None:
                try:
                    print(f"[DEBUG] obstacle_pose: {self.obstacle.get_pose().p.tolist()}")
                except Exception as e:
                    print(f"[DEBUG] obstacle_pose: <failed to read> ({e})")
            # 直接抛异常，让你在 VS Code 里看到准确堆栈位置
            raise RuntimeError(f"Planning failed at stage={stage} (plan=None)")
        # plan 非 None，正常执行
        return self.move(plan)

    def move_bowl(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        # 1) grasp 规划（先拿到 plan，再 move）
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
                extra={
                    "arm_tag": str(arm_tag),
                    "actor_pose": actor_pose.tolist(),
                },
            )
        else:
            # 注意：这里是“双动作 move”。任何一个为 None 都会出问题，所以分别判空。
            back_plan = self.back_to_origin(arm_tag=arm_tag.opposite)

            if grasp_plan is None or back_plan is None:
                self._move_or_debug(
                    None,
                    stage="grasp_actor + back_to_origin",
                    extra={
                        "arm_tag": str(arm_tag),
                        "actor_pose": actor_pose.tolist(),
                        "grasp_plan_is_none": grasp_plan is None,
                        "back_plan_is_none": back_plan is None,
                    },
                )
            self.move(grasp_plan, back_plan)

        # 2) lift
        lift_plan = self.move_by_displacement(arm_tag, z=0.1)
        self._move_or_debug(
            lift_plan,
            stage="move_by_displacement(lift)",
            extra={"arm_tag": str(arm_tag)},
        )

        # 3) place 规划（同样先拿 plan，再 move）
        place_plan = self.place_actor(
            actor,
            target_pose=target_pose.tolist() + self.quat_of_target_pose,
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.09,
            dis=0,
            constrain="align",
        )
        self._move_or_debug(
            place_plan,
            stage="place_actor",
            extra={
                "arm_tag": str(arm_tag),
                "target_pose_xyz": np.asarray(target_pose).tolist(),
            },
        )

        # 4) retreat lift
        retreat_plan = self.move_by_displacement(arm_tag, z=0.09)
        self._move_or_debug(
            retreat_plan,
            stage="move_by_displacement(retreat)",
            extra={"arm_tag": str(arm_tag)},
        )

        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        # Initialize last arm used to None
        self.las_arm = None

        # Move bowl1 to position [0, -0.1, 0.76]
        print("Moving bowl1 to target position...", flush=True)
        arm_tag1 = self.move_bowl(self.bowl1, self.bowl1_target_pose)
        print("Bowl1 moved.")
        # Move bowl2 to be 0.05m above bowl1's position
        arm_tag2 = self.move_bowl(self.bowl2, self.bowl1.get_pose().p + [0, 0, 0.05])

        # Store information about the bowls and arms used in the info dictionary
        self.info["info"] = {
            "{A}": f"002_bowl/base3",
            "{B}": f"002_bowl/base3",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
        }
        return self.info

    def check_success(self):
        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        bowl1_pose, bowl2_pose = sorted([bowl1_pose, bowl2_pose], key=lambda x: x[2])
        target_height = [
            0.74 + self.table_z_bias,
            0.77 + self.table_z_bias,
        ]
        eps = 0.02
        eps2 = 0.04
        return (np.all(abs(bowl1_pose[:2] - bowl2_pose[:2]) < eps2)
                and np.all(np.array([bowl1_pose[2], bowl2_pose[2]]) - target_height < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
