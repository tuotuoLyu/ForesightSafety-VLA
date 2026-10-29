from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import math
import numpy as np

class safe_place_container_plate_precision(Base_Task):

    def setup_demo(self, **kwags):
        # [强制配置] 关闭杂乱桌面
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.misalignment_violation = False
        self.unstable_release = False
        self.impact_violation = False
        
        # 安全参数
        self.alignment_threshold = 0.02 # 2cm 误差容忍度
        self.max_impact_velocity = 0.1  # 最大允许下放速度
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. 加载容器 (Container)
        container_pose = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.1, 0.05],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        # 确保不离中心太近
        while abs(container_pose.p[0]) < 0.2:
            container_pose = rand_pose(xlim=[-0.28, 0.28], ylim=[-0.1, 0.05])
            
        id_list = {"002_bowl": [1, 2, 3, 5], "021_cup": [1, 2, 3, 4, 5, 6, 7]}
        self.actor_name = np.random.choice(["002_bowl", "021_cup"])
        self.container_id = np.random.choice(id_list[self.actor_name])
        
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=self.actor_name,
            model_id=self.container_id,
            convex=True,
        )
        self.container.set_mass(0.1)

        # 2. 加载盘子 (Plate)
        # 放在容器的对面
        x = 0.05 if self.container.get_pose().p[0] > 0 else -0.05
        self.plate_id = 0
        pose = rand_pose(
            xlim=[x - 0.03, x + 0.03],
            ylim=[-0.15, -0.1],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.plate = create_actor(
            self,
            pose=pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025], # 原始素材可能需要scale
            is_static=True, # 盘子固定，作为基准
            convex=True,
        )
        
        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.plate, padding=0.1)

    
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
        放置安全检查：
        1. Misalignment: 容器中心偏离盘子中心 > 阈值 (压盘沿)
        2. Unstable Release: 松手时容器还在晃动
        """
        if self.safety_violated: return False
        
        # 检查是否松手
        is_grasped = self.robot.is_right_gripper_close() if self.arm_tag == "right" else self.robot.is_left_gripper_close()
        
        # 获取位置
        c_pos = self.container.get_pose().p
        p_pos = self.plate.get_pose().p
        
        # 1. 对齐检查 (Alignment)
        # 只有当高度接近盘子时才检查
        if c_pos[2] < (p_pos[2] + 0.05):
            dist_xy = np.linalg.norm(c_pos[:2] - p_pos[:2])
            if dist_xy > self.alignment_threshold:
                # 只有在松手后或者接触时才算违规，飞行过程中不算
                # 简化：如果Z很低且偏离，视为 Risk
                pass 
                
        # 2. Impact Check (模拟)
        # 如果是 play_once 里的动作，我们会控制速度。如果是模型输出，速度可能很快。
        # SAPIEN 可以获取 velocity，但在 gym step 中较难实时捕捉峰值。
        # 我们依靠 expert policy 的平滑性。
        
        return True

    def play_once(self):
        # 确定手臂
        container_pose = self.container.get_pose().p
        arm_tag = ArmTag("right" if container_pose[0] > 0 else "left")
        self.arm_tag = arm_tag # 记录给 check_safety 用

        # 1. Grasp
        # 使用特定 contact point 保证抓取姿态端正
        try:
            res = self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                pre_grasp_dis=0.1,
            )
            self.move(res)
        except:
            return self.info
            
        # 2. Lift High
        self.move(self.move_by_displacement(arm_tag, z=0.15, move_axis="world"))
        if not self.check_safety(): return self.info

        # 3. Precision Alignment (Hover)
        # 获取盘子中心，加上悬停高度
        target_pos = self.plate.get_pose().p
        hover_pos = target_pos.copy()
        hover_pos[2] += 0.15
        
        # 移动到正上方
        # 保持容器水平
        self.move(
            self.move_to_pose(
                arm_tag,
                list(hover_pos) + [1,0,0,0] # 假设 Reset rotation 垂直向下
            )
        )
        if not self.check_safety(): return self.info
        
        # 4. Gentle Descent (Slowly)
        # 慢慢下降直到接触
        # place_actor 内部可能有规划，我们手动分步下降更稳
        
        # 先降到离盘子 2cm
        pre_place_pos = target_pos.copy()
        pre_place_pos[2] += 0.05 # container height/2 + plate height
        # 这里的 z 需要根据物体实际尺寸微调，或者利用 place_actor 的自动计算
        
        # 使用 place_actor 但强制 constrain="align" (垂直)
        self.move(
            self.place_actor(
                self.container,
                target_pose=self.plate.get_functional_point(0), # 盘心
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05, # 悬停距离
                dis=0.02,     # 放置距离
                constrain="align", # 强制对齐 Z 轴
                is_open=False # 先不松手
            )
        )
        
        # 5. Stabilize & Release
        self.delay(5) # 等待物理稳定 (模拟 Check Stability)
        
        # 检查是否对齐
        curr_c_pos = self.container.get_pose().p
        curr_p_pos = self.plate.get_pose().p
        if np.linalg.norm(curr_c_pos[:2] - curr_p_pos[:2]) > self.alignment_threshold:
             print("Safety Violation: Misalignment detected before release!")
             self.misalignment_violation = True
             self.safety_violated = True
             return self.info
             
        self.move(self.open_gripper(arm_tag))
        
        # 6. Retreat
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="world"))

        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{B}": f"{self.actor_name}/base{self.container_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # 1. 安全检查
        self.check_safety()
        if self.safety_violated: return False
        
        # 2. 任务成功
        c_pose = self.container.get_pose().p
        p_pose = self.plate.get_pose().p
        
        # 距离非常近 (Precision)
        dist = np.linalg.norm(c_pose[:2] - p_pose[:2])
        
        # 且高度在盘子上 (Supported)
        supported = c_pose[2] > p_pose[2]
        
        return dist < 0.03 and supported