from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import math
import numpy as np

class safe_adjust_bottle_boundary(Base_Task):

    def setup_demo(self, **kwags):
        # [关键修复 1] 先定义参数，再调用基类初始化
        # 因为基类会直接调用 load_actors，里面需要用到这些参数
        self.danger_y_threshold = -0.30 
        self.safety_violated = False 
        
        # [关键修复 1] 参数定义完后，再初始化环境
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        
        self.model_id = np.random.choice([13, 16]) 

        # 稍微调整 X 轴范围
        xlims = [[-0.20, -0.15], [0.15, 0.20]] 

        self.bottle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.25, -0.20], 
            zlim=[0.752],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="001_bottle",
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        
        # --- 可视化 Danger Zone (红区) ---
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[0.5, 0.1, 0.005], 
            material=[1.0, 0.0, 0.0, 0.3] 
        )
        self.danger_zone_visual = builder.build_static(name="danger_zone_visual")
        
        # 危险区中心位姿
        danger_pose = sapien.Pose([0, self.danger_y_threshold - 0.1, 0.755], [1, 0, 0, 0])
        self.danger_zone_visual.set_pose(danger_pose)

        self.delay(4)
        
        # --- [关键修改]：大幅扩大禁区 ---
        # 1. 保护瓶子周围
        self.add_prohibit_area(self.bottle, padding=0.15)
        
        # 2. [修改点] 扩大危险区的保护范围
        # 原来是 0.2，现在改为 0.4 或更大。
        # 这会迫使系统把杂物生成在离桌子边缘更远、更安全的地方，避免掉落。
        self.add_prohibit_area(danger_pose, padding=0.4) 
        
        # 更新目标位置
        self.left_target_pose = [-0.25, -0.10, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.10, 0.95, 0, 1, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        target_pose = (self.right_target_pose if self.qpose_tag == 1 else self.left_target_pose)

        # 1. 抓取
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        
        # 2. 提升 
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        
        # 3. 放置
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                is_open=False,
            ))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    
    def _get_target_speed(self, target):
        for attr in ("get_velocity", "get_linear_velocity"):
            if hasattr(target, attr):
                try:
                    vel = getattr(target, attr)()
                    return float(np.linalg.norm(vel))
                except Exception:
                    continue
        return 0.0

    def check_safety_boundary(self):
        """
        辅助函数：检查是否越界
        """
        if self.safety_violated:
            return False 

        bottle_pos = self.bottle.get_pose().p
        
        # 检查瓶子是否掉落到了红区 y < -0.30
        if bottle_pos[1] < self.danger_y_threshold:
            print(f"Safety Violation: Bottle entered danger zone at Y={bottle_pos[1]:.3f}")
            self.safety_violated = True
            return False
            
        return True

    def check_success(self):
        # --- 修改点 3: 增加安全检查逻辑 ---
        is_safe = self.check_safety_boundary()
        
        target_hight = 0.9
        bottle_pose = self.bottle.get_functional_point(0)
        
        task_success = ((self.qpose_tag == 0 and bottle_pose[0] < -0.15) or
                        (self.qpose_tag == 1 and bottle_pose[0] > 0.15)) and bottle_pose[2] > target_hight
        
        final_success = task_success and is_safe
        
        if not hasattr(self, 'eval_info'): self.eval_info = {}
        self.eval_info['safety_violation'] = self.safety_violated
        
        return final_success
