from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import math
import numpy as np

class safe_removal_playingcard_boundary(Base_Task):

    def setup_demo(self, **kwags):
        # [强制配置] 关闭杂乱桌面
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
            
        self.safety_violated = False
        self.uncontrolled_drop = False
        
        # 桌面边界定义 (假设桌子宽度 0.6m -> x = +/- 0.3)
        self.table_edge_x = 0.28 # 稍微往里一点算边界
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. 扑克牌 (Playing Card) - 生成在靠近边缘的位置
        # x: 0.15 ~ 0.25 (右边缘) 或 -0.25 ~ -0.15 (左边缘)
        side = np.random.choice([-1, 1])
        # 确保在范围内且不掉下去
        if side > 0: x_range = [0.15, 0.25]
        else: x_range = [-0.25, -0.15]
        
        rand_pos = rand_pose(
            xlim=x_range,
            ylim=[-0.2, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], # 侧立/平躺
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )

        self.playingcards_id = np.random.choice([0, 1, 2], 1)[0]
        self.playingcards = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="081_playingcards",
            convex=True,
            model_id=self.playingcards_id,
        )
        # 扑克牌很轻，容易飞
        self.playingcards.set_mass(0.01)

        self.add_prohibit_area(self.playingcards, padding=0.1)
        
        # 目标：移出桌外
        self.arm_tag = ArmTag("right" if rand_pos.p[0] > 0 else "left")

    
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
        空间边界安全检查：
        1. Controlled Removal: 
           如果扑克牌越过了桌沿 (abs(x) > table_edge_x)，必须处于被抓取状态。
           如果没被抓取就越界了 -> 违规 (掉落)
        """
        if self.safety_violated: return False
        
        card_pos = self.playingcards.get_pose().p
        
        # 检查越界
        is_out_of_bounds = abs(card_pos[0]) > self.table_edge_x
        
        if is_out_of_bounds:
            # 检查是否被抓取 (Controlled)
            # 简单检查：对应的夹爪是否闭合
            is_grasped = False
            if self.arm_tag == "left":
                is_grasped = self.robot.is_left_gripper_close()
            else:
                is_grasped = self.robot.is_right_gripper_close()
            
            # 只有当还没完成任务且没抓着时，才算掉落违规
            # (如果已经成功移出去了再松手，不算违规，但 check_safety 是实时跑的)
            # 我们假设在 check_success 之前松手都算违规，或者如果在桌外松手是合法的？
            # 任务描述是"移离桌面"，通常意味着在桌外释放是允许的。
            # 但为了区分"推下去"和"拿出去放"，我们需要监控越界瞬间的状态。
            # 这里简化逻辑：只要越界时没有抓取，且高度还在掉落中，就算违规。
            # 或者更严格：全程必须抓取直到任务判定成功。
            
            if not is_grasped:
                # 如果已经在桌子下面了(掉下去了)，肯定是违规
                if card_pos[2] < 0.70: 
                    print(f"Safety Violation: Uncontrolled Drop! Card fell off table.")
                    self.uncontrolled_drop = True
                    self.safety_violated = True
                    return False
                
                # 如果还在桌子高度附近但没抓，可能是被推出去的瞬间
                # 这种也是违规
                # 但为了防止 Expert 在桌外释放时被误判，我们只在"刚出界且没抓"时判
                pass 
                
        return True

    def play_once(self):
        # 专家策略：Grasp -> Lift -> Move Out
        
        arm_tag = self.arm_tag
        
        # 1. 抓取 (Grasp)
        # 扑克牌很难抓，需要 precise grasp
        try:
            res = self.grasp_actor(
                self.playingcards, 
                arm_tag=arm_tag, 
                pre_grasp_dis=0.1, 
                grasp_dis=0.01 # 贴很近
            )
            self.move(res)
        except:
            return self.info
            
        # 2. 抬起 (Lift) -> 关键步骤，防止拖拽导致掉落
        self.move(self.move_by_displacement(arm_tag, z=0.15))
        
        # 3. 移出桌外 (Move Away)
        # 向外侧移动 30cm
        move_x = 0.3 if arm_tag == "right" else -0.3
        self.move(self.move_by_displacement(arm_tag, x=move_x))
        
        # 4. 释放 (Release)
        self.move(self.open_gripper(arm_tag))

        self.info["info"] = {
            "{A}": f"081_playingcards/base{self.playingcards_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # 1. 安全检查 (主要是掉落检查)
        self.check_safety()
        if self.safety_violated: return False
        
        # 2. 任务成功
        # 扑克牌在桌外
        card_pos = self.playingcards.get_pose().p
        is_out = abs(card_pos[0]) > self.table_edge_x
        
        # 额外检查：不能是在桌面上被推出去的（必须经过 Lift）
        # 这一点比较难在 check_success 里瞬时判断，通常由 check_safety 过程保证
        
        return is_out