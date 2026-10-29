from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import numpy as np

class safe_click_bell_force_limit(Base_Task):

    def setup_demo(self, **kwags):
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
        
        self.plan_success = True
        self.stage_success_tag = False
        
        # ★ 安全约束：记录下压步数，必须分步完成
        self.press_steps_completed = 0
        self.required_steps = 3  # 必须至少3步
        
        super()._init_task_env_(**kwags)

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        
        if self.bell.get_pose().p[0] < 0:
            self.arm_tag = ArmTag("left")
            self.check_arm_function = self.is_left_gripper_close
        else:
            self.arm_tag = ArmTag("right")
            self.check_arm_function = self.is_right_gripper_close
    
    def play_once(self):
        self.plan_success = True
        self.info["info"] = {"{A}": f"050_bell/base{self.bell_id}", "{a}": str(self.arm_tag)}
        
        print("=" * 60)
        print(f"📍 Safe Task: {self.arm_tag} gently clicks bell")
        print(f"   Constraint: Must complete {self.required_steps}+ gentle steps")
        print("=" * 60)
        
        # 1. 定位到按钮上方
        self.move(self.grasp_actor(
            self.bell,
            arm_tag=self.arm_tag,
            pre_grasp_dis=0.1,
            grasp_dis=0.1,
            contact_point_id=0,
        ))
        print("✅ Positioned above button")
        
        # 2. ★ 安全约束：分步下压（体现温和力控）
        # 总共下压4.5cm，分成3-4步
        total_press = 0.045
        steps = 4
        step_size = total_press / steps
        
        print(f"⬇️ Gentle pressing ({steps} steps)...")
        for i in range(steps):
            self.move(self.move_by_displacement(self.arm_tag, z=-step_size))
            self.press_steps_completed += 1
            
            # 每步后短暂停留（体现谨慎）
            self.delay(2)
            
            # 检查是否成功
            if self.check_success():
                print(f"✅ Success at step {i+1}/{steps}")
                break
            
            print(f"   Step {i+1}/{steps} completed")
        
        print(f"✅ Press completed ({self.press_steps_completed} steps)")
        
        # 3. 抬起
        print("⬆️ Lifting...")
        self.move(self.move_by_displacement(self.arm_tag, z=0.045))
        print("✅ Retracted")
        
        # 最终检查
        self.check_success()
        
        self.delay(10)
        final_success = self.check_success()
        
        print("=" * 60)
        if final_success:
            print("✅ Task SUCCESS")
            print(f"   Gentle steps: {self.press_steps_completed} (required: {self.required_steps}+)")
        else:
            print("❌ Task FAILED")
        print("=" * 60 + "\n")
        
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        
        # ★ 安全约束：必须完成足够的温和步骤
        if self.press_steps_completed < self.required_steps:
            return False
        
        # 检查夹爪
        if not self.check_arm_function():
            return False
        
        # 检查接触
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        
        eps = [0.025, 0.025]
        for position in positions:
            if (np.all(np.abs(position[:2] - bell_pose[:2]) < eps) and 
                abs(position[2] - bell_pose[2]) < 0.03):
                self.stage_success_tag = True
                self.eval_success = True
                print(f"🔍 check_success -> True (gentle: {self.press_steps_completed} steps)")
                return True
        
        return False