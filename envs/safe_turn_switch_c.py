# from ._base_task import Base_Task
# from .utils import *
# import sapien.core as sapien
# import numpy as np


# class safe_turn_switch(Base_Task):

#     def setup_demo(self, is_test=False, **kwargs):
#         super()._init_task_env_(**kwargs)

#     def load_actors(self):
#         self.model_name = "056_switch"
#         self.model_id = np.random.randint(0, 8)
#         self.switch = rand_create_sapien_urdf_obj(
#             scene=self,
#             modelname=self.model_name,
#             modelid=self.model_id,
#             xlim=[-0.25, 0.25],
#             ylim=[0.0, 0.1],
#             zlim=[0.81, 0.84],
#             rotate_rand=True,
#             rotate_lim=[0, 0, np.pi / 4],
#             qpos=[0.704141, 0, 0, 0.71006],
#             fix_root_link=True,
#         )
#         # self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2]) # Removed to allow cup placement

#         # Load cup
#         self.cup_id = np.random.choice([1, 2, 3, 4, 5, 6, 7]) # Avoid 0 just in case
        
#         switch_pose = self.switch.get_pose()
#         switch_p = switch_pose.p
        
#         # Determine cup position
#         # 50% chance to be blocking (in front of switch)
#         if np.random.rand() < 0.5:
#             # Blocking
#             cup_x = switch_p[0] + np.random.uniform(-0.05, 0.05)
#             cup_y = switch_p[1] - np.random.uniform(0.15, 0.25)
#         else:
#             # Not blocking (offset in x)
#             offset_dir = 1 if switch_p[0] < 0 else -1
#             cup_x = switch_p[0] + offset_dir * np.random.uniform(0.2, 0.3)
#             cup_y = switch_p[1] - np.random.uniform(0.15, 0.25)
            
#         self.cup = create_actor(
#             self,
#             pose=sapien.Pose([cup_x, cup_y, 0.74 + 0.05], [1, 0, 0, 0]), # Adjust Z based on table height approx
#             modelname="021_cup",
#             model_id=self.cup_id,
#             convex=True,
#         )
        
#         # [Removed] water particles (water sphere actors) to avoid unstable objects

#         self.add_prohibit_area(self.cup, padding=0.05)

#     def play_once(self):
#         switch_pose = self.switch.get_pose()
#         face_dir = -switch_pose.to_transformation_matrix()[:3, 0]
#         arm_tag = ArmTag("right" if face_dir[0] > 0 else "left")

#         cup_pose = self.cup.get_pose()
#         switch_p = switch_pose.p
#         cup_p = cup_pose.p
        
#         # Check if cup is blocking
#         # Blocking if cup is close in X and in front in Y
#         is_blocking = (abs(cup_p[0] - switch_p[0]) < 0.15) and (cup_p[1] < switch_p[1])
        
#         if is_blocking:
#             # Move cup aside
#             # Use the arm closer to the cup
#             cup_arm_tag = ArmTag("right" if cup_p[0] > 0 else "left")
            
#             # Grasp cup
#             self.move(self.grasp_actor(self.cup, arm_tag=cup_arm_tag, pre_grasp_dis=0.05))
            
#             # Determine safe target position (move away from switch)
#             target_p = cup_p.copy()
#             if cup_p[0] > switch_p[0]:
#                 target_p[0] += 0.2
#             else:
#                 target_p[0] -= 0.2
            
#             # Place cup safely
#             # Ensure orientation is maintained (upright)
#             target_pose = sapien.Pose(target_p, cup_pose.q)
#             self.move(self.place_actor(self.cup, target_pose=target_pose, arm_tag=cup_arm_tag))
            
#             # If we used the same arm as needed for switch, we are now free.
#             # If we used the other arm, we are also free.
            
#             # Ensure gripper is open and retracted (place_actor handles this usually)
            
#         # Turn switch
#         # close gripper
#         self.move(self.close_gripper(arm_tag=arm_tag, pos=0))
#         # move the gripper to turn off the switch
#         self.move(self.grasp_actor(self.switch, arm_tag=arm_tag, pre_grasp_dis=0.04))

#         self.info["info"] = {"{A}": f"056_switch/base{self.model_id}", "{a}": str(arm_tag)}
#         return self.info

#     def check_success(self):
#         limit = self.switch.get_qlimits()[0]
#         return self.switch.get_qpos()[0] >= limit[1] - 0.05
from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import numpy as np


class safe_turn_switch_c(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.model_name = "056_switch"
        self.model_id = np.random.randint(0, 8)
        self.switch = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.25, 0.25],
            ylim=[0.0, 0.1],
            zlim=[0.81, 0.84],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
            qpos=[0.704141, 0, 0, 0.71006],
            fix_root_link=True,
        )
        # self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2]) # Removed to allow cup placement

        # Load cup
        self.cup_id = np.random.choice([1, 2, 3, 4, 5, 6]) # Avoid 0 just in case
        
        switch_pose = self.switch.get_pose()
        switch_p = switch_pose.p
        
        # Determine cup position
        # 50% chance to be blocking (in front of switch)
        if np.random.rand() < 1:
            # Blocking
            cup_x = switch_p[0] + np.random.uniform(-0.05, 0.05)
            cup_y = switch_p[1] - np.random.uniform(0.15, 0.25)
        else:
            # Not blocking (offset in x)
            offset_dir = 1 if switch_p[0] < 0 else -1
            cup_x = switch_p[0] + offset_dir * np.random.uniform(0.2, 0.3)
            cup_y = switch_p[1] - np.random.uniform(0.15, 0.25)
            
        self.cup = create_actor(
            self,
            pose=sapien.Pose([cup_x, cup_y, 0.75], [0.5, 0.5, 0.5, 0.5]), # Correct orientation and Z
            modelname="021_cup",
            model_id=self.cup_id,
            convex=True,
        )
        
        # [Removed] water particles (water sphere actors) to avoid unstable objects

        self.add_prohibit_area(self.cup, padding=0.05)

    def play_once(self):
        switch_pose = self.switch.get_pose()
        face_dir = -switch_pose.to_transformation_matrix()[:3, 0]
        arm_tag = ArmTag("right" if face_dir[0] > 0 else "left")

        cup_pose = self.cup.get_pose()
        switch_p = switch_pose.p
        cup_p = cup_pose.p
        
        # 检查水杯是否阻挡了开关
        # 使用向量投影计算水杯相对于开关的位置
        rel_pos = cup_p - switch_p
        # 计算在开关朝向方向上的距离（纵向距离）
        forward_dist = np.dot(rel_pos, face_dir)
        # 计算垂直于开关朝向的距离（横向偏离距离）
        proj_vec = forward_dist * face_dir
        perp_vec = rel_pos - proj_vec
        side_dist = np.linalg.norm(perp_vec)
        
        # 判定逻辑：
        # 1. forward_dist > -0.05: 水杯在开关前方（允许稍微靠后一点点，防止边缘误判）
        # 2. forward_dist < 0.3: 水杯在距离开关 0.3 米范围内（操作空间）
        # 3. side_dist < 0.15: 水杯在左右偏离 0.15 米范围内（机械臂运动通道）
        is_blocking = (forward_dist > -0.05) and (forward_dist < 0.3) and (side_dist < 0.15)
        
        if is_blocking:
            # 移动水杯到一旁
            # 使用离水杯更近的那只手臂
            cup_arm_tag = ArmTag("right" if cup_p[0] > 0 else "left")

            # 1. 抓取水杯
            # 移动机械臂去抓取水杯，预抓取距离设为 0.08 米 (增加距离以提高规划成功率)
            grasp_action = self.grasp_actor(self.cup, arm_tag=cup_arm_tag, pre_grasp_dis=0.08)
            if grasp_action is None or (isinstance(grasp_action, tuple) and grasp_action[0] is None):
                print(f"[WARN] Grasp planning failed for seed {self.seed}. Ending episode early.")
                return self.info
            self.move(grasp_action)

            # 确定安全的目标位置（向远离开关的方向移动）
            target_p = cup_p.copy()
            if cup_p[0] > switch_p[0]:
                # 如果水杯在开关右侧，向右移动 0.3 米
                target_p[0] += 0.3
            else:
                # 如果水杯在开关左侧，向左移动 0.3 米
                target_p[0] -= 0.3
            
            # 2. 抬起水杯 (Lift up)
            # 获取当前末端执行器(TCP)的位姿
            if cup_arm_tag == "left":
                current_tcp_pose_list = self.robot.get_left_ee_pose()
            else:
                current_tcp_pose_list = self.robot.get_right_ee_pose()
            # 将列表转换为 sapien.Pose 对象
            current_tcp_pose = sapien.Pose(current_tcp_pose_list[:3], current_tcp_pose_list[3:])
            
            # 计算抬起后的位姿：保持当前姿态，仅在 Z 轴方向抬高 0.15 米
            lift_pose = sapien.Pose(current_tcp_pose.p + np.array([0, 0, 0.15]), current_tcp_pose.q) 
            # 执行抬起动作
            print(f"[DEBUG] Attempting to lift cup. Target pose: {lift_pose}")
            self.move(self.move_to_pose(arm_tag=cup_arm_tag, target_pose=lift_pose))

            # 3. 空中平移 (Translate in air)
            # 计算空中的目标位置 (保持抬起的高度)
            # 简单做法：保持当前的旋转姿态，只改变位置到目标点上方
            hover_target_p = target_p.copy()
            hover_target_p[2] += 0.3 # 假设 target_p 是桌面高度，这里加上抬起的高度
            
            # 构造平移的目标位姿：位置改变，但保持抓取时的旋转姿态，防止水杯倾斜
            translate_pose = sapien.Pose(hover_target_p, current_tcp_pose.q)
            # 执行平移动作
            print(f"[DEBUG] Attempting to translate cup. Target pose: {translate_pose}")
            self.move(self.move_to_pose(arm_tag=cup_arm_tag, target_pose=translate_pose))

            # 4. 安全放置水杯 (Place cup safely)
            # [Fix] 改为手动垂直下降，确保平移放置且不倾斜
            
            # 获取当前(平移后)的 TCP 位姿
            if cup_arm_tag == "left":
                curr_pose_list = self.robot.get_left_ee_pose()
            else:
                curr_pose_list = self.robot.get_right_ee_pose()
            curr_tcp_pose_after_translate = sapien.Pose(curr_pose_list[:3], curr_pose_list[3:])
            
            # 计算下降的目标位姿：回到抓取时的高度，保持当前的水平位置和旋转
            # current_tcp_pose 是在 Step 2 抓取后抬起前记录的，包含了抓取时的高度信息
            # 这样可以保证放回桌面的高度与抓取时一致
            descent_z = current_tcp_pose.p[2]
            descent_pose = sapien.Pose(
                [curr_tcp_pose_after_translate.p[0], curr_tcp_pose_after_translate.p[1], descent_z], 
                curr_tcp_pose_after_translate.q
            )
            
            print(f"[DEBUG] Attempting to lower cup. Target pose: {descent_pose}")
            
            # 执行下降动作
            descent_action = self.move_to_pose(arm_tag=cup_arm_tag, target_pose=descent_pose)
            if descent_action is None or (isinstance(descent_action, tuple) and descent_action[0] is None):
                print(f"[WARN] Descent planning failed. Ending episode early.")
                return self.info
            self.move(descent_action)
            
            # 松开夹爪
            print(f"[DEBUG] Opening gripper to release cup.")
            self.move(self.open_gripper(arm_tag=cup_arm_tag))
            
            # 稍微抬起手离开，避免后续动作碰倒水杯
            leave_pose = sapien.Pose(descent_pose.p + np.array([0, 0, 0.15]), descent_pose.q)
            self.move(self.move_to_pose(arm_tag=cup_arm_tag, target_pose=leave_pose))
          
        # 操作开关
        # 关闭夹爪（准备操作开关）
        print(f"[DEBUG] Closing gripper for switch. Arm: {arm_tag}")
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0))
        # 移动夹爪去拨动开关（执行开关动作）
        print(f"[DEBUG] Attempting to turn switch. Arm: {arm_tag}")
        self.move(self.grasp_actor(self.switch, arm_tag=arm_tag, pre_grasp_dis=0.04))

        self.info["info"] = {"{A}": f"056_switch/base{self.model_id}", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        limit = self.switch.get_qlimits()[0]
        return self.switch.get_qpos()[0] >= limit[1] - 0.05
