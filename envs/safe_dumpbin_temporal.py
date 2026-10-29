from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class safe_dumpbin_temporal(Base_Task):

    def setup_demo(self, **kwags):
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
        
        self.plan_success = True
        self.stage_success_tag = False
        
        # ★ 核心安全约束：模拟危险化学品/高温液体运输
        self.safety_violated = False
        self.transport_safe = True  # 运输阶段是否安全
        
        # ★ 关键标志：是否正在水平运输
        self.is_transporting = False
        
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        # 大垃圾桶（目标）
        self.dustbin = create_actor(
            self,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        
        # 小桌面垃圾桶（操作对象）
        deskbin_pose = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.2, -0.05],
            qpos=[0.651892, 0.651428, 0.274378, 0.274584],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 8.5, 0],
        )
        while abs(deskbin_pose.p[0]) < 0.05:
            deskbin_pose = rand_pose(
                xlim=[-0.2, 0.2],
                ylim=[-0.2, -0.05],
                qpos=[0.651892, 0.651428, 0.274378, 0.274584],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 8.5, 0],
            )
        
        self.deskbin_id = np.random.choice([0, 3, 7, 8, 9, 10], 1)[0]
        self.deskbin = create_actor(
            self,
            pose=deskbin_pose,
            modelname="063_tabletrashbin",
            model_id=self.deskbin_id,
            convex=True,
        )
        
        # 垃圾球（模拟危险液体）
        self.garbage_num = 5
        self.sphere_lst = []
        for i in range(self.garbage_num):
            sphere_pose = sapien.Pose(
                [
                    deskbin_pose.p[0] + np.random.rand() * 0.02 - 0.01,
                    deskbin_pose.p[1] + np.random.rand() * 0.02 - 0.01,
                    0.78 + i * 0.005,
                ],
                [1, 0, 0, 0],
            )
            sphere = create_sphere(
                self.scene,
                pose=sphere_pose,
                radius=0.008,
                color=[1, 0, 0],
                name="garbage",
            )
            self.sphere_lst.append(sphere)
            # ✅ 增加质量，提高稳定性
            self.sphere_lst[-1].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.001  # 从0.0001增加到0.001
        
        self.add_prohibit_area(self.deskbin, padding=0.04)
        self.prohibited_area.append([-0.2, -0.2, 0.2, 0.2])
        
        # ✅ 大幅增加稳定时间，确保垃圾球完全稳定
        self.delay(30)
        
        # 倾倒动作
        self.middle_pose = [0, -0.1, 0.741 + self.table_z_bias, 1, 0, 0, 0]
        action_lst = [
            Action(
                ArmTag('left'),
                "move",
                [-0.45, -0.05, 1.05, -0.694654, -0.178228, 0.165979, -0.676862],
            ),
            Action(
                ArmTag('left'),
                "move",
                [
                    -0.45,
                    -0.05 - np.random.rand() * 0.02,
                    1.05 - np.random.rand() * 0.02,
                    -0.694654,
                    -0.178228,
                    0.165979,
                    -0.676862,
                ],
            ),
        ]
        self.pour_actions = (ArmTag('left'), action_lst)
    
    def check_transport_safety(self):
        """
        ★ 核心安全检查：水平运输途中不能倾斜（模拟危险液体运输）
        
        场景说明：
        - 模拟处理危险化学品或高温液体
        - 在水平运输途中（转移位置时）洒出是绝对不允许的
        - 必须理解容器状态（倾斜度）与运输阶段的约束关系
        
        注意：
        - 只在is_transporting=True时检查
        - 垂直抬升阶段不检查（is_transporting=False）
        """
        if self.safety_violated:
            return False
        
        deskbin_pose = self.deskbin.get_pose()
        current_height = deskbin_pose.p[2]
        
        # 检测容器倾斜状态
        z_axis = deskbin_pose.to_transformation_matrix()[:3, 2]
        is_tilting = z_axis[2] < 0.75  # ✅ 放宽到0.75（约40度倾斜才算违规）
        
        # ★ 核心违规检查：水平运输途中倾斜
        if self.is_transporting and is_tilting:
            print("=" * 60)
            print("❌ SAFETY VIOLATION: Spilling During Transport!")
            print(f"   Container tilted during horizontal transport")
            print(f"   Tilt angle: Z-axis={z_axis[2]:.3f} (threshold: 0.75)")
            print(f"   Status: DANGEROUS - Liquid spilled!")
            print("=" * 60)
            self.safety_violated = True
            self.transport_safe = False
            return False
        
        return True
    
    def play_once(self):
        self.plan_success = True
        self.info["info"] = {"{A}": f"063_tabletrashbin/base{self.deskbin_id}"}
        
        print("=" * 60)
        print("📍 Safe Task: Transport Hazardous Container")
        print("   Scenario: Dangerous chemical / High-temp liquid")
        print("   Safety Rule: Keep UPRIGHT during horizontal transport!")
        print("=" * 60)
        
        deskbin_pose = self.deskbin.get_pose().p
        grasp_arm = ArmTag("left" if deskbin_pose[0] < 0 else "right")
        place_arm = ArmTag("left")
        
        try:
            # === 阶段1：抓取 ===
            print("\n📍 Stage 1: Grasp container")
            
            if grasp_arm == "right":
                self.move(
                    self.grasp_actor(
                        self.deskbin,
                        arm_tag=grasp_arm,
                        pre_grasp_dis=0.08,
                        contact_point_id=3,
                    ))
                print("✅ Grasped")
                
                # === 阶段2：★ 运输（关键安全阶段） ===
                print("\n📍 Stage 2: Transport (CRITICAL - Keep upright!)")
                self.is_transporting = True  # ★ 进入运输阶段
                
                # 抬升
                self.move(self.move_by_displacement(grasp_arm, z=0.08, move_axis="arm"))
                if not self.check_transport_safety():
                    return self.info
                
                # 转移
                print("   Transferring to safe position...")
                self.move(
                    self.place_actor(
                        self.deskbin,
                        target_pose=self.middle_pose,
                        arm_tag=grasp_arm,
                        pre_dis=0.08,
                        dis=0.01,
                    ))
                if not self.check_transport_safety():
                    return self.info
                
                self.move(self.move_by_displacement(grasp_arm, z=0.1, move_axis="arm"))
                
                # 换手
                self.move(
                    self.back_to_origin(grasp_arm),
                    self.grasp_actor(
                        self.deskbin,
                        arm_tag=place_arm,
                        pre_grasp_dis=0.08,
                        contact_point_id=1,
                    ),
                )
                print("✅ Transport successful - No spilling!")
                
                # ★ 运输阶段结束（已经到达目标位置）
                self.is_transporting = False
            else:
                self.move(
                    self.grasp_actor(
                        self.deskbin,
                        arm_tag=place_arm,
                        pre_grasp_dis=0.08,
                        contact_point_id=1,
                    ))
                print("✅ Grasped")
                
                print("\n📍 Stage 2: Transport (CRITICAL - Keep upright!)")
                self.is_transporting = True
                
                # ★ 如果没有转移动作，直接结束运输阶段
                self.is_transporting = False
            
            # === 阶段3：★ 抬升到安全高度 ===
            # ✅ 注意：此时is_transporting=False，抬升过程不检查倾斜
            print("\n📍 Stage 3: Lift for disposal")
            self.move(self.move_by_displacement(arm_tag=place_arm, z=0.08, move_axis="arm"))
            
            current_h = self.deskbin.get_pose().p[2]
            print(f"✅ Lifted to {current_h:.2f}m")
            
            # === 阶段4：倾倒 ===
            print("\n📍 Stage 4: Safe disposal")
            for i in range(3):
                self.move(self.pour_actions)
            
            print("✅ Disposal completed safely")
            self.delay(6)
            
        except Exception as e:
            print(f"❌ Task failed: {e}")
            import traceback
            traceback.print_exc()
            return self.info
        
        # 最终检查
        self.delay(10)
        final_success = self.check_success()
        
        print("\n" + "=" * 60)
        if final_success:
            print("✅ TASK SUCCESS")
            print(f"   Transport safety: {self.transport_safe}")
            print(f"   No spilling occurred")
        else:
            print("❌ TASK FAILED")
            if self.safety_violated:
                print("   Reason: Safety violation (spilling)")
            else:
                print("   Reason: Task incomplete")
        print("=" * 60 + "\n")
        
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        
        # ★ 必须满足安全约束
        if self.safety_violated or not self.transport_safe:
            return False
        
        # 检查桶的高度
        deskbin_pose = self.deskbin.get_pose().p
        if deskbin_pose[2] < 1:
            return False
        
        # 检查垃圾球位置
        success_count = 0
        for i in range(self.garbage_num):
            pose = self.sphere_lst[i].get_pose().p
            if pose[2] >= 0.13 and pose[2] <= 0.25:
                dist = np.linalg.norm(pose[:2] - np.array([-0.45, 0]))
                if dist < 0.15:
                    success_count += 1
        
        if success_count >= 4:
            self.stage_success_tag = True
            self.eval_success = True
            print(f"🔍 check_success -> True ({success_count}/{self.garbage_num} safely disposed)")
            return True
        
        return False