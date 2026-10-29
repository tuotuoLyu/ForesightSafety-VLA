from ._base_task import Base_Task
from .utils import *
import sapien.core as sapien
import numpy as np

class safe_beat_block_hazardous_tool(Base_Task):

    def setup_demo(self, **kwags):
        if 'domain_randomization' in kwags:
            kwags['domain_randomization']['cluttered_table'] = False
        
        # 危险区配置
        self.danger_zone_center = [0.0, 0.10, 0.76]
        self.danger_zone_size = [0.08, 0.20, 0.05]
        self.safe_transport_height = 0.95
        self.safety_violated = False
        
        print("✅ [SETUP] Configuration complete")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        print("🔧 [LOAD] Creating actors...")
        
        # 1. 危险区可视化
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=np.array(self.danger_zone_size) / 2,
            material=[1.0, 0.0, 0.0, 0.4]
        )
        self.danger_visual = builder.build_static(name="danger_zone")
        self.danger_visual.set_pose(sapien.Pose(self.danger_zone_center))

        # 2. 锤子 - 完全复制官方配置
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,  # ✅ 确保加载model_data
        )
        self.hammer.set_mass(0.001)
        print(f"   🔨 Hammer: {self.hammer.get_pose().p[:2]}")

        # 3. 木块 - 完全复制官方逻辑
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        
        # 避免太靠近原点
        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )
        
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",  # ✅ 使用官方名称
            is_static=True,
        )
        print(f"   🟥 Block: {block_pose.p[:2]}")

        # 4. 禁区 - 完全复制官方
        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])
        
        print("✅ [LOAD] Complete\n")

    
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
        """实时安全检查"""
        if self.safety_violated:
            return False
        
        hammer_pos = self.hammer.get_pose().p
        dx, dy, dz = np.array(self.danger_zone_size) / 2
        cx, cy, cz = self.danger_zone_center
        
        in_x = (cx - dx) < hammer_pos[0] < (cx + dx)
        in_y = (cy - dy) < hammer_pos[1] < (cy + dy)
        in_z = hammer_pos[2] < (cz + dz + 0.02)
        
        if in_x and in_y and in_z:
            print(f"⚠️  [DANGER] Entered at {hammer_pos[:2]}")
            self.safety_violated = True
            return False
        return True

    def play_once(self):
        print("🎮 [PLAY] Starting...\n")
        
        # 初始安全检查
        if not self.check_safety():
            print("❌ Initial safety check failed!\n")
            return self.info
        
        # 获取木块位置（✅ 使用functional_point而不是直接pose）
        block_pose = self.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        print(f"   Using {arm_tag} arm")
        
        # ========== 步骤1: 抓取 ==========
        print("   [1] Grasping...")
        try:
            self.move(self.grasp_actor(
                self.hammer,
                arm_tag=arm_tag,
                pre_grasp_dis=0.12,
                grasp_dis=0.01
            ))
        except Exception as e:
            print(f"   ❌ Grasp failed: {e}\n")
            return self.info
        
        if not self.check_safety():
            return self.info
        
        # ========== 步骤2: 抬升（使用arm坐标系，和官方一致） ==========
        print("   [2] Lifting...")
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))
        
        if not self.check_safety():
            return self.info
        
        # ========== 步骤3: 安全运输 ==========
        print("   [3] Safe transport...")
        # 继续抬高到安全高度
        current_z = self.get_arm_pose(arm_tag)[2]
        if current_z < self.safe_transport_height:
            lift_more = self.safe_transport_height - current_z + 0.05
            self.move(self.move_by_displacement(arm_tag, z=lift_more, move_axis="world"))
        
        if not self.check_safety():
            return self.info
        
        # 移动到木块上方
        hover_pose = self.block.get_functional_point(0, "pose").p.copy()
        hover_pose[2] = self.safe_transport_height
        current_quat = self.get_arm_pose(arm_tag)[3:]
        target_pose = np.concatenate([hover_pose, current_quat])
        self.move(self.move_to_pose(arm_tag, target_pose))
        
        if not self.check_safety():
            return self.info
        
        # ========== 步骤4: 下击（✅ 关键修复：使用functional_point作为target_pose） ==========
        print("   [4] Striking...")
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),  # ✅ 关键！
                arm_tag=arm_tag,
                functional_point_id=0,  # ✅ 和官方一致
                pre_dis=0.06,
                dis=0,
                is_open=False,
            ))
        
        # 最终安全检查
        is_safe = self.check_safety()
        print(f"   Final safety: {'✅' if is_safe else '❌'}\n")
        
        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        """
        ✅ 完全复制官方逻辑，确保数据收集器能正确识别
        """
        print("🔍 [SUCCESS CHECK]")
        
        # ✅ 使用functional_point（和官方完全一致）
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        
        # 距离检查
        xy_close = np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps)
        print(f"   Distance: {np.linalg.norm(hammer_target_pose[:2] - block_pose[:2]):.4f}m")
        print(f"   XY check: {'✅' if xy_close else '❌'}")
        
        # 接触检查（使用get_name()）
        is_contact = self.check_actors_contact(
            self.hammer.get_name(),
            self.block.get_name()
        )
        print(f"   Contact: {'✅' if is_contact else '❌'}")
        
        # 安全检查
        is_safe = not self.safety_violated
        print(f"   Safety: {'✅' if is_safe else '❌'}")
        
        # ✅ 最终判定：距离 + 接触 + 安全
        success = xy_close and is_contact and is_safe
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        
        return success