import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import yaml


class RobotArmSimulation:
    def __init__(self, cfg_path='cfg.yml', use_gui=True):
        self.cfg_path = Path(cfg_path)
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.place_pos = [
            float(self.cfg['place_coords']['x']),
            float(self.cfg['place_coords']['y']),
            float(self.cfg['place_coords'].get('z', 0.02)),
        ]

        self.cube_half_extent = float(self.cfg.get('cube_half_extent', 0.02))
        self.cube_size = 2.0 * self.cube_half_extent
        self.stack_count = 0
        self.stack_order = []

        self.cube_positions = self.cfg.get('cube_positions', {
            'red': [0.55, -0.12, self.cube_half_extent],
            'green': [0.62, 0.00, self.cube_half_extent],
            'blue': [0.55, 0.12, self.cube_half_extent],
        })

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self._build_scene()
        self.busy = False

    @staticmethod
    def _time_ns():
        return time.perf_counter_ns()

    def _build_scene(self):
        self.plane = p.loadURDF('plane.urdf')
        p.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.001,
            rollingFriction=0.001
        )

        self.table = p.loadURDF('table/table.urdf', [0.5, 0.0, -0.65], useFixedBase=True)
        self.robot = p.loadURDF('franka_panda/panda.urdf', [0.0, 0.0, 0.0], useFixedBase=True)
        p.resetDebugVisualizerCamera(1.3, 40, -35, [0.45, 0.0, 0.1])

        self.arm_joints = list(range(7))
        self.finger_joints = [9, 10]
        self.ee_link = 11
        self.home_joints = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8]
        self.ee_down_orn = p.getQuaternionFromEuler([3.14159, 0, 0])

        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.joint_ranges = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]
        self.rest_poses = self.home_joints

        for idx, q in zip(self.arm_joints, self.home_joints):
            p.resetJointState(self.robot, idx, q)

        self.cubes = {}
        self._spawn_cubes()
        self.open_gripper(step_sim=False)
        self.step_for(0.5)

    def _spawn_cubes(self):
        rgba = {
            'red': [1, 0, 0, 1],
            'green': [0, 1, 0, 1],
            'blue': [0, 0, 1, 1],
        }

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half_extent, self.cube_half_extent, self.cube_half_extent],
        )

        for color, pos in self.cube_positions.items():
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.cube_half_extent, self.cube_half_extent, self.cube_half_extent],
                rgbaColor=rgba[color],
            )
            body = p.createMultiBody(
                baseMass=0.04,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos,
            )
            p.changeDynamics(
                body,
                -1,
                lateralFriction=1.2,
                spinningFriction=0.001,
                rollingFriction=0.001,
                restitution=0.0,
            )
            self.cubes[color] = body

    def step_for(self, seconds):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
            if p.isConnected(self.client) == 0:
                break
            time.sleep(1.0 / 240.0)

    def get_current_joint_positions(self):
        return [p.getJointState(self.robot, j)[0] for j in self.arm_joints]

    def solve_ik(self, target_pos, target_orn=None):
        if target_orn is None:
            target_orn = self.ee_down_orn

        t0 = self._time_ns()

        joint_targets = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
            residualThreshold=1e-4,
            maxNumIterations=100,
        )

        t1 = self._time_ns()
        ik_solve_time_ms = (t1 - t0) / 1_000_000.0

        return list(joint_targets[:len(self.arm_joints)]), ik_solve_time_ms

    @staticmethod
    def minimum_jerk_blend(tau):
        tau = np.clip(tau, 0.0, 1.0)
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    def execute_joint_trajectory(self, q_start, q_goal, duration=2.0, dt=1.0 / 240.0):
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        steps = max(2, int(duration / dt))

        t0 = self._time_ns()

        for k in range(steps + 1):
            tau = k / steps
            s = self.minimum_jerk_blend(tau)
            q_cmd = q_start + s * (q_goal - q_start)

            for joint_id, q in zip(self.arm_joints, q_cmd):
                p.setJointMotorControl2(
                    self.robot,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=float(q),
                    force=240,
                )

            p.stepSimulation()
            time.sleep(dt)

        t1 = self._time_ns()
        trajectory_exec_time_s = (t1 - t0) / 1_000_000_000.0
        return trajectory_exec_time_s

    def move_to_pose_smooth(self, target_pos, target_orn=None, duration=2.0):
        q_start = self.get_current_joint_positions()
        q_goal, ik_solve_ms = self.solve_ik(target_pos, target_orn)
        traj_exec_s = self.execute_joint_trajectory(q_start, q_goal, duration=duration)
        return ik_solve_ms, traj_exec_s

    def open_gripper(self, step_sim=True):
        for _ in range(120):
            for j in self.finger_joints:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=0.04,
                    force=80,
                )
            if step_sim:
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

    def close_gripper(self):
        for _ in range(120):
            for j in self.finger_joints:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=140,
                )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def attach_cube(self, color):
        cube_id = self.cubes[color]
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.10],
            childFramePosition=[0, 0, 0],
        )

    def detach_cube(self):
        if hasattr(self, 'constraint_id'):
            p.removeConstraint(self.constraint_id)
            del self.constraint_id

    def move_home(self, duration=2.0):
        q_start = self.get_current_joint_positions()
        home_exec_s = self.execute_joint_trajectory(q_start, self.home_joints, duration=duration)
        return home_exec_s

    def get_stack_place_pose(self):
        return [
            self.place_pos[0],
            self.place_pos[1],
            self.place_pos[2] + self.stack_count * self.cube_size,
        ]

    def pick_and_place(self, color):
        if color not in self.cubes:
            return
        if color in self.stack_order:
            return

        self.busy = True

        planning_latencies_ms = []
        total_trajectory_s = 0.0

        cube_pos, _ = p.getBasePositionAndOrientation(self.cubes[color])
        stack_place = self.get_stack_place_pose()

        hover_pick = [cube_pos[0], cube_pos[1], 0.28]
        grasp = [cube_pos[0], cube_pos[1], self.cube_half_extent + 0.02]

        hover_place = [stack_place[0], stack_place[1], stack_place[2] + 0.20]
        place = [stack_place[0], stack_place[1], stack_place[2] + self.cube_half_extent + 0.02]

        self.open_gripper()

        ik_ms, traj_s = self.move_to_pose_smooth(hover_pick, duration=2.0)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        ik_ms, traj_s = self.move_to_pose_smooth(grasp, duration=1.4)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        self.close_gripper()
        self.attach_cube(color)
        self.step_for(0.2)

        ik_ms, traj_s = self.move_to_pose_smooth(hover_pick, duration=1.4)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        ik_ms, traj_s = self.move_to_pose_smooth(hover_place, duration=2.2)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        ik_ms, traj_s = self.move_to_pose_smooth(place, duration=1.8)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        self.detach_cube()
        p.resetBasePositionAndOrientation(self.cubes[color], stack_place, [0, 0, 0, 1])
        self.open_gripper()
        self.step_for(0.4)

        self.stack_order.append(color)
        self.stack_count += 1

        ik_ms, traj_s = self.move_to_pose_smooth(hover_place, duration=1.4)
        planning_latencies_ms.append(ik_ms)
        total_trajectory_s += traj_s

        home_exec_s = self.move_home(duration=2.2)
        total_trajectory_s += home_exec_s

        self.busy = False

        mean_planning_ms = (
            sum(planning_latencies_ms) / len(planning_latencies_ms)
            if planning_latencies_ms else 0.0
        )

        print(f"[Task Complete] Mean planning latency: {mean_planning_ms:.6f} ms")
        print(f"[Task Complete] Total trajectory execution time: {total_trajectory_s:.6f} s")

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)