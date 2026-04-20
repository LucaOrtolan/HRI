import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import yaml


class RobotArmSimulation:
    def __init__(self, cfg_path='cfg.yml', use_gui=True):
        # Load simulation config from YAML
        self.cfg_path = Path(cfg_path)
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        # Target stacking position read from cfg.yml
        self.place_pos = [
            float(self.cfg['place_coords']['x']),
            float(self.cfg['place_coords']['y']),
            float(self.cfg['place_coords'].get('z', 0.02)),
        ]

        # Cube geometry and stacking state
        self.cube_half_extent = float(self.cfg.get('cube_half_extent', 0.02))
        self.cube_size = 2.0 * self.cube_half_extent
        self.stack_count = 0
        self.stack_order = []

        # Initial cube spawn positions
        self.cube_positions = self.cfg.get('cube_positions', {
            'red': [0.55, -0.12, self.cube_half_extent],
            'green': [0.62, 0.00, self.cube_half_extent],
            'blue': [0.55, 0.12, self.cube_half_extent],
        })

        # Connect to PyBullet and configure physics
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Build world and set initial state
        self._build_scene()
        self.busy = False

    def _build_scene(self):
        # Ground plane and friction tuning
        self.plane = p.loadURDF('plane.urdf')
        p.changeDynamics(self.plane, -1, lateralFriction=1.2, spinningFriction=0.001, rollingFriction=0.001)

        # Add table and robot arm
        self.table = p.loadURDF('table/table.urdf', [0.5, 0.0, -0.65], useFixedBase=True)
        self.robot = p.loadURDF('franka_panda/panda.urdf', [0.0, 0.0, 0.0], useFixedBase=True)
        p.resetDebugVisualizerCamera(1.3, 40, -35, [0.45, 0.0, 0.1])

        # Panda joint setup
        self.arm_joints = list(range(7))
        self.finger_joints = [9, 10]
        self.ee_link = 11
        self.home_joints = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8]
        self.ee_down_orn = p.getQuaternionFromEuler([3.14159, 0, 0])

        # IK tuning values for a more stable solve
        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.joint_ranges = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]
        self.rest_poses = self.home_joints

        # Reset arm to a comfortable starting pose
        for idx, q in zip(self.arm_joints, self.home_joints):
            p.resetJointState(self.robot, idx, q)

        # Spawn cubes and open gripper
        self.cubes = {}
        self._spawn_cubes()
        self.open_gripper(step_sim=False)
        self.step_for(0.5)

    def _spawn_cubes(self):
        # RGB colors for the three cubes
        rgba = {
            'red': [1, 0, 0, 1],
            'green': [0, 1, 0, 1],
            'blue': [0, 0, 1, 1],
        }

        # Shared collision geometry for all cubes
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half_extent, self.cube_half_extent, self.cube_half_extent],
        )

        # Create each cube as a small rigid body and tune contact behavior
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
        # Advance the simulation for a fixed amount of time
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
            if p.isConnected(self.client) == 0:
                break
            time.sleep(1.0 / 240.0)

    def get_current_joint_positions(self):
        # Read the current arm joint state
        return [p.getJointState(self.robot, j)[0] for j in self.arm_joints]

    def solve_ik(self, target_pos, target_orn=None):
        # Solve inverse kinematics for the target pose
        if target_orn is None:
            target_orn = self.ee_down_orn

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
        return list(joint_targets[:len(self.arm_joints)])

    @staticmethod
    def minimum_jerk_blend(tau):
        # Quintic minimum-jerk blend: zero vel/acc at endpoints
        tau = np.clip(tau, 0.0, 1.0)
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    def execute_joint_trajectory(self, q_start, q_goal, duration=2.0, dt=1.0 / 240.0):
        # Follow a smooth joint-space trajectory from q_start to q_goal
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        steps = max(2, int(duration / dt))

        for k in range(steps + 1):
            tau = k / steps
            s = self.minimum_jerk_blend(tau)
            q_cmd = q_start + s * (q_goal - q_start)

            # Send the interpolated joint targets to PyBullet
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

    def move_to_pose_smooth(self, target_pos, target_orn=None, duration=2.0):
        # Solve IK for the target pose and execute a smooth trajectory
        q_start = self.get_current_joint_positions()
        q_goal = self.solve_ik(target_pos, target_orn)
        self.execute_joint_trajectory(q_start, q_goal, duration=duration)

    def open_gripper(self, step_sim=True):
        # Open the Panda gripper
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
        # Close the Panda gripper to grasp an object
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
        # Create a fixed constraint between the end-effector and the selected cube
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
        # Remove the grasp constraint after placement
        if hasattr(self, 'constraint_id'):
            p.removeConstraint(self.constraint_id)
            del self.constraint_id

    def move_home(self, duration=2.0):
        # Return the arm to its home pose smoothly
        q_start = self.get_current_joint_positions()
        self.execute_joint_trajectory(q_start, self.home_joints, duration=duration)

    def get_stack_place_pose(self):
        # Compute the next stacking position using the current stack height
        return [
            self.place_pos[0],
            self.place_pos[1],
            self.place_pos[2] + self.stack_count * self.cube_size,
        ]

    def pick_and_place(self, color):
        # Ignore invalid colors or cubes that have already been stacked
        if color not in self.cubes:
            return
        if color in self.stack_order:
            return

        self.busy = True

        # Get current source cube position and target stack position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cubes[color])
        stack_place = self.get_stack_place_pose()

        # Approach the source cube from above
        hover_pick = [cube_pos[0], cube_pos[1], 0.28]
        grasp = [cube_pos[0], cube_pos[1], self.cube_half_extent + 0.02]

        # Approach the stacking location from above
        hover_place = [stack_place[0], stack_place[1], stack_place[2] + 0.20]
        place = [stack_place[0], stack_place[1], stack_place[2] + self.cube_half_extent + 0.02]

        # Execute grasp sequence
        self.open_gripper()
        self.move_to_pose_smooth(hover_pick, duration=2.0)
        self.move_to_pose_smooth(grasp, duration=1.4)
        self.close_gripper()
        self.attach_cube(color)
        self.step_for(0.2)

        # Move the cube to the stack location
        self.move_to_pose_smooth(hover_pick, duration=1.4)
        self.move_to_pose_smooth(hover_place, duration=2.2)
        self.move_to_pose_smooth(place, duration=1.8)

        # Release the cube and lock it into the new stack position
        self.detach_cube()
        p.resetBasePositionAndOrientation(self.cubes[color], stack_place, [0, 0, 0, 1])
        self.open_gripper()
        self.step_for(0.4)

        # Update stacking state
        self.stack_order.append(color)
        self.stack_count += 1

        # Retract and return home
        self.move_to_pose_smooth(hover_place, duration=1.4)
        self.move_home(duration=2.2)
        self.busy = False

    def close(self):
        # Cleanly disconnect from PyBullet
        if p.isConnected(self.client):
            p.disconnect(self.client)