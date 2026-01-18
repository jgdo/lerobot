# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specif

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.ego_kinematic_processor import RobotEgoKinematicProcessor
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEgoTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEgoTeleop
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadEgoTeleop
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadEgoTeleopConfig
from lerobot.teleoperators.so101_poti.config_so101_poti import SO101PotiConfig
from lerobot.teleoperators.so101_poti.so101_poti import SO101Poti

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def create_teleop_keyboard():
    teleop_config = KeyboardEgoTeleopConfig()
    teleop_device = KeyboardEgoTeleop(teleop_config)
    return teleop_device

def create_teleop_gamepad():
    teleop_config = GamepadEgoTeleopConfig()
    teleop_device = GamepadEgoTeleop(teleop_config)
    return teleop_device

def create_teleop_so101_poti(motor_names: list[str]):
    teleop_config = SO101PotiConfig(
        port="/dev/ttyACM0",
        motor_names=motor_names,
    )
    teleop_device = SO101Poti(teleop_config)
    return teleop_device

def main():
    # Initialize the robot and teleoperator
    robot_config = SO101FollowerConfig(
        port="/dev/ttyUSB0", id="my_lerobot", use_degrees=False
    )

    # Initialize the robot and teleoperator
    robot = SO101Follower(robot_config)
    teleop_device = create_teleop_keyboard()
    # teleop_device = create_teleop_gamepad()
    # teleop_device = create_teleop_so101_poti(motor_names=list(robot.bus.motors.keys()))

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="/home/jgdo/dev/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            RobotEgoKinematicProcessor(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            # ),
            # EEBoundsAndSafety(
            #     end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            #     max_ee_step_m=0.10,
            # ),
            # GripperVelocityToJoint(
            #      speed_factor=20.0,
            # ),
            InverseKinematicsEEToJoints(
                  kinematics=kinematics_solver,
                  motor_names=list(robot.bus.motors.keys()),
                  initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    forward_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            InverseKinematicsEEToJoints(
                 kinematics=kinematics_solver,
                 motor_names=list(robot.bus.motors.keys()),
                 initial_guess_current_joints=True,
            ),
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer
    init_rerun(session_name="my_lerobot_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")
    
    robot.bus.disable_torque()

    print("Starting teleop loop. Move your phone to teleoperate the robot...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = robot.get_observation()
        print("robot obs: ", robot_obs)

        # Get teleop action
        teleop_action = teleop_device.get_action()
        #print("teleop: ", teleop_action)

        # Phone -> EE pose -> Joints transition
        joint_action = phone_to_robot_joints_processor((teleop_action, robot_obs))
        print("joint_action: ", joint_action)

        #eef = forward_pipeline((joint_action, robot_obs))
        #print("iv f: ", eef)

        # Send action to robot
        _ = robot.send_action(joint_action)

        # Visualize
        log_rerun_data(observation=teleop_action, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
