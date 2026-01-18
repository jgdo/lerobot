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
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from enum import IntEnum
from typing import Any

import numpy as np
import pygame

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadEgoTeleopConfig, GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn’t reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass

class GamepadEgoTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadEgoTeleopConfig
    name = "gamepad_ego"

    def __init__(self, config: GamepadEgoTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": {"enabled": 0, 
                      "delta_gripper": 1, 
                      "delta_forward": 2, 
                      "delta_rot_base": 3,
                      "delta_tilt_front": 4,
                      "delta_gripper_twist": 5,
                      "delta_up": 6,
                      },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn’t reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        self.gamepad.update()
        action_dict = self.get_controls()

        return action_dict

    def get_controls(self) -> dict[str, float]:
        """Get the current movement deltas from gamepad state."""

        joystick = self.gamepad.joystick
        # buttons = {i: joystick.get_button(i) for i in range(joystick.get_numbuttons())}
        # print(" buttons: ", buttons)
        # axes = {i: joystick.get_axis(i) for i in range(joystick.get_numaxes())}
        # print(" axes: ", axes)
        deadzone = self.config.deadzone
        pos_step_size = 0.0015
        rot_step_size = 0.012

        delta_forward = 0.0
        delta_rot_base = 0.0
        delta_tilt_front = 0.0
        delta_gripper = 0.0
        delta_gripper_twist = 0.0
        delta_up = 0.0
        enabled = False

        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            delta_forward = -joystick.get_axis(1)  # Up/Down (often inverted)
            delta_gripper_twist = joystick.get_axis(0)

            # Right stick Y (typically axis 3 or 4)
            delta_up = joystick.get_axis(4)  # Up/Down for Z
            delta_rot_base = -joystick.get_axis(3)  # Left/Right for rotation
            delta_tilt_front = joystick.get_button(13) -joystick.get_button(14)  # D-pad Up/Down for tilt

            # Apply deadzone to avoid drift
            delta_forward = 0 if abs(delta_forward) < deadzone else delta_forward
            delta_gripper_twist = 0 if abs(delta_gripper_twist) < deadzone else delta_gripper_twist
            delta_up = 0 if abs(delta_up) < deadzone else delta_up
            delta_rot_base = 0 if abs(delta_rot_base) < deadzone else delta_rot_base
            delta_tilt_front = 0 if abs(delta_tilt_front) < deadzone else delta_tilt_front
            delta_gripper  = joystick.get_button(4) - joystick.get_button(5) # LB - RB for gripper open/close

            enabled = (
                abs(delta_forward) > 0 
                or abs(delta_rot_base) > 0 
                or abs(delta_up) > 0 
                or abs(delta_gripper_twist) > 0
                or abs(delta_tilt_front) > 0
                or abs(delta_gripper) > 0
            )
            
        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            delta_forward = 0.0
            delta_rot_base = 0.0
            delta_tilt_front = 0.0
            delta_gripper_twist = 0.0
            delta_up = 0.0
            delta_gripper = 0.0
            enabled = False
        
        action_dict = {
                "delta_forward": delta_forward * pos_step_size,
                "delta_rot_base": delta_rot_base * rot_step_size,
                "delta_tilt_front": delta_tilt_front * rot_step_size,
                "delta_gripper_twist": delta_gripper_twist * rot_step_size * 2,
                "delta_gripper": delta_gripper,
                "delta_up": delta_up * pos_step_size,
                "enabled": enabled,
            }
        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        print("GamepadEgoTeleop get_teleop_events called")
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        print("GamepadEgoTeleop get_teleop_events: is_intervention=", is_intervention,
              ", terminate_episode=", terminate_episode,
              ", success=", success,
              ", rerecord_episode=", rerecord_episode)

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
