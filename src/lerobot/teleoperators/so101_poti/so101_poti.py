#!/usr/bin/env python

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
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import serial
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_poti import SO101PotiConfig

import numpy as np

logger = logging.getLogger(__name__)


class SO101Poti(Teleoperator):
    """
    SO-101 Poti Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101PotiConfig
    name = "so101_poti"

    def __init__(self, config: SO101PotiConfig):
        super().__init__(config)
        self.config = config
        self.serial = serial.Serial()
        self.serial.port = self.config.port
        self.serial.baudrate = 115200
        assert self.config.motor_names is not None, "motor_names must be provided in SO101PotiConfig"
        assert len(self.config.motor_names) == 6, "motor_names must contain exactly 6 motor names"
        

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.serial.is_open

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.serial.open()

        # self.bus.connect()
        # if not self.is_calibrated and calibrate:
        #     logger.info(
        #         "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
        #     )
        #     self.calibrate()

        # self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # return self.bus.is_calibrated
        return True

    def calibrate(self) -> None:
        pass
        # if self.calibration:
        #     # Calibration file exists, ask user whether to use it or run new calibration
        #     user_input = input(
        #         f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
        #     )
        #     if user_input.strip().lower() != "c":
        #         logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
        #         self.bus.write_calibration(self.calibration)
        #         return

        # logger.info(f"\nRunning calibration of {self}")
        # self.bus.disable_torque()
        # for motor in self.bus.motors:
        #     self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # input(f"Move {self} to the middle of its range of motion and press ENTER....")
        # homing_offsets = self.bus.set_half_turn_homings()

        # print(
        #     "Move all joints sequentially through their entire ranges "
        #     "of motion.\nRecording positions. Press ENTER to stop..."
        # )
        # range_mins, range_maxes = self.bus.record_ranges_of_motion()

        # self.calibration = {}
        # for motor, m in self.bus.motors.items():
        #     self.calibration[motor] = MotorCalibration(
        #         id=m.id,
        #         drive_mode=0,
        #         homing_offset=homing_offsets[motor],
        #         range_min=range_mins[motor],
        #         range_max=range_maxes[motor],
        #     )

        # self.bus.write_calibration(self.calibration)
        # self._save_calibration()
        # print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        pass
    #     self.bus.disable_torque()
    #     self.bus.configure_motors()
    #     for motor in self.bus.motors:
    #         self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # def setup_motors(self) -> None:
    #     for motor in reversed(self.bus.motors):
    #         input(f"Connect the controller board to the '{motor}' motor only and press enter.")
    #         self.bus.setup_motor(motor)
    #         print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        assert self.is_connected, DeviceNotConnectedError(f"{self} is not connected.")

        adc = None
        while self.serial.in_waiting or adc is None:
            line = self.serial.readline().decode('utf-8').strip().lstrip()
            if line == "":
                continue
            # print("line: ", line)
            adc = list(map(int, line.split(" ")))
            if len(adc) != 6:
                adc = None

        # print("adc: ", adc)
        
        angle_range = 180
        adc_range = 4096
        # normalize adc values
        angles = [(val / adc_range - 0.5) * angle_range for val in adc[0:5]]
        
        angles_factors = [1.4, -1.5, 1.65, -2, 1.6]
        angles = [a * f for a, f in zip(angles, angles_factors, strict=True)]
        angles[1] -= 5

        gripper = (adc[5] -1400) / 800 * 100
        gripper = 100 - np.clip(gripper, 0, 100).item()

        # print("angles: ", angles[2])

        action = {f"{self.config.motor_names[motor]}.pos": val for motor, val in enumerate(angles)}
        action["gripper.pos"] = gripper
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.serial.close()
        logger.info(f"{self} disconnected.")
