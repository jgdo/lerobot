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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("robot_ego_kinematic_processor")
@dataclass
class RobotEgoKinematicProcessor(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: tuple[np.ndarray, float] | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            # print(observation)
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        delta_forward = float(action.pop("delta_forward"))
        delta_rot_base = float(action.pop("delta_rot_base"))
        delta_tilt_front = float(action.pop("delta_tilt_front"))
        delta_gripper_twist = float(action.pop("delta_gripper_twist"))
        delta_gripper = float(action.pop("delta_gripper"))
        delta_up = float(action.pop("delta_up"))

        # desired = None

        if enabled:
            if self.use_latched_reference and self._command_when_disabled is not None:
                ref = self._command_when_disabled[0]
                gripper_ref = self._command_when_disabled[1]
                if not np.isclose(ref, t_curr, atol=0.2, rtol=0.5).all():
                    print("############################################")
                    print("ref: ", ref)
                    print("t_curr: ", t_curr)
                    ref = t_curr
                    self._command_when_disabled = (ref.copy(), gripper_ref)

            else:
                ref = t_curr
                gripper_ref = q_raw[-1]
        
            delta = np.eye(4, dtype=float)
            delta[0, 3] = delta_up
            delta[2, 3] = delta_forward
            delta[:3, :3] = Rotation.from_rotvec([0, delta_tilt_front, delta_gripper_twist]).as_matrix()

            def translate(xyz):
                t = np.eye(4, dtype=float)
                t[:3, 3] = xyz
                return t
            

            rot_base =  np.eye(4, dtype=float)
            rot_base[:3, :3] = Rotation.from_rotvec([0, 0, delta_rot_base]).as_matrix()
            displacement = translate([0.0388353, 0, 0])
            rot_base =  displacement @ rot_base @ np.linalg.inv(displacement)

            desired = rot_base @ ref @ delta
            desired_gripper = gripper_ref + delta_gripper

            # print("desired: ", desired)

            self._command_when_disabled = (desired.copy(), desired_gripper)
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = (t_curr.copy(), q_raw[-1])
            desired = self._command_when_disabled[0]
            desired_gripper = self._command_when_disabled[1]

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_pos"] = desired_gripper

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "delta_forward",
            "delta_rot_base",
            "delta_tilt_front",
            "delta_gripper_twist",
            "delta_gripper",
            "delta_up",
            "enabled"
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
