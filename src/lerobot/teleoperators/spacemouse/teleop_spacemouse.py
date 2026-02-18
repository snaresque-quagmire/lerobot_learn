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

from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_spacemouse import SpaceMouseTeleopConfig
from .spacemouse_utils import SpaceMouseController


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


class SpaceMouseTeleop(Teleoperator):
    """
    Teleop class to use a 3DConnexion SpaceMouse for control.

    Provides 3-DoF translation (delta_x, delta_y, delta_z) and optional gripper
    control via SpaceMouse buttons. Compatible with the delta-action pipeline
    used by GamepadTeleop and KeyboardEndEffectorTeleop.

    Usage:
        ```shell
        lerobot-teleoperate \\
            --robot.type=<your_robot> \\
            --robot.port=<your_port> \\
            --teleop.type=spacemouse \\
            --display_data=true
        ```
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.controller: SpaceMouseController | None = None

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
        """Connect to the SpaceMouse device."""
        self.controller = SpaceMouseController(
            deadzone=self.config.deadzone,
            translation_sensitivity=self.config.translation_sensitivity,
            rotation_sensitivity=self.config.rotation_sensitivity,
        )
        self.controller.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read the current SpaceMouse state and return a delta action dict."""
        self.controller.update()

        delta_x, delta_y, delta_z = self.controller.get_deltas()
        roll, pitch, yaw = self.controller.get_rotation_deltas()

        action_dict = {
            "delta_x": float(delta_x),
            "delta_y": float(delta_y),
            "delta_z": -float(delta_z),
            "delta_wx": -float(roll),
            "delta_wy": float(pitch),
            "delta_wz": float(yaw),
        }

        if self.config.use_gripper:
            gripper_action = GripperAction.STAY.value
            if self.controller.get_button_state(self.config.gripper_open_button):
                gripper_action = GripperAction.OPEN.value
            elif self.controller.get_button_state(self.config.gripper_close_button):
                gripper_action = GripperAction.CLOSE.value
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the SpaceMouse.

        Returns:
            Dictionary containing:
                - is_intervention: bool — True if any axis input is detected
                - terminate_episode: bool — Whether to terminate the current episode
                - success: bool — Whether the episode was successful
                - rerecord_episode: bool — Whether to rerecord the episode
        """
        if self.controller is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        self.controller.update()

        # Consider intervention active if any translation axis has non-zero input
        delta_x, delta_y, delta_z = self.controller.get_deltas()
        is_intervention = any(abs(d) > 0 for d in [delta_x, delta_y, delta_z])

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def disconnect(self) -> None:
        """Disconnect from the SpaceMouse."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    @property
    def is_connected(self) -> bool:
        """Check if the SpaceMouse is connected."""
        return self.controller is not None and self.controller.is_open

    def calibrate(self) -> None:
        """Calibrate the SpaceMouse (no-op, not required)."""
        pass

    def is_calibrated(self) -> bool:
        """SpaceMouse does not require calibration."""
        return True

    def configure(self) -> None:
        """Configure the SpaceMouse (no-op)."""
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the SpaceMouse (not supported)."""
        pass
    