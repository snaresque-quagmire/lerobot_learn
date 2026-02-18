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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    # Whether to include gripper action in the output
    use_gripper: bool = True
    # Scale factor for translation axes
    translation_sensitivity: float = 1.0
    # Scale factor for rotation axes
    rotation_sensitivity: float = 1.0
    # Deadzone for filtering noise (fraction of max, 0.0 to 1.0)
    deadzone: float = 0.05
    # Button index for gripper open
    gripper_open_button: int = 0
    # Button index for gripper close
    gripper_close_button: int = 1
