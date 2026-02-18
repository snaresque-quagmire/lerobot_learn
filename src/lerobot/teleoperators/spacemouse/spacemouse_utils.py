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

PYSPACEMOUSE_AVAILABLE = True
try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None
    PYSPACEMOUSE_AVAILABLE = False
    logging.warning(
        "pyspacemouse is not installed. Install it with: pip install pyspacemouse "
        "or install lerobot with the spacemouse extra: pip install 'lerobot[spacemouse]'"
    )


class SpaceMouseController:
    """
    Wrapper around pyspacemouse providing a consistent interface for the SpaceMouseTeleop class.

    Handles opening/closing the device, reading state, applying deadzone filtering,
    and extracting button states.

    Compatible with pyspacemouse >= 2.0 (device-object API).
    """

    def __init__(self, deadzone: float = 0.02, translation_sensitivity: float = 1.0, rotation_sensitivity: float = 1.0):
        """
        Initialize the SpaceMouse controller.

        Args:
            deadzone: Deadzone threshold (0.0 to 1.0). Axis values below this
                      magnitude are treated as zero.
            translation_sensitivity: Scale factor applied to translation axes.
            rotation_sensitivity: Scale factor applied to rotation axes.
        """
        if not PYSPACEMOUSE_AVAILABLE:
            raise ImportError(
                "pyspacemouse is required for SpaceMouse teleoperation. "
                "Install it with: pip install pyspacemouse "
                "or install lerobot with the spacemouse extra: pip install 'lerobot[spacemouse]'"
            )

        self.deadzone = deadzone
        self.translation_sensitivity = translation_sensitivity
        self.rotation_sensitivity = rotation_sensitivity
        self._device = None
        self._state = None

    def start(self) -> None:
        """Open the SpaceMouse device."""
        self._device = pyspacemouse.open()
        if self._device is None:
            raise ConnectionError(
                "Failed to open SpaceMouse device. Make sure it is connected "
                "and not being used by another application."
            )
        self._is_open = True
        logging.info("SpaceMouse connected successfully.")

    def stop(self) -> None:
        """Close the SpaceMouse device and release resources."""
        if self._device is not None:
            self._device.close()
            self._device = None
            logging.info("SpaceMouse disconnected.")

    @property
    def is_open(self) -> bool:
        return self._device is not None

    def update(self) -> None:
        """Read the latest state from the SpaceMouse device.

        Drains all buffered HID reports so we always use the freshest data.
        Without this, stale reports queue up in the USB buffer (especially
        when the knob springs back to center) causing delayed response.
        """
        if self._device is None:
            return
        # Read once to get current state
        state = self._device.read()
        # Drain any remaining buffered reports — keep reading until
        # the timestamp stops advancing (no more queued reports).
        while True:
            next_state = self._device.read()
            if next_state.t == state.t:
                break
            state = next_state
        self._state = state

    def _apply_deadzone(self, value: float) -> float:
        """Zero out values below the deadzone threshold."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_deltas(self) -> tuple[float, float, float]:
        """
        Get the current translation deltas from the SpaceMouse.

        Returns:
            Tuple of (delta_x, delta_y, delta_z) scaled by sensitivity.
            Values are in the range [-1.0, 1.0] * sensitivity.
        """
        if self._state is None:
            return (0.0, 0.0, 0.0)

        dx = self._apply_deadzone(self._state.x) * self.translation_sensitivity
        dy = self._apply_deadzone(self._state.y) * self.translation_sensitivity
        dz = self._apply_deadzone(self._state.z) * self.translation_sensitivity

        return (dx, dy, dz)

    def get_rotation_deltas(self) -> tuple[float, float, float]:
        """
        Get the current rotation deltas from the SpaceMouse.

        Returns:
            Tuple of (roll, pitch, yaw) scaled by rotation sensitivity.
            Values are in the range [-1.0, 1.0] * rotation_sensitivity.
        """
        if self._state is None:
            return (0.0, 0.0, 0.0)

        roll = self._apply_deadzone(self._state.roll) * self.rotation_sensitivity
        pitch = self._apply_deadzone(self._state.pitch) * self.rotation_sensitivity
        yaw = self._apply_deadzone(self._state.yaw) * self.rotation_sensitivity

        return (roll, pitch, yaw)

    def get_button_state(self, button_index: int) -> bool:
        """
        Check whether a specific button is currently pressed.

        Args:
            button_index: Zero-based index of the button to check.

        Returns:
            True if the button is pressed, False otherwise.
        """
        if self._state is None:
            return False

        buttons = self._state.buttons
        if button_index < len(buttons):
            return bool(buttons[button_index])
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


