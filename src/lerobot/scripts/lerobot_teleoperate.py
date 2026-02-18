# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Example (leader-follower):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --display_data=true
```

Example (SpaceMouse, direct joint control):

```shell
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=spacemouse \
    --direct_joint.step_size=2.0 \
    --debug=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    spacemouse,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DirectJointConfig:
    """Maps SpaceMouse axes directly to robot joints (no IK needed).

    SpaceMouse axes: x, y, z (translation), roll, pitch, yaw (rotation).
    Set map_<joint> to a SpaceMouse axis name, or "none" to disable.

    Available axes: x, y, z, roll, pitch, yaw, none

    Examples:
        # Enable direct joint control (pass ANY --direct_joint.* flag):
        --direct_joint.step_size=2.0

        # Test only shoulder_pan:
        --direct_joint.active_joints=shoulder_pan

        # Make shoulder slower, elbow faster:
        --direct_joint.step_shoulder_pan=1.0
        --direct_joint.step_elbow_flex=4.0

        # Swap shoulder_pan to use X axis instead of yaw:
        --direct_joint.map_shoulder_pan=x
    """

    # Global default step size (degrees per tick at full deflection)
    step_size: float = 2.0

    # Per-joint step size overrides (None = use global step_size)
    step_shoulder_pan: float | None = None
    step_shoulder_lift: float | None = None
    step_elbow_flex: float | None = None
    step_wrist_flex: float | None = None
    step_wrist_roll: float | None = None

    # Axis mapping: which SpaceMouse axis controls which joint
    map_shoulder_pan: str = "yaw"       # twist left/right → rotate base
    map_shoulder_lift: str = "y"    # tilt forward/back → raise/lower shoulder
    map_elbow_flex: str = "z"           # push up/down → bend/extend elbow
    map_wrist_flex: str = "pitch"        # tilt left/right → flex wrist
    map_wrist_roll: str = "roll"           # push left/right → roll wrist

    # Active joints filter: comma-separated (empty = all).
    # Only joints listed here will respond to the SpaceMouse.
    # Example: "shoulder_pan" or "shoulder_pan,elbow_flex"
    active_joints: str = ""

    def get_axis_map(self) -> list[tuple[str, str, float]]:
        """Returns list of (axis_name, joint_name, step_size) tuples."""
        joints = [
            ("shoulder_pan",   self.map_shoulder_pan,   self.step_shoulder_pan),
            ("shoulder_lift",  self.map_shoulder_lift,  self.step_shoulder_lift),
            ("elbow_flex",     self.map_elbow_flex,     self.step_elbow_flex),
            ("wrist_flex",     self.map_wrist_flex,     self.step_wrist_flex),
            ("wrist_roll",     self.map_wrist_roll,     self.step_wrist_roll),
        ]

        active = set()
        if self.active_joints.strip():
            active = {j.strip() for j in self.active_joints.split(",")}

        result = []
        for joint_name, axis_name, step_override in joints:
            if axis_name.lower() == "none":
                continue
            if active and joint_name not in active:
                continue
            step = step_override if step_override is not None else self.step_size
            result.append((axis_name, joint_name, step))
        return result


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    max_gripper_pos: float = 100.0
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    # Print debug info (joint positions, axes, deltas) to terminal
    debug: bool = False
    # Direct joint control (activate by passing any --direct_joint.* flag)
    direct_joint: DirectJointConfig | None = None




# ─────────────────────────────────────────────────────────────────────────────
# Direct joint control loop
# ─────────────────────────────────────────────────────────────────────────────

def direct_joint_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    djc: DirectJointConfig,
    max_gripper_pos: float = 100.0,
    duration: float | None = None,
    debug: bool = False,
):
    """Maps SpaceMouse axes directly to joint velocities.

    Each tick: joint_pos += axis_value * step_size.
    Push = move, release = hold position.
    """
    axis_map = djc.get_axis_map()

    # Log the active mapping at startup
    logging.info("Direct joint control — active axis mapping:")
    for axis_name, joint_name, step in axis_map:
        logging.info(f"  SpaceMouse {axis_name:>5s} → {joint_name:<16s}  (step={step:.1f} deg/tick)")
    if not axis_map:
        logging.warning("No joints are active! Check --direct_joint.active_joints")

    # Initialize joint positions from current robot state
    obs = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())
    joint_positions = {}
    for name in motor_names:
        key = f"{name}.pos"
        joint_positions[name] = float(obs[key]) if key in obs else 0.0

    # Read joint limits from calibration (degrees)
    joint_limits = {}
    for motor in motor_names:
        cal = robot.bus.calibration[motor]
        model = robot.bus.motors[motor].model
        max_res = robot.bus.model_resolution_table[model] - 1
        mid = (cal.range_min + cal.range_max) / 2
        min_deg = (cal.range_min - mid) * 360 / max_res
        max_deg = (cal.range_max - mid) * 360 / max_res
        joint_limits[motor] = (min(min_deg, max_deg), max(min_deg, max_deg))
        logging.info(f"  {motor:<16s} limits: [{joint_limits[motor][0]:>7.1f}, {joint_limits[motor][1]:>7.1f}] deg")

    gripper_pos = joint_positions.get("gripper", 0.0)
    start = time.perf_counter()
    col_w = max(len(n) for n in motor_names) + 2

    while True:
        loop_start = time.perf_counter()
        raw_action = teleop.get_action()
        t_read = time.perf_counter()

        # All 6 SpaceMouse axes
        axes = {
            "x":     float(raw_action.get("delta_x", 0.0)),
            "y":     float(raw_action.get("delta_y", 0.0)),
            "z":     float(raw_action.get("delta_z", 0.0)),
            "roll":  float(raw_action.get("delta_wx", 0.0)),
            "pitch": float(raw_action.get("delta_wy", 0.0)),
            "yaw":   float(raw_action.get("delta_wz", 0.0)),
        }

        # Accumulate into joint positions and clamp to limits
        for axis_name, joint_name, step in axis_map:
            val = axes.get(axis_name, 0.0)
            if joint_name in joint_positions:
                joint_positions[joint_name] += val * step
                if joint_name in joint_limits:
                    lo, hi = joint_limits[joint_name]
                    joint_positions[joint_name] = max(lo, min(hi, joint_positions[joint_name]))

        # Gripper via buttons
        gripper_action = raw_action.get("gripper", 1)
        if gripper_action == 2:    # OPEN
            gripper_pos = min(gripper_pos + 2.0, max_gripper_pos)
        elif gripper_action == 0:  # CLOSE
            gripper_pos = max(gripper_pos - 2.0, 0.0)
        joint_positions["gripper"] = gripper_pos

        # Send to robot
        action = {f"{name}.pos": val for name, val in joint_positions.items()}
        robot.send_action(action)
        t_send = time.perf_counter()

        if debug:
            read_ms = (t_read - loop_start) * 1e3
            send_ms = (t_send - t_read) * 1e3
            lines = ["--- JOINT DEBUG ---"]
            lines.append(f"  SM:  {' '.join(f'{k}={v:+.2f}' for k, v in axes.items())}")
            for name in motor_names:
                marker = " *" if any(j == name for _, j, _ in axis_map) else ""
                lines.append(f"  {name:<{col_w}} {joint_positions[name]:>7.1f}{marker}")
            lines.append(f"  read={read_ms:.1f}ms  send={send_ms:.1f}ms")
            print("\n".join(lines))
            move_cursor_up(len(lines))

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


# ─────────────────────────────────────────────────────────────────────────────
# Standard pipeline teleop loop (leader-follower / IK)
# ─────────────────────────────────────────────────────────────────────────────

def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    debug: bool = False,
):
    """Standard teleoperation loop with processor pipeline."""
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if debug:
            debug_lines = ["--- TELEOP DEBUG ---"]
            for k, v in (robot_action_to_send or {}).items():
                if isinstance(k, str):
                    debug_lines.append(f"  {k:<20s} {v:>7.2f}")
            print("\n".join(debug_lines))
            move_cursor_up(len(debug_lines))

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action, compress_images=display_compressed_images)
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    try:
        if cfg.direct_joint is not None:
            direct_joint_teleop_loop(
                teleop=teleop,
                robot=robot,
                fps=cfg.fps,
                djc=cfg.direct_joint,
                max_gripper_pos=cfg.max_gripper_pos,
                duration=cfg.teleop_time_s,
                debug=cfg.debug,
            )
        else:
            teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
            teleop_loop(
                teleop=teleop,
                robot=robot,
                fps=cfg.fps,
                display_data=cfg.display_data,
                duration=cfg.teleop_time_s,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                display_compressed_images=display_compressed_images,
                debug=cfg.debug,
            )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
