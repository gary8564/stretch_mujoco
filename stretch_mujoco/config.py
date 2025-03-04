camera_settings = {
    "d405_rgb": {"fovy": 50, "width": 640, "height": 480},
    "d405_depth": {"fovy": 50, "width": 640, "height": 480},
    "d435i_camera_rgb": {"fovy": 62, "width": 640, "height": 480},
    "d435i_camera_depth": {"fovy": 62, "width": 640, "height": 480},
}

robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}

depth_limits = {"d405": 1, "d435i": 10}

actuator_names = [
    "arm",
    "gripper",
    "head_pan",
    "head_tilt",
    "left_wheel_vel",
    "lift",
    "right_wheel_vel",
    "wrist_pitch",
    "wrist_roll",
    "wrist_yaw",
]

allowed_position_actuators = [
    "arm",
    "gripper",
    "head_pan",
    "head_tilt",
    "lift",
    "wrist_pitch",
    "wrist_roll",
    "wrist_yaw",
    "base_rotate",
    "base_translate",
]

# default value from hello robot:
# default x vel: 0.3
# default r vel: 1.0
base_motion = {"timeout": 15, "default_x_vel": 0.2, "default_r_vel": 0.6}

# TODO: Add params to tune joints response motion profiles
