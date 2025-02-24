"""
Python sample script for interfacing with the Stretch Mujoco simulator
"""

import os
import threading
import time
from typing import Any, Dict, Optional

import click
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

import stretch_mujoco.config as config
import stretch_mujoco.utils as utils


class StretchMujocoSimulator:
    """
    Stretch Mujoco Simulator class for interfacing with the Mujoco simulator
    """

    def __init__(
        self, scene_xml_path: Optional[str] = None, model: Optional[MjModel] = None
    ) -> None:
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
        """
        if scene_xml_path is None:
            scene_xml_path = utils.default_scene_xml_path
            self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        elif model is None:
            self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        if model is not None:
            self.mjmodel = model
        self.mjdata = mujoco.MjData(self.mjmodel)
        self._set_camera_properties()
        self.urdf_model = utils.URDFmodel()

        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()
        self.wheel_diameter = config.robot_settings["wheel_diameter"]
        self.wheel_separation = config.robot_settings["wheel_separation"]
        self.status = {
            "time": None,
            "base": {
                "x": None,
                "y": None,
                "theta": None,
                "x_vel": None, 
                "theta_vel": None
            },
            "lift": {
                "pos": None, 
                "vel": None, 
                "motor": {"effort_pct": None}},
            "arm": {
                "pos": None, 
                "vel": None,
                "motor": {"effort_pct": None}},
            "head_pan": {
                "pos": None, 
                "vel": None, 
                "effort": None},
            "head_tilt": {
                "pos": None, 
                "vel": None, 
                "effort": None},
            "wrist_yaw": {"pos": None, "vel": None, "effort": None},
            "wrist_pitch": {"pos": None, "vel": None, "effort": None},
            "wrist_roll": {"pos": None, "vel": None, "effort": None},
            "gripper": {"pos": None, "pos_pct": None, "vel": None, "effort": None},
            "pimu": {
                "imu": {'ax': 0.0, 'ay': 0.0, 'az': 0.0,
                        'gx': 0.0, 'gy': 0.0, 'gz': 0.0,
                        'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0}
            },
            "wacc": {
                "ax": None,
                "ay": None,
                "az": None
            }
        }
        self._running = False
        self.viewer = mujoco.viewer
        self._base_in_pos_motion = False
        self._headless_running = False
        self.end_of_arm_name = 'gripper' 
        self.waypoints = []         # [time, [pos], [vel], [acc], [eff]] with joints in order!
        self.trajectory = False
        self.cur_wp_time = 0
        self.lidar_data = None
        self.mjmodel.opt.timestep = 0.002  # Increase from default

    def _set_camera_properties(self):
        """
        Set the camera properties
        """
        for camera_name, settings in config.camera_settings.items():
            self.set_camera_params(
                camera_name, settings["fovy"], (settings["width"], settings["height"])
            )

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["sim_gripper_min_max"],
            config.robot_settings["gripper_min_max"],
        )

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["gripper_min_max"],
            config.robot_settings["sim_gripper_min_max"],
        )

    def home(self) -> None:
        """
        Move the robot to home position
        """
        self.mjdata.ctrl = self.mjmodel.keyframe("home").ctrl
    
    def is_homed(self) -> bool: #.TODO test this
        if np.allclose(self.mjdata.ctrl, self.mjmodel.keyframe("home").ctrl):
            return True
        else:
            return False

    def stow(self) -> None:
        """
        Move the robot to stow position
        """
        self.mjdata.ctrl = self.mjmodel.keyframe("stow").ctrl

    def move_to(self, actuator_name: str, pos: float) -> None:
        """
        Move the actuator to a specific position
        """
        if actuator_name in config.allowed_position_actuators:
            if actuator_name not in ["base_translate", "base_rotate"]:
                if actuator_name == "gripper":
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                else:
                    self.mjdata.actuator(actuator_name).ctrl = pos
            else:
                click.secho(f"{actuator_name} not allowed for move_to", fg="red")
        else:
            click.secho(
                f"Actuator {actuator_name} not reccognized."
                f"\n Available position actuators: {config.allowed_position_actuators}",
                fg="red",
            )

    def move_by(self, actuator_name: str, pos: float) -> None:
        """
        Move the actuator by a specific amount
        Args:
            actuator_name: str, name of the actuator
            pos: float, position to increment by
        """
        if actuator_name in config.allowed_position_actuators:
            if actuator_name in ["base_translate", "base_rotate"]:
                if self._base_in_pos_motion:
                    self._stop_base_pos_tracking()
                    time.sleep(1 / 20)
                if actuator_name == "base_translate":
                    threading.Thread(target=self._base_translate_by, args=(pos,)).start()
                else:
                    threading.Thread(target=self._base_rotate_by, args=(pos,)).start()
            else:
                if actuator_name == "gripper":
                    current_pos = self.status[actuator_name]["pos"]
                    new_pos = self._to_sim_gripper_range(
                        current_pos + pos
                    )
                    self.mjdata.actuator(actuator_name).ctrl = new_pos
                    self.status["gripper"]["pos"] = new_pos
                else:
                    current_pos = self.status[actuator_name]["pos"]
                    self.mjdata.actuator(actuator_name).ctrl = (
                        current_pos + pos
                    )
        else:
            click.secho(
                f"Actuator {actuator_name} not reccognized."
                f"\n Available position actuators: {config.allowed_position_actuators}",
                fg="red",
            )

    def set_base_velocity(self, v_linear: float, omega: float, _override=False) -> None:
        """
        Set the base velocity of the robot
        Args:
            v_linear: float, linear velocity
            omega: float, angular velocity
        """
        if not _override and self._base_in_pos_motion:
            self._stop_base_pos_tracking()
            time.sleep(1 / 20)
        w_left, w_right = self.diff_drive_inv_kinematics(v_linear, omega)
        self.mjdata.actuator("left_wheel_vel").ctrl = w_left
        self.mjdata.actuator("right_wheel_vel").ctrl = w_right

    def set_velocity(self, actuator_name: str, vel: float) -> None:
        """
        Set the velocity of the actuator
        """
        # TODO: Implement this method by ether moving to an integrated velocity acuators or have
        # separate robot xml configured by replacing position with velocity ctrl actuators
        raise NotImplementedError

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mjdata.body("base_link").xpos
        rotation = self.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def get_ee_pose(self) -> np.ndarray:
        return self.get_link_pose("link_grasp_center")

    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Pose of link in world frame"""
        cfg = {
            "wrist_yaw": self.status["wrist_yaw"]["pos"],
            "wrist_pitch": self.status["wrist_pitch"]["pos"],
            "wrist_roll": self.status["wrist_roll"]["pos"],
            "lift": self.status["lift"]["pos"],
            "arm": self.status["arm"]["pos"],
            "head_pan": self.status["head_pan"]["pos"],
            "head_tilt": self.status["head_tilt"]["pos"],
        }
        T = self.urdf_model.get_transform(cfg, link_name)
        base_xyt = self.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = utils.Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        world_coord = np.matmul(base_4x4, T)
        return world_coord

    def get_lidar(self):
        self.lidar_data = self.mjdata.sensordata[6:] #.TODO find function to return the correct data, hardcoded only temp fix
        return [r if (r > 0.17) else 12.0 for r in self.lidar_data]


    def _pull_status(self) -> Dict[str, Any]:
        """
        Pull joints status of the robot from the simulator
        """
        self.status["time"] = self.mjdata.time
        self.status["lift"]["pos"] = self.mjdata.actuator("lift").length[0]
        self.status["lift"]["vel"] = self.mjdata.actuator("lift").velocity[0]
        self.status["lift"]["motor"]["effort_pct"] = self.calculate_effort(self.mjdata.actuator("lift").force[0])

        self.status["arm"]["pos"] = self.mjdata.actuator("arm").length[0]
        self.status["arm"]["vel"] = self.mjdata.actuator("arm").velocity[0]
        self.status["arm"]["motor"]["effort_pct"] = self.calculate_effort(self.mjdata.actuator("arm").force[0])

        self.status["head_pan"]["pos"] = self.mjdata.actuator("head_pan").length[0]
        self.status["head_pan"]["vel"] = self.mjdata.actuator("head_pan").velocity[0]
        self.status["head_pan"]["effort"] = self.calculate_effort(self.mjdata.actuator("head_pan").force[0])

        self.status["head_tilt"]["pos"] = self.mjdata.actuator("head_tilt").length[0]
        self.status["head_tilt"]["vel"] = self.mjdata.actuator("head_tilt").velocity[0]
        self.status["head_tilt"]["effort"] = self.calculate_effort(self.mjdata.actuator("head_tilt").force[0])

        self.status["wrist_yaw"]["pos"] = self.mjdata.actuator("wrist_yaw").length[0]
        self.status["wrist_yaw"]["vel"] = self.mjdata.actuator("wrist_yaw").velocity[0]
        self.status["wrist_yaw"]["effort"] = self.calculate_effort(self.mjdata.actuator("wrist_yaw").force[0])

        self.status["wrist_pitch"]["pos"] = self.mjdata.actuator("wrist_pitch").length[0]
        self.status["wrist_pitch"]["vel"] = self.mjdata.actuator("wrist_pitch").velocity[0]
        self.status["wrist_pitch"]["effort"] = self.calculate_effort(self.mjdata.actuator("wrist_pitch").force[0])
        
        self.status["wrist_roll"]["pos"] = self.mjdata.actuator("wrist_roll").length[0]
        self.status["wrist_roll"]["vel"] = self.mjdata.actuator("wrist_roll").velocity[0]
        self.status["wrist_roll"]["effort"] = self.calculate_effort(self.mjdata.actuator("wrist_roll").force[0])
        
        real_gripper_pos = self._to_real_gripper_range(self.mjdata.actuator("gripper").length[0])
        # self.status["gripper"]["pos"] = real_gripper_pos
        self.status["gripper"]["pos_pct"] = real_gripper_pos
        self.status["gripper"]["vel"] = self.mjdata.actuator("gripper").velocity[
            0
        ]  # This is still in sim gripper range
        self.status["gripper"]["effort"] = self.calculate_effort(self.mjdata.actuator("gripper").force[0])
        left_wheel_vel = self.mjdata.actuator("left_wheel_vel").velocity[0]
        right_wheel_vel = self.mjdata.actuator("right_wheel_vel").velocity[0]

        (
            self.status["base"]["x"],
            self.status["base"]["y"],
            self.status["base"]["theta"],
        ) = self.get_base_pose()
        (
            self.status["base"]["x_vel"],
            self.status["base"]["theta_vel"],
        ) = self.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)
        self.status["base"]["y_vel"] = 0.0
        #calculations partially taken from diff_drive_fwd_kinematics 
        R = self.wheel_diameter / 2.0
        linear_velocity = R * (left_wheel_vel + right_wheel_vel) / 2.0
        self.status["wacc"]["ax"] = linear_velocity * np.cos(self.mjdata.qpos[2])
        self.status["wacc"]["ay"] = linear_velocity * np.sin(self.mjdata.qpos[2])
        self.status["wacc"]["az"] = -9.81
        # PIMU
        imu_gyro = self.mjdata.sensordata[0:3] #.TODO find function to return the correct data, hardcoded only temp fix
        self.status["pimu"]["imu"]["gx"] = imu_gyro[0]
        self.status["pimu"]["imu"]["gy"] = imu_gyro[1]
        self.status["pimu"]["imu"]["gz"] = imu_gyro[2]
        imu_accel = self.mjdata.sensordata[3:6] #.TODO find function to return the correct data, hardcoded only temp fix
        self.status["pimu"]["imu"]["ax"] = imu_accel[0]
        self.status["pimu"]["imu"]["ay"] = imu_accel[1]
        self.status["pimu"]["imu"]["az"] = imu_accel[2]
        imu_quat = self.mjdata.xquat[0] #.TODO should be correct index, because first def body in xml
        self.status["pimu"]["imu"]["qw"] = imu_quat[0]
        self.status["pimu"]["imu"]["qx"] = imu_quat[1]
        self.status["pimu"]["imu"]["qy"] = imu_quat[2]
        self.status["pimu"]["imu"]["qz"] = imu_quat[3]

        return self.status
    
    def get_status(self):
        return self.status.copy()
    
    def get_specific_status(self, name):
        for key, value in self.status:
            if key == name:
                return value
            elif isinstance(value, dict):
                for key, value in self.status:
                    if key == name:
                        return value
        return None
    
    def calculate_effort(self, force): 
        return force #.? * 100 / MAX_FORCE

    def pull_camera_data(self) -> dict:
        """
        Pull camera data from the simulator and return as a dictionary
        """
        data = {}
        data["time"] = self.mjdata.time

        self.rgb_renderer.update_scene(self.mjdata, "d405_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d405_rgb")

        data["cam_d405_rgb"] = self.rgb_renderer.render()
        data["cam_d405_depth"] = utils.limit_depth_distance(
            self.depth_renderer.render(), config.depth_limits["d405"]
        )
        data["cam_d405_K"] = self.get_camera_params("d405_rgb")

        self.rgb_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d435i_camera_rgb")

        data["cam_d435i_rgb"] = self.rgb_renderer.render()
        data["cam_d435i_depth"] = utils.limit_depth_distance(
            self.depth_renderer.render(), config.depth_limits["d435i"]
        )
        data["cam_d435i_K"] = self.get_camera_params("d435i_camera_rgb")

        self.rgb_renderer.update_scene(self.mjdata, "nav_camera_rgb")
        data["cam_nav_rgb"] = self.rgb_renderer.render()
        return data

    def set_camera_params(self, camera_name: str, fovy: float, res: tuple) -> None:
        """
        Set camera parameters
        Args:
            camera_name: str, name of the camera
            fovy: float, vertical field of view in degrees
            res: tuple, size of the camera Image
        """
        cam = self.mjmodel.camera(camera_name)
        self.mjmodel.cam_fovy[cam.id] = fovy
        self.mjmodel.cam_resolution[cam.id] = res

    def get_camera_params(self, camera_name: str) -> np.ndarray:
        """
        Get camera parameters
        """
        cam = self.mjmodel.camera(camera_name)
        d = {
            "fovy": cam.fovy,
            "f": self.mjmodel.cam_intrinsic[cam.id][:2],
            "p": self.mjmodel.cam_intrinsic[cam.id][2:],
            "res": self.mjmodel.cam_resolution[cam.id],
        }
        K = utils.compute_K(d["fovy"][0], d["res"][0], d["res"][1])
        return K

    def __ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.mjdata = data
        self.mjmodel = model
        self._pull_status()
        if self.trajectory:
            threading.Thread(
                target=self.follow_waypoints, name="mujoco_headless_thread"
            ).start()


    def diff_drive_inv_kinematics(self, V: float, omega: float) -> tuple:
        """
        Calculate the rotational velocities of the left and right wheels for a
        differential drive robot.
        """
        R = self.wheel_diameter / 2
        L = self.wheel_separation
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")

        # Calculate the rotational velocities of the wheels
        w_left = (V - (omega * L / 2)) / R
        w_right = (V + (omega * L / 2)) / R

        return (w_left, w_right)

    def diff_drive_fwd_kinematics(self, w_left: float, w_right: float) -> tuple:
        """
        Calculate the linear and angular velocity of a differential drive robot.
        """
        R = self.wheel_diameter / 2
        L = self.wheel_separation
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")

        # Linear velocity (V) is the average of the linear velocities of the two wheels
        V = R * (w_left + w_right) / 2.0

        # Angular velocity (omega) is the difference in linear velocities divided by the distance
        # between the wheels
        omega = R * (w_right - w_left) / L

        return (V, omega)
    
    def add_waypoint(self, duration, positions, velocities, accelerations, effort):
        #print("waypoints", self.waypoints)
        new_waypoint = []
        new_waypoint.append(self.mjdata.time + duration.sec + duration.nanosec)
        new_waypoint.append(positions)
        new_waypoint.append(velocities)
        new_waypoint.append(accelerations)
        new_waypoint.append(effort)
        #following code copied from stretch_body.trajectories
        if len(self.waypoints) == 0:
            self.waypoints.append(new_waypoint)
            return
        if new_waypoint in self.waypoints:
            return
        if new_waypoint[0] < self.waypoints[0][0]:
            self.waypoints.insert(0, new_waypoint)
            return
        if new_waypoint[0] > self.waypoints[0][0]:
            self.waypoints.append(new_waypoint)
            return
        for i, other_waypoint in enumerate(self.waypoints):
            if new_waypoint[0] < other_waypoint[0]:
                self.waypoints.insert(i, new_waypoint)
                return

    def stop_trajectory(self):
        self.set_base_velocity(0.0, 0.0)
        self.move_to('arm', 0.0)
        self.move_to('lift', 0.0)
        self.move_to('head_pan', 0.0)
        self.move_to('head_tilt', 0.0)
        self.move_to('wrist_yaw', 0.0)
        self.move_to('wrist_pitch', 0.0)
        self.move_to('wrist_roll', 0.0)
        self.move_to('gripper', 0.0)
        self.clear_trajectory()
        #.TODO test if this is working
    
    def clear_trajectory(self):
        self.waypoints = []
    
    def follow_trajectory(self):
        if len(self.waypoints) == 0: #.TODO add more poss abort conditions
            #print("No waypoints")
            return False
        self.trajectory = True
        return True

    def follow_waypoints(self):
        # for simplicity we ignore vel, acc and eff for now
        if len(self.waypoints) == 0:
            return
        if self.waypoints[0][0] < self.mjdata.time:
            self.waypoints.pop(0)
            if len(self.waypoints) == 0:
                return
        if self.waypoints[0][0] == self.cur_wp_time:
            return
        cur_waypoint = self.waypoints[0][1]
        self.cur_wp_time = self.waypoints[0][0]
        self._base_translate_by(cur_waypoint[0])
        self._base_rotate_by_relative_to_world(cur_waypoint[1])
        self.move_to('lift', cur_waypoint[2])
        self.move_to('arm', cur_waypoint[3])
        self.move_to('head_pan', cur_waypoint[4])
        self.move_to('head_tilt', cur_waypoint[5])
        self.move_to('wrist_yaw', cur_waypoint[6])
        self.move_to('wrist_pitch', cur_waypoint[7])
        self.move_to('wrist_roll', cur_waypoint[8])
        self.move_to('gripper', cur_waypoint[9])
        self.waypoints.pop(0)

    def __run(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        mujoco.set_mjcb_control(self.__ctrl_callback)
        self.viewer.launch(
            self.mjmodel,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        )

    def __run_headless_simulation(self) -> None:
        """
        Run the simulation without the viewer headless
        """
        print("Running headless simulation...")
        self._headless_running = True
        while self._headless_running:
            start_ts = time.perf_counter()
            mujoco.mj_step(self.mjmodel, self.mjdata)
            self.__ctrl_callback(self.mjmodel, self.mjdata)
            elapsed = time.perf_counter() - start_ts
            if elapsed < self.mjmodel.opt.timestep:
                time.sleep(self.mjmodel.opt.timestep - elapsed)

    def _stop_base_pos_tracking(self) -> None:
        """
        Stop the base position tracking
        """
        self._base_in_pos_motion = False

    def _base_translate_by(self, x_inc: float) -> None:
        """
        Translate the base by a certain w.r.t base global pose
        """
        start_pose = self.get_base_pose()[:2]
        self._base_in_pos_motion = True
        sign = 1 if x_inc > 0 else -1
        start_ts = time.perf_counter()
        while np.linalg.norm(self.get_base_pose()[:2] - start_pose) <= abs(x_inc):
            if self._base_in_pos_motion:
                self.set_base_velocity(
                    config.base_motion["default_x_vel"] * sign, 0, _override=True
                )
                if time.perf_counter() - start_ts > config.base_motion["timeout"]:
                    click.secho("Base translation timeout", fg="red")
                    break
            else:
                break
            time.sleep(1 / 30)
        self.set_base_velocity(0, 0)
        self._base_in_pos_motion = False

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.get_base_pose()[-1]
        self._base_in_pos_motion = True
        sign = 1 if theta_inc > 0 else -1
        start_ts = time.perf_counter()
        while abs(start_pose - self.get_base_pose()[-1]) <= abs(theta_inc):
            if self._base_in_pos_motion:
                self.set_base_velocity(
                    0, config.base_motion["default_r_vel"] * sign, _override=True
                )
                time.sleep(1 / 30)
                if time.perf_counter() - start_ts > config.base_motion["timeout"]:
                    click.secho("Base rotation timeout", fg="red")
                    break
            else:
                break
        self.set_base_velocity(0, 0)
        self._base_in_pos_motion = False

    def _base_rotate_by_relative_to_world(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.get_base_pose()[-1]
        self._base_in_pos_motion = True
        sign = 1 if ((theta_inc - start_pose)% 6.283185) < 3.141592 else -1
        start_ts = time.perf_counter()
        while (sign * (theta_inc - self.get_base_pose()[-1]))% 6.283185 < 3.141592: 
            if self._base_in_pos_motion:
                self.set_base_velocity(
                    0, config.base_motion["default_r_vel"] * sign, _override=True
                )
                time.sleep(1 / 30)
                if time.perf_counter() - start_ts > config.base_motion["timeout"]:
                    click.secho("Base rotation timeout", fg="red")
                    break
            else:
                break
        self.set_base_velocity(0, 0)
        self._base_in_pos_motion = False


    def is_running(self) -> bool:
        """
        Check if the simulator is running
        """
        return self._running or self._headless_running

    def start(self, show_viewer_ui: bool = False, headless: bool = False) -> None:
        """
        Start the simulator in a using blocking Managed-vieiwer for precise timing. And user code
        is looped through callback. Some projects might need non-blocking Passive-vieiwer.
        For more info visit: https://mujoco.readthedocs.io/en/stable/python.html#managed-viewer
        Args:
            show_viewer_ui: bool, whether to show the Mujoco viewer UI
            headless: bool, whether to run the simulation in headless mode
        """
        if not headless:
            threading.Thread(
                target=self.__run, name="mujoco_viewer_thread", args=(show_viewer_ui,)
            ).start()
        else:
            threading.Thread(
                target=self.__run_headless_simulation, name="mujoco_headless_thread"
            ).start()
        click.secho("Starting Stretch Mujoco Simulator...", fg="green")
        while not self.mjdata.time:
            time.sleep(0.2)
        self._running = True
        self.home()

    def reset_state(self) -> None:
        """
        Reset the simulator to initial state (experimental)
        """
        _headless_reset = self._headless_running
        if self._headless_running:
            self._headless_running = False
            time.sleep(0.3)
        else:
            click.secho(
                "StretchMujocoSimulator.reset_state() method is experimental with Viewer running",
                fg="yellow",
            )
        mujoco.mj_resetData(self.mjmodel, self.mjdata)
        print("Resetting the simulator to initial state...")
        if _headless_reset:
            threading.Thread(
                target=self.__run_headless_simulation, name="mujoco_headless_thread"
            ).start()
        while not self.mjdata.time:
            time.sleep(0.2)
        self.home()

    def stop(self) -> None:
        """
        Stop the simulator
        """
        self._running = False
        if not self._headless_running:
            click.secho(
                f"Exiting Stretch Mujoco Simulator with viewer... runtime={self.status['time']}s",
                fg="red",
            )
            os.kill(os.getpid(), 9)
        else:
            click.secho(
                f"Stopping headless simulation... runtime={self.status['time']}s", fg="yellow"
            )
            self._headless_running = False
            time.sleep(0.5)
            mujoco.mj_resetData(self.mjmodel, self.mjdata)


@click.command()
@click.option(
    "--scene-xml-path", default=utils.default_scene_xml_path, help="Path to the scene xml file"
)
@click.option("--headless", is_flag=True, help="Run the simulation headless")
def main(
    scene_xml_path: str,
    headless: bool,
) -> None:
    robot_sim = StretchMujocoSimulator(scene_xml_path)
    robot_sim.start(headless=headless)
    # display camera feeds
    try:
        while robot_sim.is_running():
            camera_data = robot_sim.pull_camera_data()
            cv2.imshow("cam_d405_rgb", cv2.cvtColor(camera_data["cam_d405_rgb"], cv2.COLOR_RGB2BGR))
            cv2.imshow("cam_d405_depth", camera_data["cam_d405_depth"])
            cv2.imshow(
                "cam_d435i_rgb", cv2.cvtColor(camera_data["cam_d435i_rgb"], cv2.COLOR_RGB2BGR)
            )
            cv2.imshow("cam_d435i_depth", camera_data["cam_d435i_depth"])
            cv2.imshow("cam_nav_rgb", cv2.cvtColor(camera_data["cam_nav_rgb"], cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        robot_sim.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import warnings
    warnings.warn("use 'python -m stretch_mujoco', not 'python -m stretch_mujoco.stretch_mujoco'", DeprecationWarning)

    # Check if we are on macOS
    if os.uname().sysname == "Darwin":
        print("macOS detected. Please use the following command to run the simulator:")
        print("python3 -m stretch_mujoco")
    else:
        main()