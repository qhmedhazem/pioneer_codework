#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import rospy
import rosgraph
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image, Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


SETUP_PATH = os.getcwd() + "/pose_estimator/devel/setup.bash"


class VINS:
    """
    VINS front-end helper that can:

    • Spawn `roscore`, `vins_node`, and `loop_fusion_node` in child processes
      (all killed automatically on reset / object destruction).
    • Publish / subscribe sensor data exactly as before.
    • Provide timeout-aware waiters to avoid soft dead-locks.
    """

    # ------------------------------------------------------------------ #
    # construction & teardown
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_file: str,
        *,
        auto_start: bool = True,
        start_timeout: float = 10.0,
    ):
        """
        Args
        ----
        config_file : str
            Path to VINS-Fusion YAML configuration.
        auto_start  : bool
            If True the three ROS nodes are spawned immediately.
        start_timeout : float
            Seconds to wait for ROS master before giving up.
        """
        self.config_file = str(Path(config_file).expanduser().resolve())
        if not Path(self.config_file).is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        self._procs: dict[str, subprocess.Popen] = {}
        self.bridge = CvBridge()

        self.node_initalized = False

        self.pub_vins_restart = None
        self.pub_image_left = None
        self.pub_image_right = None
        self.pub_imu = None
        self.pub_gps = None
        self.latest_odom = None
        self.latest_odom_time = 0.0
        self.sub_odom = None
        self.latest_track = None
        self.latest_track_time = 0.0
        self.sub_track = None

        if auto_start:
            self.start(start_timeout)


    def __del__(self):
        """Best-effort cleanup."""
        self.reset()
        rospy.signal_shutdown("VINS object destroyed")
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # public control API
    # ------------------------------------------------------------------ #
    def start(self, timeout: float = 20.0):
        """Start the VINS interface, blocking until all nodes are ready."""
        self._start_child_processes(timeout)

        if not self.node_initalized:
            rospy.init_node("vins_interface", anonymous=True)
            self.node_initalized = True
        
        self._init_pub_sub()


    def reset(self):
        """Clear cached data _and_ restart the back-end."""
        self._terminate_child_processes()

        self.pub_vins_restart = None
        self.pub_image_left = None
        self.pub_image_right = None
        self.pub_imu = None
        self.pub_gps = None

        self.sub_odom = None
        self.latest_odom = None
        self.latest_odom_time = 0.0

        self.sub_track = None
        self.latest_track = None
        self.latest_track_time = 0.0




    def _start_ros_core(self, timeout: float = 10.0):
        """Start the ROS master (roscore) and wait for it to be online."""
        if "roscore" in self._procs:
            print("[POSE ESTIMATOR] roscore already running, skipping.")
            return

        try:
            self._procs["roscore"] = subprocess.Popen(
                ["bash", "-c", "roscore"],
                shell=False,
            )
            self._wait_for_master(timeout)
            print("[POSE ESTIMATOR] roscore started successfully.")
        except Exception as e:
            print(f"Error starting roscore: {e}")
            raise

    def _start_vins_node(self, timeout: float = 10.0):
        """Start the VINS node and wait for it to be ready."""
        if "vins_node" in self._procs:
            print("[POSE ESTIMATOR] vins_node already running, skipping.")
            return

        try:
            self._procs["vins_node"] = subprocess.Popen([
                    "bash",
                    "-c",
                    f"source {SETUP_PATH} && rosrun vins vins_node {self.config_file}"
                ],
                shell=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for the specific message in stdout
            start_time = time.time()
            while True:
                if self._procs["vins_node"].stdout is None:
                    break
                line = self._procs["vins_node"].stdout.readline()
                if not line:
                    if self._procs["vins_node"].poll() is not None:
                        raise RuntimeError("vins_node process exited unexpectedly.")
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Timed out waiting for vins_node to be ready.")
                    continue
                decoded_line = line.decode("utf-8", errors="replace").strip()
                print(f"[POSE ESTIMATOR] vins_node: {decoded_line}")

                if "waiting for image and imu" in decoded_line:
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timed out waiting for vins_node to be ready.")
            
            print("[POSE ESTIMATOR] vins_node started successfully.")
        except Exception as e:
            print(f"Error starting vins_node: {e}")
            raise

    def _start_loop_fusion_node(self, timeout: float = 10.0):
        """Start the loop fusion node and wait for it to be ready."""
        if "loop_fusion_node" in self._procs:
            print("[POSE ESTIMATOR] loop_fusion_node already running, skipping.")
            return

        try:
            self._procs["loop_fusion_node"] = subprocess.Popen([
                    "bash",
                    "-c",
                    f"source {SETUP_PATH} && rosrun loop_fusion loop_fusion_node {self.config_file}"
                ],
                shell=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for the specific message in stdout
            start_time = time.time()
            while True:
                if self._procs["loop_fusion_node"].stdout is None:
                    break
                line = self._procs["loop_fusion_node"].stdout.readline()
                if not line:
                    if self._procs["loop_fusion_node"].poll() is not None:
                        raise RuntimeError("loop_fusion_node process exited unexpectedly.")
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Timed out waiting for loop_fusion_node to be ready.")
                    continue
                decoded_line = line.decode("utf-8", errors="replace").strip()
                print(f"[POSE ESTIMATOR] loop_fusion_node: {decoded_line}")

                if "loop start load vocabulary" in decoded_line:
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timed out waiting for loop_fusion_node to be ready.")

            print("[POSE ESTIMATOR] loop_fusion_node started successfully.")
        except Exception as e:
            print(f"Error starting loop_fusion_node: {e}")
            raise

    # ------------------------------------------------------------------ #
    # external process management
    # ------------------------------------------------------------------ #
    def _start_child_processes(self, timeout: float) -> None:
        """Spawn roscore + vins_node + loop_fusion_node, with retries."""
        try:
            # Launch roscore first
            self._start_ros_core(timeout)
            self._start_vins_node(timeout)
            self._start_loop_fusion_node(timeout)
        except Exception as e:
            print(f"Error {e}")
            self.reset()
            raise

    def _terminate_child_processes(self) -> None:
        """Gracefully SIGINT, then SIGKILL leftover children."""
        for key, proc in self._procs.items():
            if proc.poll() is None:  # still running
                try:
                    proc.send_signal(subprocess.signal.SIGTERM)
                    proc.wait(timeout=100)
                    print("[POSE ESTIMATOR] Terminated " + key)
                except Exception:
                    # ensures zombies are not left over
                    try:
                        proc.send_signal(subprocess.signal.SIGKILL)
                    except Exception:
                        pass
                finally:
                    proc.terminate()
            self._procs[key] = None

    # ------------------------------------------------------------------ #
    # ROS I/O helpers
    # ------------------------------------------------------------------ #
    def _init_pub_sub(self):
        # Publishers
        self.pub_vins_restart = rospy.Publisher("/vins_restart", Bool, queue_size=10)
        self.pub_image_left = rospy.Publisher("/cam0/image_raw", Image, queue_size=10)
        self.pub_image_right = rospy.Publisher("/cam1/image_raw", Image, queue_size=10)
        self.pub_imu = rospy.Publisher("/imu0", Imu, queue_size=200)
        self.pub_gps = rospy.Publisher("/gps", NavSatFix, queue_size=10)

        # Subscribers
        self.latest_odom: Optional[np.ndarray] = None
        self.latest_odom_time: float = 0.0
        self.sub_odom = rospy.Subscriber(
            "/vins_estimator/odometry", Odometry, self._odom_callback, queue_size=100
        )

        self.latest_track: Optional[np.ndarray] = None
        self.latest_track_time: float = 0.0
        self.sub_track = rospy.Subscriber(
            "/vins_estimator/image_track", Image, self._track_callback, queue_size=50
        )


    # ------------------------------------------------------------------ #
    # data injection helpers
    # ------------------------------------------------------------------ #
    def input_image(self, cv_image: np.ndarray, timestamp: float):
        try:
            stamp = rospy.Time.from_sec(float(timestamp))
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            self.pub_image_left.publish(msg)
        except Exception as exc:
            rospy.logerr(f"Failed to publish image: {exc}")

    def input_stereo(
        self, cv_img_left: np.ndarray, cv_img_right: np.ndarray, timestamp: float
    ):
        try:
            stamp = rospy.Time.from_sec(float(timestamp))
            left_msg = self.bridge.cv2_to_imgmsg(cv_img_left, encoding="mono8")
            right_msg = self.bridge.cv2_to_imgmsg(cv_img_right, encoding="mono8")
            for m in (left_msg, right_msg):
                m.header.stamp = stamp
                m.header.frame_id = "world"
            self.pub_image_left.publish(left_msg)
            self.pub_image_right.publish(right_msg)
        except Exception as exc:
            rospy.logerr(f"Failed to publish stereo images: {exc}")

    def input_imu(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float):
        try:
            stamp = rospy.Time.from_sec(float(timestamp))
            msg = Imu()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
            self.pub_imu.publish(msg)
        except Exception as exc:
            rospy.logerr(f"Failed to publish IMU: {exc}")

    def input_gps(self, lat: float, lon: float, alt: float, timestamp: float):
        try:
            stamp = rospy.Time.from_sec(float(timestamp))
            msg = NavSatFix()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.latitude, msg.longitude, msg.altitude = lat, lon, alt
            self.pub_gps.publish(msg)
        except Exception as exc:
            rospy.logerr(f"Failed to publish GPS: {exc}")

    # ------------------------------------------------------------------ #
    # timeout-aware waiters
    # ------------------------------------------------------------------ #
    def wait_for_odom(self, target_time: float, timeout: float = 20.0):
        _timeout_wait(
            lambda: self.latest_odom_time >= target_time,
            timeout,
            "odometry message",
        )

    def wait_for_trackimg(self, target_time: float, timeout: float = 20.0):
        _timeout_wait(
            lambda: self.latest_track_time >= target_time,
            timeout,
            "tracking image",
        )

    # ------------------------------------------------------------------ #
    # getters
    # ------------------------------------------------------------------ #
    def get_track_image(self) -> Optional[np.ndarray]:
        return self.latest_track

    def get_pose_in_world_frame(self) -> Optional[np.ndarray]:
        return self.latest_odom

    # ------------------------------------------------------------------ #
    # subscriber callbacks
    # ------------------------------------------------------------------ #
    def _odom_callback(self, msg: Odometry):
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            q /= np.linalg.norm(q)
            x, y, z, w = q
            R = np.array(
                [
                    [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
                ]
            )
            t = np.array([position.x, position.y, position.z])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            self.latest_odom = T
            self.latest_odom_time = msg.header.stamp.to_sec()
        except Exception as exc:
            rospy.logwarn(f"Bad odom message: {exc}")

    def _track_callback(self, msg: Image):
        try:
            self.latest_track = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            self.latest_track_time = msg.header.stamp.to_sec()
        except Exception as exc:
            rospy.logwarn(f"Bad track image: {exc}")


    # ---------------------------------------------------------------------- #
    # helper utilities
    # ---------------------------------------------------------------------- #
    def _wait_for_master(self, timeout: float):
        """Block until ROS master is online or raise TimeoutError."""
        start = time.time()
        while time.time() - start < timeout:
            if rosgraph.is_master_online():
                return
            time.sleep(0.2)
        raise TimeoutError("Timed out waiting for roscore to start.")

    def _wait_for_nodes(self, timeout: float):
        """Wait until ROS master has at least one listener (subscriber or service client)."""
        start = time.time()
        master = rosgraph.Master('/cam0/image_raw')
        while time.time() - start < timeout:
            try:
                state = master.getSystemState()
                print(state)
                publishers, subscribers, services = state
                num_listeners = sum(len(s[1]) for s in subscribers) + sum(len(s[1]) for s in services)
                if num_listeners > 0:
                    return
            except Exception:
                pass
            rospy.sleep(0.2)
        raise TimeoutError("Timed out waiting for ROS nodes to have listeners.")


def _timeout_wait(predicate, timeout: float, what: str):
    """Spin until predicate() is True or raise TimeoutError."""
    start = time.time()
    while not predicate():
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {what}.")
        rospy.sleep(0.01)
