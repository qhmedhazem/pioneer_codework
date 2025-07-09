import numpy as np
import time
import csv

from pathlib import Path
from typing import Iterator, Tuple, Union

from datasets import ImuMeasurement, CameraFrame, DataType

TIMESTAMP_SCALER = 1e-9


class EuRoCDataset(Iterator):
    def __init__(self, path):
        self.root = Path(path)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path {self.root} does not exist.")
        self.cache = dict(
            {
                "imu": [],
                "cam": [],
            }
        )

        self.imu_path = self.root / "mav0" / "imu0" / "data.csv"
        self.imu_offest = 0
        self.imu_cache = dict()
        self.imu_timestamps = []
        self.imu_limit = self.count_csv_entries(self.imu_path)

        # self.groundtruth_path = (
        #     self.root / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        # )
        # self.groundtruth_offest = 0
        # self.groundtruth_cache = dict()
        # self.groundtruth_timestamps = []
        # self.groundtruth_limit = self.count_csv_entries(self.groundtruth_path)

        self.cam0_path = self.root / "mav0" / "cam0" / "data.csv"
        self.cam0_frames_path = self.root / "mav0" / "cam0" / "data"
        self.cam0_timestamps = []
        self.cam0_limit = self.count_csv_entries(self.cam0_path)

        self.cam1_path = self.root / "mav0" / "cam1" / "data.csv"
        self.cam1_frames_path = self.root / "mav0" / "cam1" / "data"
        self.cam1_timestamps = []
        self.cam1_limit = self.count_csv_entries(self.cam1_path)

        self.cam_offest = 0
        self.cam_timestamps = []
        self.cam_cache = dict()
        self.cam_limit = min(self.cam0_limit, self.cam1_limit)

        self.load_cache()
        self.pipestart = 0.0
        self.starttime = min(
            self.imu_timestamps[0],
            self.cam0_timestamps[0],
            self.cam1_timestamps[0],
        )

    def parse_imu_frame(self, line: str) -> Tuple[float, ImuMeasurement]:
        """
        Parse a line from the IMU data CSV file.
        Format: (timestamp [ns], w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],
                 a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2])
        """
        line = [float(_) for _ in line.strip().split(",")]
        timestamp = line[0] * TIMESTAMP_SCALER
        wm = np.array(line[1:4])
        am = np.array(line[4:7])

        imu_measurement = ImuMeasurement(
            timestamp=timestamp,
            angular_velocity=wm,
            linear_acceleration=am,
        )

        return timestamp, imu_measurement

    def parse_camera_frame(
        self, cam0_line: str, cam1_line: str
    ) -> Tuple[float, CameraFrame]:
        """
        Parse a line from the camera data CSV file.
        Format: (timestamp [ns], image_cam0_path, image_cam1_path)
        """
        cam0_line = cam0_line.strip().split(",")
        cam1_line = cam1_line.strip().split(",")

        timestamp = float(cam0_line[0]) * TIMESTAMP_SCALER
        image_cam0_path = self.cam0_frames_path / cam0_line[1]
        image_cam1_path = self.cam1_frames_path / cam1_line[1]

        return timestamp, CameraFrame(
            timestamp=timestamp,
            image_cam0_path=str(image_cam0_path),
            image_cam1_path=str(image_cam1_path),
        )

    def load_cache(self):
        """Load IMU and camera data into cache."""
        with open(self.imu_path, "r") as imu_file:
            next(imu_file)
            for line in imu_file:
                ts, imu_measurement = self.parse_imu_frame(line)
                self.imu_cache[ts] = imu_measurement
                self.imu_timestamps.append(ts)

        with open(self.cam0_path, "r") as cam0_file, open(
            self.cam1_path, "r"
        ) as cam1_file:
            next(cam0_file)
            next(cam1_file)
            for cam0_line, cam1_line in zip(cam0_file, cam1_file):
                ts, camera_frame = self.parse_camera_frame(cam0_line, cam1_line)
                self.cam_cache[ts] = camera_frame
                self.cam_timestamps.append(ts)
                self.cam0_timestamps.append(ts)
                self.cam1_timestamps.append(ts)

    @staticmethod
    def count_csv_entries(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return sum(1 for _ in reader)

    def get_actual_duration(self):
        return time.time() - self.pipestart

    def get_euroc_pipe_timestamp(self):
        return self.starttime + self.get_actual_duration()

    def start(self):
        self.pipestart = time.time()

    def get_imu_read(self):
        imu_ts = self.imu_timestamps[self.imu_offest]
        if imu_ts < self.get_euroc_pipe_timestamp():
            imu_measurement = self.imu_cache[imu_ts]
            self.imu_offest += 1
            return True, DataType.IMU, imu_measurement
        else:
            return False, DataType.IMU, None

    def get_camera_read(self):
        cam_ts = self.cam_timestamps[self.cam_offest]
        if cam_ts < self.get_euroc_pipe_timestamp():
            camera_frame = self.cam_cache[cam_ts]
            self.cam_offest += 1
            return True, DataType.CAMERA, camera_frame
        else:
            return False, DataType.CAMERA, None

    def __next__(self) -> Tuple[DataType, Union[ImuMeasurement, CameraFrame]]:
        while True:
            # print(
            #     "current euroc time: ",
            #     self.get_euroc_pipe_timestamp(),
            #     "current actual duration: ",
            #     self.get_actual_duration(),
            #     "imu next frame: ",
            #     self.imu_timestamps[self.imu_offest],
            #     "camera next frame: ",
            #     self.cam_timestamps[self.cam_offest],
            # )

            if self.imu_offest < self.imu_limit and self.cam_offest < self.cam_limit:
                imu_ts = self.imu_timestamps[self.imu_offest]
                cam_ts = self.cam_timestamps[self.cam_offest]
                if imu_ts <= cam_ts:
                    ret, data_type, imu_measurement = self.get_imu_read()
                    if ret:
                        return data_type, imu_measurement
                elif cam_ts < imu_ts:
                    ret, data_type, camera_frame = self.get_camera_read()
                    if ret:
                        return data_type, camera_frame
            elif self.cam_offest < self.cam_limit:
                ret, data_type, camera_frame = self.get_camera_read()
                if ret:
                    return data_type, camera_frame
            elif self.imu_offest < self.imu_limit:
                ret, data_type, imu_measurement = self.get_imu_read()
                if ret:
                    return data_type, imu_measurement
            else:
                raise StopIteration()

            time.sleep(0.0005)
