from enum import Enum

from datasets.entities.camera_frame import CameraFrame
from datasets.entities.imu_measurement import ImuMeasurement


class DataType(Enum):
    """Enum for different data types in the dataset."""

    IMU = "imu"
    CAMERA = "camera"
    GROUND_TRUTH = "ground_truth"
