"""
PackNet-SfM datasets
====================

These datasets output images, camera calibration, depth maps and poses for depth and pose estimation

- KITTIDataset: reads from KITTI_raw
- DGPDataset: reads from a DGP .json file
- ImageDataset: reads from a folder containing image sequences (no support for depth maps)

"""


from enum import Enum

from datasets.entities.camera_frame import CameraFrame
from datasets.entities.imu_measurement import ImuMeasurement


class DataType(Enum):
    """Enum for different data types in the dataset."""

    IMU = "imu"
    CAMERA = "camera"
    GROUND_TRUTH = "ground_truth"

