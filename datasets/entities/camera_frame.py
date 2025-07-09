import cv2
from dataclasses import dataclass


@dataclass
class CameraFrame:
    timestamp: float

    image_cam0_path: str
    image_cam1_path: str

    depth_cam0_path: str = None
    depth_cam1_path: str = None

    @staticmethod
    def read_img_from_path(path: str):
        return cv2.imread(path, cv2.COLOR_BGR2GRAY)

    def read_cam0(self):
        return self.read_img_from_path(self.image_cam0_path)

    def read_cam1(self):
        return self.read_img_from_path(self.image_cam1_path)

    def read_depth_cam0(self):
        if self.depth_cam0_path is not None:
            return self.read_img_from_path(self.depth_cam0_path)
        else:
            raise ValueError("Depth path for camera 0 is not set.")

    def read_depth_cam1(self):
        if self.depth_cam1_path is not None:
            return self.read_img_from_path(self.depth_cam1_path)
        else:
            raise ValueError("Depth path for camera 1 is not set.")
