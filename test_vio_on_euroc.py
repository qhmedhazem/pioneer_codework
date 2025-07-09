import time
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ---- project imports ---------------------------------------------------------
from datasets.euroc_dataset import EuRoCDataset, DataType  # <- your loader
from pose_estimator.devel.lib.vinspy import Estimator
from config.vio_parameters import Parameters  # <- your parameters class
# from visualization.pose_visualizer import PoseVisualizer

# -----------------------------------------------------------------------------
# Configuration (adjust as needed)
# -----------------------------------------------------------------------------
CONFIG_DIR = Path("./config/euroc/euroc_mono_imu_config.yaml")
DATA_DIR = Path("./data/euroc/MH_01_easy")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

ts = None
img = None
track_img = None
angular_velocity = None
linear_acceleration = None

def run_on_euroc():
    # must be a 4x4 transformation matrix
    # visualizer = PoseVisualizer()
    vio = Estimator("./config/euroc/euroc_mono_imu_config.yaml")

    class_attributes = list(name for name in dir(vio) if not name.startswith('__'))
    print(class_attributes)

    dataset = EuRoCDataset(str(DATA_DIR))
    dataset.start()

    initalized = vio.initialize()

    print("Is Initalized?", initalized)

        # clear_state
        # set_parameter
        # change_sensor_type
        # input_image
        # input_imu
        # get_pose_in_world_frame
        # get_pose_in_world_frame_at

    for dtype, data in dataset:
        # print(f"Processing data of type: {dtype}")
        if dtype == DataType.CAMERA:
            ts = data.timestamp
            img = data.read_cam0()
            if img is None:
                print("No Image")
                continue
            
            
            vio.input_image(ts, img.copy())
            vio.wait_for_vio(ts)
            # vio.wait_for_optimization(ts)

            ret_track, track_img = vio.get_track_image()
            if ret_track:
                track_img = track_img.copy()
            else:
                track_img = np.zeros((480, 640, 3), dtype=np.uint8)

            if img is not None:
                cv2.imshow("Camera Image", img)
            if track_img is not None:
                cv2.imshow("Tracked Image", track_img)

            cv2.waitKey(0)
        elif dtype == DataType.IMU:
            ts = data.timestamp
            angular_velocity = data.angular_velocity
            linear_acceleration = data.linear_acceleration
            vio.input_imu(ts, linear_acceleration.copy(), angular_velocity.copy())

        print("Receving pose matrix")
        ret_pose, pose_matrix = vio.get_pose_in_world_frame()
        if ret_pose:
            print(f"Pose Matrix:\n{pose_matrix}")
            # visualizer.add_pose(pose_matrix)
        print("Received pose matrix")



if __name__ == "__main__":
    # print(Estimator)
    run_on_euroc()