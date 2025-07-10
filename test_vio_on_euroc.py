import time
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ---- project imports ---------------------------------------------------------
from datasets.euroc_dataset import EuRoCDataset, DataType  # <- your loader
from geometry.pose_estimator import VINS
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
    vio = VINS(str(CONFIG_DIR))

    dataset = EuRoCDataset(str(DATA_DIR))
    dataset.start()

    # try:
    #     print("Starting VINS...")
    #     vio.start()
    # except Exception as e:
    #     print(f"Error starting VINS: {e}")
    #     return
    
    print("VINS started successfully.")

    for dtype, data in dataset:
        # print(f"Processing data of type: {dtype}")
        if dtype == DataType.CAMERA:
            ts = data.timestamp
            img = data.read_cam0()
            if img is None:
                print("No Image")
                continue
            
            vio.input_image(img.copy(), ts)
            vio.wait_for_trackimg(ts)
            # vio.wait_for_odom(ts)
            # vio.wait_for_vio(ts)
            # # vio.wait_for_optimization(ts)

            track_img = vio.get_track_image()
            if track_img is not None:
                track_img = track_img.copy()

            if img is not None:
                cv2.imshow("Camera Image", img)

            if track_img is not None:
                cv2.imshow("Tracked Image", track_img)

            cv2.waitKey(1)
        elif dtype == DataType.IMU:
            ts = data.timestamp
            angular_velocity = data.angular_velocity
            linear_acceleration = data.linear_acceleration
            vio.input_imu(linear_acceleration.copy(), angular_velocity.copy(), ts)

        latest_odom = vio.get_pose_in_world_frame()
        if latest_odom is not None:
            print(f"Pose Matrix:\n{latest_odom}")


    del vio
    del dataset
    print("VINS and dataset cleaned up.")
        # time.sleep(0.01)


if __name__ == "__main__":
    # print(Estimator)
    run_on_euroc()