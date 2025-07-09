import time
import numpy as np
import matplotlib.pyplot as plt

from queue import Queue
from datasets.euroc_dataset import EuRoCDataset, DataType


def handle_imu(x):
    print("imu", x.timestamp, x.angular_velocity, x.linear_acceleration)


def handle_gt(x):
    print("gt", x.timestamp)


if __name__ == "__main__":
    path = "./data/euroc/cam_checkerboard"
    dataset = EuRoCDataset(path)
    dataset.start()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    img_plot = None

    for type, data in dataset:
        # Assuming d contains an image as d.image (adjust as needed)
        if type == DataType.CAMERA:
            image0 = data.read_cam0()
            if img_plot is None:
                img_plot = ax.imshow(image0, cmap="gray")
            else:
                img_plot.set_data(image0)
            ax.set_title(f"Camera Frame at {data.timestamp:.3f} s")
            plt.pause(0.001)
        elif type == DataType.IMU:
            handle_imu(data)
        print(type, data)
