import cv2
import os
from datetime import datetime
from pathlib import Path


from geometry.pose_estimator import VINS
CONFIG_DIR = Path("./config/kitti_raw/kitti_09_30_config.yaml")


def load_kitti_timestamps(path):
    timestamps_path = os.path.join(path, "image_02", "timestamps.txt")
    print(f"Loading timestamps from: {timestamps_path}")

    timestamps = []
    with open(timestamps_path, "r") as f:
        for line in f:
            time_str = line.strip()
            # Truncate to microsecond precision
            if '.' in time_str:
                date_part, frac_part = time_str.split('.')
                frac_part = frac_part[:6]  # Keep only 6 digits
                time_str = f"{date_part}.{frac_part}"
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(dt.timestamp())
    return timestamps

def load_kitti_pair(sequence_path, index):
    img_name = f"{index:010d}.png"
    
    left_path = os.path.join(sequence_path, "image_02", "data", img_name)
    right_path = os.path.join(sequence_path, "image_03", "data", img_name)

    print(f"Loading image pair: {left_path}, {right_path}")
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        raise FileNotFoundError(f"Could not load image pair at index {index}")

    return img_left, img_right

def precompute_kitti_pose(
    sequence_dir: str,
    outfile: str = "poses.txt"
):
    vio = VINS(str(CONFIG_DIR))
    output_file = os.path.join(sequence_dir, "image_02", outfile)

    timestamps = load_kitti_timestamps(sequence_dir)
    poses = []

    for frame_idx, ts in enumerate(timestamps):
        left_img, right_img = load_kitti_pair(sequence_dir, frame_idx)
        vio.input_stereo(left_img, right_img, ts)
        vio.wait_for_trackimg(ts)

        track_img = vio.get_track_image() 

        latest_odom = vio.get_pose_in_world_frame()
        if latest_odom is not None:
            print(f"Pose Matrix:\n{latest_odom}")

        # Display images
        # cv2.imshow("Left Image", left_img)
        # cv2.imshow("Right Image", right_img)
        # cv2.imshow("Tracked Image", track_img)

        # cv2.waitKey(10)

    with open(output_file, "w") as f:
        for pose in poses:
            f.write(f"{pose}\n")

    print(f"Computed poses saved to {output_file}")
    # cv2.destroyAllWindows()
    


if __name__ == "__main__":
    sequence_dir = "./data/datasets/KITTI_raw/2011_09_26/2011_09_26_drive_0002_sync" 
    precompute_kitti_pose(sequence_dir)