import argparse
import glob
import os
import cv2
import numpy as np


from geometry.pinhole_camera import PinholeCamera
from lib.camera_helpers import load_intrinsics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Undistort images using camera calibration."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing images to undistort",
    )
    parser.add_argument(
        "--calib",
        type=str,
        required=True,
        help="Calibration file (npz) from calibration script",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/undistorted",
        help="Directory to save undistorted images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    image_files = glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(
        os.path.join(args.images, "*.png")
    )
    if not image_files:
        print(f"No images found in {args.images}")
        return

    camera = PinholeCamera.from_calib_file(args.calib)

    for filename in image_files:
        img = cv2.imread(filename)
        undistorted = camera.undistort(img)
        out_path = os.path.join(args.output, os.path.basename(filename))
        cv2.imwrite(out_path, undistorted)
        print(f"Saved undistorted image to {out_path}")


if __name__ == "__main__":
    main()
