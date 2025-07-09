import cv2
import argparse
import numpy as np

from lib.camera_helpers import construct_K
from geometry.pinhole_camera import PinholeCamera


def parse_args():
    parser = argparse.ArgumentParser(description="Live camera distortion.")
    parser.add_argument(
        "--calib",
        type=str,
        default="./results/camera_calib.npz",
        help="File to save calibration data (default: ./results/camera_calib.npz)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    camera = PinholeCamera.from_calib_file(args.calib)

    print(f"Using camera intrinsics from {camera.K}")
    print(f"Using camera D from {camera.D}")
    print(f"Using camera resolution: {camera.width}x{camera.height}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit("Error: Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        undistorted_frame = camera.undistort(frame)
        cv2.imshow("Undistorted Frame", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
