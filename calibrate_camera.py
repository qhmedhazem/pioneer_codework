import argparse
import glob
import os
import cv2
import numpy as np

from lib.camera_helpers import construct_K
from geometry.pinhole_camera import PinholeCamera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using chessboard images."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing chessboard images for calibration",
    )
    parser.add_argument(
        "--pattern_rows",
        type=int,
        default=9,
        help="Number of inner corners per chessboard row (default: 9)",
    )
    parser.add_argument(
        "--pattern_cols",
        type=int,
        default=6,
        help="Number of inner corners per chessboard column (default: 6)",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=1.0,
        help="Size of a chessboard square (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/camera_calib.npz",
        help="File to save calibration data (default: ./results/camera_calib.npz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Camera device (optional, not used in this script)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no image display)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_files = glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(
        os.path.join(args.images, "*.png")
    )
    if not image_files:
        print(f"No images found in {args.images}")
        return

    images = []
    for filename in image_files:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    if not args.headless:
        cv2.destroyAllWindows()

    pattern_size = (args.pattern_rows, args.pattern_cols)
    shape = images[0].shape

    camera = PinholeCamera(construct_K(), res=shape)
    K, dist = camera.calibrate(
        images, pattern_size, args.square_size, args.device, args.headless
    )

    print(f"Calibration complete.")

    if args.output:
        np.savez(args.output, K=K, D=dist, res=shape)
        print(f"Calibration data saved to {args.output}")


if __name__ == "__main__":
    main()
