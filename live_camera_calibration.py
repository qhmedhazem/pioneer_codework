import cv2
import argparse
import numpy as np

from lib.camera_helpers import construct_K
from geometry.pinhole_camera import PinholeCamera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using live camera."
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
    return parser.parse_args()


def find_camera_chessboard_corners(frame, pattern_rows=9, pattern_cols=6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern_size = (pattern_cols - 1, pattern_rows - 1)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found:
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
        return gray, corners

    return None


def main():

    args = parse_args()

    pattern_size = (args.pattern_rows, args.pattern_cols)
    cap = cv2.VideoCapture(0)
    images = []

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    objpoints = []
    imgpoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        data = find_camera_chessboard_corners(
            frame, pattern_rows=args.pattern_rows, pattern_cols=args.pattern_cols
        )

        if data is not None:
            cv2.imshow("Detected Chessboard", frame)
            print(
                "Chessboard detected. Press 'y' to confirm and save this image, or any other key to skip."
            )
            key = cv2.waitKey(0)
            cv2.destroyWindow("Detected Chessboard")
            if key != ord("y"):
                continue

        if data is not None:
            gray, corners = data
            objpoints.append(objp)
            imgpoints.append(corners.squeeze())
            images.append(gray)

        if len(objpoints) >= 10:
            break

        cv2.imshow("Live Camera Feed", frame)
        if cv2.waitKey(1) == 27:
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

    shape = images[0].shape
    camera = PinholeCamera(K=construct_K(), res=(shape[0], shape[1]))
    K, dist = camera.calibrate(
        images, pattern_size, args.square_size, args.device, False
    )
    print(f"Calibration complete.")

    if args.output:
        np.savez(args.output, K=K, D=dist, res=shape)
        print(f"Calibration data saved to {args.output}")


if __name__ == "__main__":
    main()
