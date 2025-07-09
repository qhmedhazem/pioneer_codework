import cv2
from geometry.pinhole_camera import PinholeCamera
from pose_estimator.feature_tracker import FeatureTracker


def test_using_live_cam():
    cap = cv2.VideoCapture(0)
    camera = PinholeCamera.from_calib_file("./results/camera_calib.npz")
    feature_tracker = FeatureTracker(camera)

    if not cap.isOpened():
        exit("Error: Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # undistorted_frame = camera.undistort(frame)
        feature_tracker.track_image(frame)
        img_track = feature_tracker.get_track_image()
        cv2.imshow("Feature Tracking", img_track)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    return


test_using_live_cam()
