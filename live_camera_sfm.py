import cv2
import time
import numpy as np

from config.vio_parameters import Parameters
from geometry.pinhole_camera import PinholeCamera
from pose_estimator.feature_tracker import FeatureTracker
from pose_estimator.feature_manager import FeatureManager
from pose_estimator.factors.sfm import GlobalSFM, SFMFeature  # your converted class
from collections import defaultdict


def test_using_live_cam_and_sfm():
    # Load camera calibration
    parameters = Parameters("./data/euroc_mh_01_easy_evaluation.yaml")
    camera = PinholeCamera.from_calib_file("./results/camera_calib.npz")
    feature_manager = FeatureManager(parameters)
    feature_tracker = FeatureTracker(parameters)
    feature_tracker.camera = camera  # Set the camera in the tracker
    global_sfm = GlobalSFM()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit("Error: Could not open camera.")

    max_frames = 10
    tracked_features = defaultdict(list)
    frame_id = 0
    frame_timestamps = []
    sfm_features = {}

    print("[INFO] Capturing frames... Press Q to stop early.")
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()
        frame_timestamps.append(timestamp)

        feature_frame = feature_tracker.track_image(timestamp, frame)
        for fid, data in feature_frame.items():
            tracked_features[fid].append((frame_id, data[:2]))

        img_track = feature_tracker.get_track_image()
        cv2.putText(
            img_track,
            f"Frame {frame_id+1}/{max_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Feature Tracking", img_track)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Extracting matched features for SFM...")

    pts1 = []
    pts2 = []
    for fid, obs in tracked_features.items():
        if len(obs) >= 2 and obs[0][0] == 0 and obs[-1][0] == max_frames - 1:
            pts1.append(obs[0][1])
            pts2.append(obs[-1][1])
            sfm_features[fid] = SFMFeature(
                state=False, id=fid, observation=obs, position=np.zeros(3), depth=0.0
            )

    if len(pts1) < 10:
        print("[ERROR] Not enough correspondences between first and last frame.")
        return

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    min_len = min(len(pts1), len(pts2))
    pts1 = pts1[:min_len]
    pts2 = pts2[:min_len]

    # print("pts1: ", pts1)
    # print("pts2: ", pts2)

    print("[INFO] Estimating relative pose (essential matrix)...")
    E, mask = cv2.findEssentialMat(
        pts1, pts2, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, threshold=1.0
    )
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

    print("[INFO] Running Global SFM...")
    frame_num = max_frames
    rotations = [np.eye(3) for _ in range(frame_num)]
    translations = [np.zeros(3) for _ in range(frame_num)]
    sfm_tracked_points = {}

    success = global_sfm.construct(
        frame_num=frame_num,
        rotations=rotations,
        translations=translations,
        anchor_frame_index=0,
        relative_rotation=R,
        relative_translation=t.flatten(),
        sfm_features=list(sfm_features.values()),
        sfm_tracked_points=sfm_tracked_points,
    )

    if not success:
        print("[ERROR] Global SFM failed.")
        return

    print("[SUCCESS] Global SFM completed!")
    for i in range(frame_num):
        print(f"\nFrame {i} Pose:")
        print("Rotation:\n", rotations[i])
        print("Translation:\n", translations[i])

    print(f"\nTotal triangulated points: {len(sfm_tracked_points)}")


if __name__ == "__main__":
    test_using_live_cam_and_sfm()
