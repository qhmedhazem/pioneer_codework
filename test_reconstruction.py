import time
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ---- project imports ---------------------------------------------------------
from datasets.euroc_dataset import EuRoCDataset, DataType  # <- your loader
from pose_estimator.devel.lib.vinspy import Estimator
from visualization.trajectory_view import TrajectoryViewer
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

def reconstruct(prev_img, prev_pose, curr_img, curr_pose, viewer=None):
    """
    Simple visual + numeric check of camera motion between two frames.
    Args:
        prev_img  : uint8 grayscale     shape (H,W)
        prev_pose : 4x4 float64         T_w_c0
        curr_img  : uint8 grayscale     shape (H,W)
        curr_pose : 4x4 float64         T_w_c1
        viewer    : TrajectoryViewer    (optional) live 3-D path plot
    """
    if prev_img is None or prev_pose is None:
        return  # very first frame

    if np.allclose(prev_pose, curr_pose, atol=1e-6):
        print("[reconstruct] Skipping duplicate pose")
        return

    # ------------------------------------------------------------------
    # 1.  ΔPose in the *camera-0* coordinate frame
    # ------------------------------------------------------------------
    T_0w = np.linalg.inv(prev_pose)
    T_01 = T_0w @ curr_pose           # transform from cam0 to cam1
    R_01 = T_01[:3, :3]
    t_01 = T_01[:3, 3]

    # translation magnitude & rotation angle
    trans_mag = np.linalg.norm(t_01)
    rot_deg   = np.rad2deg(cv2.Rodrigues(R_01)[0].ravel().sum())

    print(f"[reconstruct] Δt = {trans_mag:.3f} m   Δθ = {rot_deg:.2f} °")

    # ------------------------------------------------------------------
    # 2.  Epipolar geometry check  (feature matches coloured by error)
    # ------------------------------------------------------------------
    orb = cv2.ORB_create(2000)
    kps0, des0 = orb.detectAndCompute(prev_img, None)
    kps1, des1 = orb.detectAndCompute(curr_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)

    # build 2-D–2-D correspondences
    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kps1[m.trainIdx].pt  for m in matches])

    # fundamental matrix from VIO poses (intrinsics hard-coded here – adjust!)
    K = np.array([[458.654, 0, 367.215],
                  [0, 457.296, 248.375],
                  [0, 0, 1]], dtype=np.float64)

    # Essential from poses
    t_skew = np.array([[      0, -t_01[2],  t_01[1]],
                       [ t_01[2],       0, -t_01[0]],
                       [-t_01[1],  t_01[0],       0]])
    E = t_skew @ R_01
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

    # compute symmetric epipolar distance
    lines1 = (F @ np.concatenate([pts0, np.ones((len(pts0),1))], axis=1).T).T
    num = np.abs(np.sum(lines1 * np.concatenate([pts1, np.ones((len(pts1),1))], axis=1), axis=1))

    # Avoid division by zero
    denom = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    denom[denom < 1e-6] = 1e-6  # small epsilon to avoid zero division

    epi_err = num / denom
    mean_epi = np.mean(epi_err)
    print(f"[reconstruct] mean symmetric epipolar error = {mean_epi:.2f} px")

    # colour-code matches by error (green good, red bad)
    good_thresh = 2.0
    out = cv2.drawMatches(
        prev_img, kps0, curr_img, kps1, matches[:200], None,
        matchColor=(0,255,0), singlePointColor=(0,0,255) )
    bad_idx = epi_err > good_thresh
    for i, m in enumerate(matches[:200]):
        col = (0,0,255) if bad_idx[i] else (0,255,0)
        cv2.line(out,
                 (int(kps0[m.queryIdx].pt[0]), int(kps0[m.queryIdx].pt[1])),
                 (int(kps1[m.trainIdx].pt[0]+prev_img.shape[1]), int(kps1[m.trainIdx].pt[1])),
                 col, 1)

    cv2.imshow("Epipolar check (green=OK, red=bad)", out)
    cv2.waitKey(1)

    # ------------------------------------------------------------------
    # 3.  Optional: update live 3-D path plot
    # ------------------------------------------------------------------
    if viewer is not None:
        viewer.add_pose(curr_pose)



def run_on_euroc():
    # must be a 4x4 transformation matrix
    # visualizer = PoseVisualizer()
    vio = Estimator("./config/euroc/euroc_mono_imu_config.yaml")
    viewer = TrajectoryViewer()

    class_attributes = list(name for name in dir(vio) if not name.startswith('__'))
    print(class_attributes)

    dataset = EuRoCDataset(str(DATA_DIR))
    dataset.start()

    initalized = vio.initialize()

    last_img = None
    last_pose = None
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
                cv2.imshow("Tracking", track_img)
            else:
                track_img = np.zeros((480, 640, 3), dtype=np.uint8)                
            
            cv2.waitKey(10)
        elif dtype == DataType.IMU:
            ts = data.timestamp
            angular_velocity = data.angular_velocity
            linear_acceleration = data.linear_acceleration
            vio.input_imu(ts, linear_acceleration.copy(), angular_velocity.copy())

        print("Receving pose matrix")
        ret_pose, pose_matrix = vio.get_pose_in_world_frame()
        if ret_pose:
            print(f"Pose Matrix:\n{pose_matrix}")
        else:
            continue

        if dtype == DataType.CAMERA:
            if last_img is not None and last_pose is not None:
                reconstruct(last_img, last_pose, img, pose_matrix, viewer)
                # shift the frame history
                viewer.add_pose(pose_matrix)
                print("Received pose matrix")

            last_img  = img.copy()
            last_pose = pose_matrix.copy()



if __name__ == "__main__":
    # print(Estimator)
    run_on_euroc()