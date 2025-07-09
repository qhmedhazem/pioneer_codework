import numpy as np
from typing import Dict
from pose_estimator.factors.image_frame import ImageFrame
from scipy.spatial.transform import Rotation as R

WINDOW_SIZE = 10
G = np.array([0.0, 0.0, 9.81])


def solve_gyroscope_bias(
    all_image_frames: Dict[float, ImageFrame], bias_gyroscope_list: list[np.ndarray]
):
    A = np.zeros((3, 3))
    b = np.zeros(3)

    timestamps = sorted(all_image_frames.keys())
    for i in range(len(timestamps) - 1):
        frame_i = all_image_frames[timestamps[i]]
        frame_j = all_image_frames[timestamps[i + 1]]

        q_ij = frame_i.rotation_matrix.T @ frame_j.rotation_matrix
        q_ij_quat = np.linalg.inv(frame_j.pre_integration.delta_rotation_matrix) @ q_ij

        tmp_A = frame_j.pre_integration.jacobian[3:6, 12:15]
        tmp_b = 2.0 * rotation_matrix_to_rotation_vector(q_ij_quat)

        A += tmp_A.T @ tmp_A
        b += tmp_A.T @ tmp_b

    delta_bg = np.linalg.solve(A, b)
    print(f"[Gyro Bias Init] Δbg = {delta_bg}")

    for i in range(WINDOW_SIZE + 1):
        bias_gyroscope_list[i] += delta_bg

    for i in range(len(timestamps) - 1):
        frame_j = all_image_frames[timestamps[i + 1]]
        frame_j.pre_integration.repropagate(np.zeros(3), bias_gyroscope_list[0])


def rotation_matrix_to_rotation_vector(R_mat):
    """Convert rotation matrix to axis-angle vector"""
    return R.from_matrix(R_mat).as_rotvec()


def tangent_basis(gravity_vector: np.ndarray) -> np.ndarray:
    """
    Construct a 3x2 matrix with two vectors orthogonal to the gravity vector.
    """
    a = gravity_vector / np.linalg.norm(gravity_vector)
    tmp = np.array([0.0, 0.0, 1.0])

    if np.allclose(a, tmp):
        tmp = np.array([1.0, 0.0, 0.0])

    b = tmp - a * (a @ tmp)
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    return np.stack((b, c), axis=1)  # shape (3, 2)


def refine_gravity(
    all_image_frames: Dict[float, ImageFrame], gravity_vector: np.ndarray, x: np.ndarray
):
    """
    Refine initial gravity vector using IMU residual minimization.
    """
    g0 = gravity_vector / np.linalg.norm(gravity_vector) * np.linalg.norm(G)
    num_frames = len(all_image_frames)
    n_state = num_frames * 3 + 2 + 1  # 3 per frame, 2 for dg, 1 for scale

    for _ in range(4):  # 4 refinement iterations
        lxly = tangent_basis(g0)
        A = np.zeros((n_state, n_state))
        b = np.zeros(n_state)

        timestamps = sorted(all_image_frames.keys())
        for i, t in enumerate(timestamps[:-1]):
            frame_i = all_image_frames[t]
            frame_j = all_image_frames[timestamps[i + 1]]

            dt = frame_j.pre_integration.sum_delta_time

            tmp_A = np.zeros((6, 9))
            tmp_b = np.zeros(6)

            R_i = frame_i.rotation_matrix
            R_j = frame_j.rotation_matrix
            T_i = frame_i.translation_vector
            T_j = frame_j.translation_vector

            tmp_A[0:3, 0:3] = -dt * np.eye(3)
            tmp_A[0:3, 6:8] = R_i.T @ (0.5 * dt * dt * np.eye(3)) @ lxly
            tmp_A[0:3, 8] = (R_i.T @ (T_j - T_i)) / 100.0

            delta_p = frame_j.pre_integration.delta_position
            bias_term = R_i.T @ R_j @ np.zeros(3)  # TIC[0] placeholder
            tmp_b[0:3] = delta_p + bias_term - R_i.T @ (0.5 * dt * dt * g0)

            tmp_A[3:6, 0:3] = -np.eye(3)
            tmp_A[3:6, 3:6] = R_i.T @ R_j
            tmp_A[3:6, 6:8] = R_i.T @ dt * np.eye(3) @ lxly

            delta_v = frame_j.pre_integration.delta_velocity
            tmp_b[3:6] = delta_v - R_i.T @ dt * np.eye(3) @ g0

            cov_inv = np.eye(6)
            r_A = tmp_A.T @ cov_inv @ tmp_A
            r_b = tmp_A.T @ cov_inv @ tmp_b

            A[i * 3 : i * 3 + 6, i * 3 : i * 3 + 6] += r_A[:6, :6]
            b[i * 3 : i * 3 + 6] += r_b[:6]

            A[-3:, -3:] += r_A[-3:, -3:]
            b[-3:] += r_b[-3:]

            A[i * 3 : i * 3 + 6, -3:] += r_A[:6, -3:]
            A[-3:, i * 3 : i * 3 + 6] += r_A[-3:, :6]

        A *= 1000.0
        b *= 1000.0

        x[:] = np.linalg.solve(A, b)
        dg = x[-3:-1]
        g0 = (g0 + lxly @ dg).reshape(3)
        g0 = g0 / np.linalg.norm(g0) * np.linalg.norm(G)

    gravity_vector[:] = g0
